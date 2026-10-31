from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PRE = "<|im_start|>user\n"
POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EOT = "<|im_end|>"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Connector(nn.Module):
    def __init__(self, d_in: int = 768, d_hidden: int = 1024, d_out: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, f: torch.Tensor, scaled: bool = True) -> torch.Tensor:
        z = self.fc2(self.act(self.fc1(f)))
        return z * self.scale if scaled else z


def pool_features(f: torch.Tensor, n_visual: int) -> torch.Tensor:
    if n_visual == 196:
        return f
    if n_visual == 49:
        b = f.shape[0]
        g = f.view(b, 14, 14, -1).permute(0, 3, 1, 2)
        return F.avg_pool2d(g, 2).permute(0, 2, 3, 1).reshape(b, 49, -1)
    raise ValueError(n_visual)


class Formatter:
    def __init__(self, tok, cfg: dict):
        self.tok = tok
        self.caps = cfg["caps"]
        self.prompts = cfg["prompts"]
        self.n_visual = int(cfg["n_visual"])
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        self.enc = enc
        self.pre = enc(PRE)
        self.caption_tail = enc("\n" + self.prompts["caption"] + POST)
        self.answer_tail = enc("\n" + self.prompts["answer"] + POST)
        self.eot = tok.convert_tokens_to_ids(EOT)
        self.placeholder = tok.convert_tokens_to_ids("<|endoftext|>")
        self.stop = [self.eot, self.placeholder]

    def check_template(self) -> dict:
        q = "How many dogs are there?"
        ref = self.tok.apply_chat_template([{"role": "user", "content": "@@IMG@@\n" + q + "\n" + self.prompts["answer"]}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
        mine = PRE + "@@IMG@@" + "\n" + q + "\n" + self.prompts["answer"] + POST
        return {"template_matches": ref == mine, "reference": ref}

    def prompt(self, ex: dict, image: bool) -> tuple[list[int], int, bool]:
        trunc = False
        if ex["question"] is None:
            body = self.caption_tail
        else:
            q = self.enc("\n" + ex["question"])
            if len(q) - 1 > int(self.caps["question"]):
                q, trunc = q[: 1 + int(self.caps["question"])], True
            body = q + self.answer_tail
        ids = self.pre + ([self.placeholder] * self.n_visual if image else []) + body
        return ids, (len(self.pre) if image else -1), trunc

    def target(self, ex: dict, with_eot: bool = True) -> tuple[list[int], bool]:
        cap = int(self.caps["caption"] if ex["question"] is None else self.caps["answer"])
        t = self.enc(ex["answer"])
        trunc = len(t) > cap
        return t[:cap] + ([self.eot] if with_eot else []), trunc


def train_batch(fmt: Formatter, examples: list[dict], image: bool = True, with_eot: bool = True, device=DEVICE) -> dict:
    seqs, starts, labels = [], [], []
    for ex in examples:
        p, s, _ = fmt.prompt(ex, image)
        t, _ = fmt.target(ex, with_eot=True)
        seqs.append(p + t)
        lab = [-100] * len(p) + (t if with_eot else t[:-1] + [-100])
        labels.append(lab)
        starts.append(s)
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), fmt.placeholder, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    lab = torch.full((len(seqs), T), -100, dtype=torch.long)
    vis = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, (s, l, st) in enumerate(zip(seqs, labels, starts)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        lab[i, :len(l)] = torch.tensor(l)
        if st >= 0:
            vis[i, st:st + fmt.n_visual] = True
    return {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device), "vis_mask": vis.to(device)}


def embed(llm, connector, batch: dict, feats: torch.Tensor | None) -> torch.Tensor:
    emb = llm.get_input_embeddings()(batch["input_ids"])
    if feats is not None and batch["vis_mask"].any():
        z = connector(feats)
        emb = emb.masked_scatter(batch["vis_mask"].unsqueeze(-1), z.to(emb.dtype))
    return emb


def per_example_nll(llm, connector, batch: dict, feats: torch.Tensor | None) -> torch.Tensor:
    emb = embed(llm, connector, batch, feats)
    h = llm.model(inputs_embeds=emb, attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    tgt = batch["labels"][:, 1:]
    mask = tgt != -100
    logits = llm.lm_head(h[:, :-1][mask]).float()
    tok_nll = F.cross_entropy(logits, tgt[mask], reduction="none")
    rows = mask.nonzero()[:, 0]
    sums = torch.zeros(tgt.shape[0], device=tok_nll.device).index_add_(0, rows, tok_nll)
    counts = mask.sum(dim=1).clamp(min=1)
    return sums / counts


@torch.no_grad()
def greedy(llm, connector, fmt: Formatter, examples: list[dict], feats: torch.Tensor | None, image: bool, max_new: int,
           device=DEVICE) -> list[dict]:
    prompts = [fmt.prompt(ex, image) for ex in examples]
    T = max(len(p) for p, _, _ in prompts)
    ids = torch.full((len(prompts), T), fmt.placeholder, dtype=torch.long)
    att = torch.zeros((len(prompts), T), dtype=torch.long)
    vis = torch.zeros((len(prompts), T), dtype=torch.bool)
    for i, (p, s, _) in enumerate(prompts):
        off = T - len(p)
        ids[i, off:] = torch.tensor(p)
        att[i, off:] = 1
        if s >= 0:
            vis[i, off + s:off + s + fmt.n_visual] = True
    batch = {"input_ids": ids.to(device), "attention_mask": att.to(device), "vis_mask": vis.to(device)}
    emb = embed(llm, connector, batch, feats if image else None)
    mask = batch["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = llm(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1)
    last = pos[:, -1]
    gen = [[] for _ in prompts]
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    for step in range(max_new):
        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                if t in fmt.stop:
                    done[i] = True
                else:
                    gen[i].append(t)
        if bool(done.all()) or step == max_new - 1:
            break
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        last = last + 1
        out = llm(input_ids=nxt[:, None], attention_mask=mask, position_ids=last[:, None], past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
    return [{"text": fmt.tok.decode(g, skip_special_tokens=True).strip(), "n_tokens": len(g), "stopped": bool(d)}
            for g, d in zip(gen, done.tolist())]


class FeatureStore:
    def __init__(self, path, index: dict, n_visual: int):
        self.arr = np.load(path, mmap_mode="r")
        self.index = {int(k): v for k, v in index.items()}
        self.n_visual = n_visual

    def get(self, image_ids: list[int], device=DEVICE) -> torch.Tensor:
        rows = [self.index[int(i)] for i in image_ids]
        f = torch.from_numpy(np.stack([self.arr[r] for r in rows]).astype(np.float32)).to(device)
        return pool_features(f, self.n_visual)
