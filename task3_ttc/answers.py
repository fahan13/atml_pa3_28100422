from __future__ import annotations

import re
from functools import lru_cache

NO_ANSWER = -1


def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and x != x:
        return True
    return not str(x).strip()


def _math_verify():
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    return parse, verify, [LatexExtractionConfig(), ExprExtractionConfig()]


@lru_cache(maxsize=None)
def math_parse(s: str):
    parse, _, cfg = _math_verify()
    return parse("$" + s + "$", extraction_config=cfg, parsing_timeout=None)


def math_verify_pair(gold: str, pred: str) -> bool:
    _, verify, _ = _math_verify()
    g, p = math_parse(gold), math_parse(pred)
    if not g or not p:
        return gold.strip() == pred.strip()
    return bool(verify(g, p, timeout_seconds=None))


def math_equivalent(a: str, b: str) -> bool:
    return a == b or math_verify_pair(a, b) or math_verify_pair(b, a)


def math_key(extracted) -> str | None:
    return None if is_missing(extracted) else str(extracted).strip()


def group_math(keys: list[str | None], gold: str) -> dict:
    present = [k for k in keys if k is not None]
    distinct = sorted(set(present), key=lambda k: (-present.count(k), keys.index(k)))
    reps, gid = [], {}
    for k in distinct:
        for g, r in enumerate(reps):
            if math_equivalent(r, k):
                gid[k] = g
                break
        else:
            gid[k] = len(reps)
            reps.append(k)
    correct = {k: math_verify_pair(gold, k) for k in distinct}
    groups = [gid[k] if k is not None else NO_ANSWER for k in keys]
    is_correct = [correct[k] if k is not None else False for k in keys]
    mixed = sum(len({correct[k] for k in distinct if gid[k] == g}) > 1 for g in range(len(reps)))
    return {"group": groups, "correct": is_correct, "representatives": reps, "mixed_groups": mixed}


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def mathvista_normalize(extracted, meta: dict) -> tuple[str | None, str]:
    if is_missing(extracted):
        return None, "missing"
    e = str(extracted).strip()
    if meta["question_type"] == "multi_choice":
        choices = meta["choices"]
        found = re.findall(r"\(([a-zA-Z])\)", e)
        if found:
            e = found[0].upper()
        seq = letters(len(choices))
        if e in seq:
            return e, "ok"
        d = [levenshtein(e, c) for c in choices]
        return seq[d.index(min(d))], "similar_choice"
    at = meta["answer_type"]
    try:
        if at == "integer":
            return str(int(float(e))), "ok"
        if at == "float":
            return str(round(float(e), int(meta["precision"]))), "ok"
    except (ValueError, OverflowError, TypeError):
        return None, "unparseable"
    return e, "ok"


def mathvista_gold(gold_answer: str, meta: dict) -> str:
    if meta["question_type"] == "multi_choice":
        seq = letters(len(meta["choices"]))
        g = str(gold_answer).strip()
        if g in seq:
            assert meta["choices"][seq.index(g)] == meta["answer"], (meta["pid"], g)
            return g
        return seq[meta["choices"].index(meta["answer"])]
    return str(meta["answer"])


def group_mathvista(extracted: list, gold_answer: str, meta: dict) -> dict:
    gold = mathvista_gold(gold_answer, meta)
    norm = [mathvista_normalize(e, meta) for e in extracted]
    keys = [k for k, _ in norm]
    present = [k for k in keys if k is not None]
    distinct = sorted(set(present), key=lambda k: (-present.count(k), keys.index(k)))
    gid = {k: i for i, k in enumerate(distinct)}
    return {"group": [gid[k] if k is not None else NO_ANSWER for k in keys],
            "correct": [k is not None and k == gold for k in keys],
            "key": keys, "status": [s for _, s in norm], "gold": gold, "representatives": distinct, "mixed_groups": 0}
