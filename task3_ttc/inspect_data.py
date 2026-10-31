from __future__ import annotations

import pandas as pd

from common.utils import repo_path, save_json

ROOT = "data/ATML-PA3-Data"
FILES = {
    "math": ["generations", "probability_metrics"],
    "mathvista": ["generations", "probability_metrics", "attention_metrics"],
}
FOLDER = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}


def short(v, n=400):
    if isinstance(v, str):
        return v[:n]
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        return {"len": len(v), "head": [float(x) if hasattr(x, "__float__") else str(x) for x in list(v)[:5]]}
    return v if isinstance(v, (int, float, bool, type(None))) else str(v)


def main() -> None:
    out = {}
    for name, files in FILES.items():
        info = {}
        for f in files:
            df = pd.read_parquet(repo_path(f"{ROOT}/{FOLDER[name]}/{f}.parquet"))
            info[f] = {"rows": len(df), "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                       "first_row": {c: short(df.iloc[0][c]) for c in df.columns}}
            if f == "generations":
                g = df
                info[f]["n_questions"] = int(g["sample_id"].nunique())
                info[f]["datasets"] = g["dataset"].unique().tolist()
                info[f]["sample_ids_head"] = sorted(g["sample_id"].unique().tolist())[:10]
                info[f]["null_extracted"] = int(g["extracted_answer"].isna().sum())
                info[f]["hit_max"] = int(g["hit_max_new_tokens"].sum())
                q = g.drop_duplicates("sample_id").sample(15, random_state=0)
                info[f]["questions"] = [{"sample_id": r.sample_id, "question_text": short(r.question_text, 700),
                                         "gold": r.gold_answer} for r in q.itertuples()]
                ex = g.sample(40, random_state=1)
                info[f]["extracted_vs_gold"] = [{"sample_id": r.sample_id, "rollout_id": int(r.rollout_id),
                                                 "extracted": r.extracted_answer, "gold": r.gold_answer,
                                                 "tail": r.generated_text[-200:]} for r in ex.itertuples()]
                info[f]["distinct_extracted_per_question"] = g.groupby("sample_id")["extracted_answer"].nunique().describe().to_dict()
        gen = pd.read_parquet(repo_path(f"{ROOT}/{FOLDER[name]}/generations.parquet")).head(200)
        prob = pd.read_parquet(repo_path(f"{ROOT}/{FOLDER[name]}/probability_metrics.parquet")).head(200)
        m = gen.merge(prob, on=["dataset", "sample_id", "rollout_id"])
        info["alignment_first200"] = {
            "joined": len(m),
            "len_ids_eq_logp": int((m["generated_token_ids"].map(len) == m["logp"].map(len)).sum()),
            "len_ids_eq_kl": int((m["generated_token_ids"].map(len) == m["kl_u_p"].map(len)).sum()),
            "len_ids_eq_n": int((m["generated_token_ids"].map(len) == m["n_generated_text_tokens"]).sum()),
        }
        out[name] = info
    save_json("results/task3/data_inspection.json", out)
    print("saved results/task3/data_inspection.json")


if __name__ == "__main__":
    main()
