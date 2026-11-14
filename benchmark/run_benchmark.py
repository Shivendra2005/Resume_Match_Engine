"""Benchmark: TF-IDF vs Sentence-Transformer embeddings for ranking resumes.

Dataset: Kaggle "Resume Dataset" (snehaanbhawal/resume-dataset) – ~2,400 real resumes,
each labelled with one of 24 job categories (HR, ENGINEERING, TEACHER, ...).

The test, in one line: for each category we write a short job description
(data/category_jds.json), rank every resume against it, and check how many of the
top-ranked resumes really belong to that category.

    python benchmark/run_benchmark.py --data path/to/Resume.csv
    python benchmark/run_benchmark.py --data path/to/Resume.csv --per-category 40   # quicker

Methods compared
  tfidf   – keyword matching: TF-IDF vectors + cosine similarity (classic ATS baseline)
  sbert   – meaning matching: all-MiniLM-L6-v2 embeddings (Hugging Face, PyTorch)
  hybrid  – average of the two (after putting both on the same scale)

Metrics (all 0–1, higher is better)
  P@10       of the 10 resumes ranked highest for a job, the share from the right category
  MRR        1 / rank of the first correct resume, averaged over jobs
  nDCG@10    like P@10 but correct resumes near the very top count more
  Top-1 acc  for each resume, is its best-matching job description its true category?
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.matcher import DEFAULT_MODEL  # noqa: E402


# --------------------------------------------------------------------------- data
def load_resumes(path: str, per_category: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Read Resume.csv (columns Resume_str, Category). `path` may be the CSV or a folder containing it."""
    p = Path(path)
    if p.is_dir():
        hits = sorted(p.rglob("Resume.csv")) or sorted(p.rglob("*.csv"))
        if not hits:
            raise FileNotFoundError(f"no CSV found under {p}")
        p = hits[0]
    df = pd.read_csv(p)
    text_col = "Resume_str" if "Resume_str" in df.columns else "Resume"
    df = df.rename(columns={text_col: "text"})[["text", "Category"]].dropna()
    df["text"] = df["text"].astype(str).map(lambda t: re.sub(r"\s+", " ", t).strip())
    df = df[df["text"].str.split().str.len() >= 30]  # drop near-empty resumes
    if per_category:
        # shuffle, then keep the first N of each category (works on every pandas version)
        df = df.sample(frac=1, random_state=seed).groupby("Category").head(per_category)
    return df.reset_index(drop=True)


def chunk(text: str, size: int = 150, overlap: int = 30) -> list[str]:
    """Resumes are longer than the model's input limit (~256 tokens), so split into word windows."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)] or [text]


# --------------------------------------------------------------------------- methods
def tfidf_scores(resumes: list[str], jds: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    vec.fit(resumes + jds)
    return cosine_similarity(vec.transform(resumes), vec.transform(jds))  # (n_resumes, n_jobs)


def sbert_scores(resumes: list[str], jds: list[str], model_name: str = DEFAULT_MODEL,
                 batch_size: int = 64) -> np.ndarray:
    """Score = similarity between the job and the resume's best-matching chunk."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    chunks, owner = [], []
    for i, r in enumerate(resumes):
        cs = chunk(r)
        chunks += cs
        owner += [i] * len(cs)
    owner = np.array(owner)
    ec = model.encode(chunks, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    ej = model.encode(jds, normalize_embeddings=True)
    sim = ec @ ej.T                                              # (n_chunks, n_jobs)
    out = np.full((len(resumes), len(jds)), -1.0)
    np.maximum.at(out, owner, sim)                               # max over each resume's chunks
    return out


def zscore(s: np.ndarray) -> np.ndarray:
    """Put scores on a common scale per job (column) so two methods can be averaged."""
    return (s - s.mean(0)) / (s.std(0) + 1e-9)


# --------------------------------------------------------------------------- metrics
def evaluate(scores: np.ndarray, labels: np.ndarray, cats: list[str], k: int = 10) -> tuple[dict, pd.Series]:
    p_at_k, rr, ndcg = [], [], []
    for j, cat in enumerate(cats):
        rel = labels[np.argsort(-scores[:, j])] == cat
        if not rel.any():
            continue
        p_at_k.append(rel[:k].mean())
        rr.append(1.0 / (np.argmax(rel) + 1))
        gains = rel[:k] / np.log2(np.arange(2, k + 2))
        ideal = (1 / np.log2(np.arange(2, min(k, rel.sum()) + 2))).sum()
        ndcg.append(gains.sum() / ideal)
    top1 = (np.array(cats)[scores.argmax(1)] == labels).mean()
    per_cat = pd.Series(p_at_k, index=[c for c in cats if (labels == c).any()], name=f"P@{k}")
    return {f"P@{k}": np.mean(p_at_k), "MRR": np.mean(rr), f"nDCG@{k}": np.mean(ndcg), "Top-1 acc": top1}, per_cat


def run(df: pd.DataFrame, jds: dict[str, str], model_name: str = DEFAULT_MODEL,
        out_dir: str | Path = "results", methods=("tfidf", "sbert", "hybrid")) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = [c for c in jds if c in set(df["Category"])]
    resumes, labels, jd_texts = df["text"].tolist(), df["Category"].to_numpy(), [jds[c] for c in cats]
    print(f"{len(resumes)} resumes, {len(cats)} job categories")

    scores, timing = {}, {}
    t = time.time(); scores["tfidf"] = tfidf_scores(resumes, jd_texts); timing["tfidf"] = time.time() - t
    if "sbert" in methods or "hybrid" in methods:
        t = time.time(); scores["sbert"] = sbert_scores(resumes, jd_texts, model_name); timing["sbert"] = time.time() - t
        scores["hybrid"] = (zscore(scores["tfidf"]) + zscore(scores["sbert"])) / 2
        timing["hybrid"] = timing["tfidf"] + timing["sbert"]

    rows, per_cat = [], {}
    for name in methods:
        m, pc = evaluate(scores[name], labels, cats)
        rows.append({"method": name, **{k: round(v, 3) for k, v in m.items()}, "seconds": round(timing[name], 1)})
        per_cat[name] = pc
    table = pd.DataFrame(rows).set_index("method")
    per_cat = pd.DataFrame(per_cat).round(2)
    table.to_csv(out_dir / "benchmark_results.csv")
    per_cat.to_csv(out_dir / "per_category_p10.csv")
    _plot(table, per_cat, out_dir)
    return table


def _plot(table: pd.DataFrame, per_cat: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"tfidf": "#8a8f98", "sbert": "#2563eb", "hybrid": "#0f766e"}
    metrics = [c for c in table.columns if c != "seconds"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(table)
    for i, (name, row) in enumerate(table.iterrows()):
        x = np.arange(len(metrics)) + i * w
        ax.bar(x, row[metrics], w, label=name, color=colors.get(name))
        for xi, v in zip(x, row[metrics]):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(metrics)) + w * (len(table) - 1) / 2, metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Resume ranking: keyword (TF-IDF) vs embeddings (Sentence-Transformers)")
    ax.legend(frameon=False, ncol=len(table), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_chart.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Resume.csv from the Kaggle dataset, or its folder")
    ap.add_argument("--jds", default=str(ROOT / "data/category_jds.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model on the Hugging Face Hub")
    ap.add_argument("--per-category", type=int, help="sample at most N resumes per category (faster)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tfidf-only", action="store_true", help="skip the embedding model")
    args = ap.parse_args()

    df = load_resumes(args.data, args.per_category)
    jds = json.loads(Path(args.jds).read_text())
    methods = ("tfidf",) if args.tfidf_only else ("tfidf", "sbert", "hybrid")
    table = run(df, jds, args.model, args.out, methods)
    print("\n" + table.to_string())
    print(f"\nSaved results to {args.out}/ (benchmark_results.csv, per_category_p10.csv, benchmark_chart.png)")


if __name__ == "__main__":
    main()
