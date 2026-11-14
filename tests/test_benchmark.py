import numpy as np
import pandas as pd

from benchmark.run_benchmark import chunk, evaluate, load_resumes, tfidf_scores, zscore


def test_chunk_covers_long_text():
    words = [f"w{i}" for i in range(400)]
    parts = chunk(" ".join(words), size=150, overlap=30)
    covered = {w for p in parts for w in p.split()}
    assert covered == set(words) and all(len(p.split()) <= 150 for p in parts)


def test_perfect_ranking_scores_one():
    labels = np.array(["A"] * 12 + ["B"] * 12)
    scores = np.zeros((24, 2)); scores[:12, 0] = 1; scores[12:, 1] = 1
    m, _ = evaluate(scores, labels, ["A", "B"])
    assert m["P@10"] == m["MRR"] == m["Top-1 acc"] == 1.0
    assert abs(m["nDCG@10"] - 1.0) < 1e-9


def test_tfidf_ranks_matching_resume_first():
    s = tfidf_scores(["python machine learning pytorch models", "teacher classroom lesson plans students",
                      "python data models", "lesson plans for students"],
                     ["machine learning engineer python", "school teacher lesson plans"])
    assert s[:, 0].argmax() == 0 and s[:, 1].argmax() == 1


def test_zscore_columns():
    z = zscore(np.array([[1.0, 10], [2, 20], [3, 30]]))
    assert np.allclose(z.mean(0), 0) and np.allclose(z.std(0), 1)


def test_load_resumes_reads_kaggle_layout(tmp_path):
    (tmp_path / "Resume").mkdir()
    long = " ".join(["word"] * 40)
    pd.DataFrame({"ID": [1, 2, 3], "Resume_str": [long, "too short", long],
                  "Resume_html": "", "Category": ["HR", "HR", "CHEF"]}).to_csv(tmp_path / "Resume/Resume.csv", index=False)
    df = load_resumes(str(tmp_path))
    assert list(df.columns) == ["text", "Category"] and len(df) == 2


def test_per_category_sampling(tmp_path):
    long = " ".join(["word"] * 40)
    pd.DataFrame({"Resume_str": [long] * 10, "Category": ["HR"] * 6 + ["CHEF"] * 4}).to_csv(tmp_path / "Resume.csv", index=False)
    df = load_resumes(str(tmp_path / "Resume.csv"), per_category=3)
    assert df["Category"].value_counts().to_dict() == {"HR": 3, "CHEF": 3}
