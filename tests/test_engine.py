from pathlib import Path

from engine import ResumeMatcher
from engine.parser import split_jd_sections
from engine.skills import canonicalize, extract_skills

DATA = Path(__file__).parent.parent / "data"
m = ResumeMatcher(use_neural=False)


def test_alias_and_masking():
    s = extract_skills("Worked on wireless communication using Keras and sklearn")
    assert "wireless communication" in s and "communication" not in s
    assert "tensorflow" in s and "scikit-learn" in s


def test_single_letter_c_only_in_lists():
    assert "c" in extract_skills("Languages: C, C++, Python")
    assert "c" not in extract_skills("I can code a cache")


def test_skill_inference():
    s = extract_skills("Built an LSTM model")
    assert {"deep learning", "machine learning"} <= set(s)


def test_canonicalize():
    assert "tensorflow" in canonicalize("keras model")


def test_jd_sections():
    sec = split_jd_sections("Requirements\nPython\nNice to have\nDocker")
    assert "Python" in sec["required"] and "Docker" in sec["preferred"]


def test_relevant_resume_ranks_higher():
    jd = (DATA / "sample_jd.txt").read_text()
    ranked = m.rank({"ml": (DATA / "sample_resume.txt").read_text(),
                     "web": (DATA / "sample_resume_2.txt").read_text()}, jd)
    assert ranked[0][0] == "ml"
    assert ranked[0][1].overall > ranked[1][1].overall + 20


def test_scores_bounded():
    r = m.match("python", "python developer needed with python skills")
    assert 0 <= r.overall <= 100
    assert all(0 <= v <= 100 for v in r.breakdown.values())
