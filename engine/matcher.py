"""Hybrid resume <-> job-description matching.

Final score = weighted blend of three signals:
  1. Skill coverage      – taxonomy-based skills required/preferred by the JD found in the resume
  2. Semantic coverage   – for every JD requirement line, best-matching resume line (sentence embeddings)
  3. Keyword similarity  – TF-IDF cosine between the whole documents (classic ATS-style signal)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .parser import clean_text, split_jd_sections
from .skills import canonicalize, category_of, extract_skills

DEFAULT_WEIGHTS = {"skills": 0.45, "semantic": 0.35, "keywords": 0.20}


# --------------------------------------------------------------------------- encoders
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _Encoder:
    """Sentence-embedding encoder with an offline fallback.

    Uses a pretrained Sentence-Transformers model from the Hugging Face Hub (runs on PyTorch).
    all-MiniLM-L6-v2 is a small 6-layer transformer that turns a sentence into a 384-number
    vector; sentences with similar meaning get vectors pointing in a similar direction.
    If the library or weights are unavailable, falls back to a skill-normalised word TF-IDF.
    `lo`/`hi` calibrate raw cosine values of each backend onto a 0–1 scale.
    """

    def __init__(self, model: str | None = None, use_neural: bool = True):
        self.backend = "tfidf-fallback"
        self.lo, self.hi = 0.0, 0.35
        self.model = None
        if use_neural:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model or DEFAULT_MODEL)
                self.backend = f"sbert:{model or DEFAULT_MODEL}"
                self.lo, self.hi = 0.20, 0.70
            except Exception as e:  # library missing, no network, bad model name
                self.load_error = str(e)
                self.model = None

    def similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        if self.model is not None:
            ea = self.model.encode(a, normalize_embeddings=True)
            eb = self.model.encode(b, normalize_embeddings=True)
            return ea @ eb.T  # cosine similarity, since vectors are unit length
        a, b = [canonicalize(x) for x in a], [canonicalize(x) for x in b]
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                              token_pattern=r"(?u)\b[\w+#/.-]{2,}\b")
        m = vec.fit_transform(a + b)
        return cosine_similarity(m[: len(a)], m[len(a):])

    def calibrate(self, x: float) -> float:
        return float(np.clip((x - self.lo) / (self.hi - self.lo), 0, 1))


# --------------------------------------------------------------------------- results
@dataclass
class RequirementMatch:
    requirement: str
    best_evidence: str
    score: float  # 0–1 calibrated


@dataclass
class MatchResult:
    overall: float
    breakdown: dict[str, float]
    matched_required: list[str]
    missing_required: list[str]
    matched_preferred: list[str]
    missing_preferred: list[str]
    extra_skills: list[str]
    requirements: list[RequirementMatch]
    suggestions: list[str]
    backend: str
    category_coverage: dict[str, float] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.overall >= 75:
            return "Strong match"
        if self.overall >= 55:
            return "Good match"
        if self.overall >= 35:
            return "Partial match"
        return "Weak match"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["requirements"] = [r.__dict__ for r in self.requirements]
        d["verdict"] = self.verdict
        return d


# --------------------------------------------------------------------------- helpers
def _lines(text: str, min_words: int = 3) -> list[str]:
    parts = re.split(r"[\n•▪●◦\u2022]|(?<=[.;])\s+", text)
    out = []
    for p in parts:
        p = re.sub(r"^\s*[-*\d.)]+\s*", "", p).strip()
        if len(p.split()) >= min_words:
            out.append(p)
    return out


_ACTION_VERBS = {"built", "developed", "designed", "implemented", "led", "optimized", "created",
                 "deployed", "trained", "improved", "reduced", "automated", "analyzed", "engineered"}


# --------------------------------------------------------------------------- engine
class ResumeMatcher:
    def __init__(self, weights: dict | None = None, use_neural: bool = True, model: str | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.encoder = _Encoder(model=model, use_neural=use_neural)

    # -- signal 1
    def _skill_score(self, resume_skills, req_skills, pref_skills):
        mr = sorted(s for s in req_skills if s in resume_skills)
        mp = sorted(s for s in pref_skills if s in resume_skills)
        req_cov = len(mr) / len(req_skills) if req_skills else 1.0
        pref_cov = len(mp) / len(pref_skills) if pref_skills else None
        score = req_cov if pref_cov is None else 0.8 * req_cov + 0.2 * pref_cov
        return score, mr, mp

    # -- signal 2
    def _semantic_coverage(self, resume: str, jd: str):
        body = jd.strip().split("\n", 1)[-1]  # drop the job-title line
        jd_lines = _lines(body, min_words=4)[:40]
        res_lines = _lines(resume, min_words=3)
        if not jd_lines or not res_lines:
            return 0.0, []
        sim = self.encoder.similarity(jd_lines, res_lines)
        best_idx = sim.argmax(axis=1)
        reqs = [
            RequirementMatch(jd_lines[i], res_lines[j], self.encoder.calibrate(sim[i, j]))
            for i, j in enumerate(best_idx)
        ]
        return float(np.mean([r.score for r in reqs])), reqs

    # -- signal 3
    @staticmethod
    def _keyword_similarity(resume: str, jd: str) -> float:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        m = vec.fit_transform([canonicalize(clean_text(jd)), canonicalize(clean_text(resume))])
        raw = float(cosine_similarity(m[0], m[1])[0, 0])
        return float(np.clip(raw / 0.30, 0, 1))  # 0.30 whole-doc cosine ≈ excellent for real resumes

    # -- suggestions
    @staticmethod
    def _suggestions(resume, missing_req, missing_pref, reqs) -> list[str]:
        tips = []
        if missing_req:
            tips.append("Add evidence for required skills you have: " + ", ".join(missing_req[:6]) + ".")
        if missing_pref:
            tips.append("Nice-to-have skills worth mentioning if true: " + ", ".join(missing_pref[:5]) + ".")
        weak = [r for r in reqs if r.score < 0.35][:3]
        for r in weak:
            tips.append(f"Weakly covered requirement: “{r.requirement[:90]}” — add a bullet that addresses it directly.")
        bullets = _lines(resume)
        if bullets:
            quantified = sum(bool(re.search(r"\d", b)) for b in bullets) / len(bullets)
            if quantified < 0.3:
                tips.append(f"Only {quantified:.0%} of lines contain numbers — quantify results (accuracy, latency, % improvement).")
            verbs = sum(b.split()[0].lower() in _ACTION_VERBS for b in bullets)
            if verbs / len(bullets) < 0.25:
                tips.append("Start more bullets with action verbs (Built, Designed, Optimized, Deployed).")
        words = len(resume.split())
        if words < 250:
            tips.append(f"Resume is short ({words} words); expand project descriptions with tools and outcomes.")
        elif words > 900:
            tips.append(f"Resume is long ({words} words); trim to the most relevant one page for internships.")
        return tips

    # -- public API
    def match(self, resume_text: str, jd_text: str) -> MatchResult:
        sections = split_jd_sections(jd_text)
        req_skills = set(extract_skills(sections["required"]))
        pref_skills = set(extract_skills(sections["preferred"])) - req_skills
        resume_skills = extract_skills(resume_text)

        s_skill, mr, mp = self._skill_score(resume_skills, req_skills, pref_skills)
        s_sem, reqs = self._semantic_coverage(resume_text, jd_text)
        s_kw = self._keyword_similarity(resume_text, jd_text)

        w = self.weights
        overall = 100 * (w["skills"] * s_skill + w["semantic"] * s_sem + w["keywords"] * s_kw)

        jd_all = req_skills | pref_skills
        cats = {}
        for s in jd_all:
            cats.setdefault(category_of(s), []).append(s in resume_skills)
        cat_cov = {c: round(100 * sum(v) / len(v), 1) for c, v in sorted(cats.items())}

        missing_req = sorted(req_skills - set(mr))
        missing_pref = sorted(pref_skills - set(mp))
        return MatchResult(
            overall=round(overall, 1),
            breakdown={"skills": round(100 * s_skill, 1), "semantic": round(100 * s_sem, 1),
                       "keywords": round(100 * s_kw, 1)},
            matched_required=mr, missing_required=missing_req,
            matched_preferred=mp, missing_preferred=missing_pref,
            extra_skills=sorted(set(resume_skills) - jd_all),
            requirements=sorted(reqs, key=lambda r: r.score),
            suggestions=self._suggestions(resume_text, missing_req, missing_pref, reqs),
            backend=self.encoder.backend,
            category_coverage=cat_cov,
        )

    def rank(self, resumes: dict[str, str], jd_text: str) -> list[tuple[str, MatchResult]]:
        """Rank many resumes against one JD (recruiter view)."""
        results = [(name, self.match(text, jd_text)) for name, text in resumes.items()]
        return sorted(results, key=lambda x: x[1].overall, reverse=True)
