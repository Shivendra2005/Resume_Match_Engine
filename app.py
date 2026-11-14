"""Streamlit web UI:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import ResumeMatcher
from engine.parser import extract_text

st.set_page_config(page_title="Resume Match Engine", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.score {font-size: 4rem; font-weight: 700; line-height: 1; margin: 0}
.chip {display:inline-block; padding:2px 10px; margin:2px; border-radius:12px; font-size:.85rem}
.ok {background:#d8f3e3; color:#14532d}
.miss {background:#fde2e1; color:#7f1d1d}
.extra {background:#e5e7eb; color:#1f2937}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_matcher():
    return ResumeMatcher()


def chips(items, cls):
    return " ".join(f'<span class="chip {cls}">{i}</span>' for i in items) or "<i>none</i>"


st.title("Resume Match Engine")
st.caption("Scores a resume against a job description with Sentence-Transformers embeddings, TF-IDF and skill-gap analysis.")

mode = st.radio("Mode", ["Single resume", "Rank multiple resumes"], horizontal=True)
left, right = st.columns(2)

with left:
    st.subheader("Resume")
    if mode == "Single resume":
        up = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"], key="r")
        resume_text = st.text_area("…or paste text", height=220,
                                   value="" if up else Path("data/sample_resume.txt").read_text())
        resumes = {up.name: extract_text(up)} if up else {"pasted resume": resume_text}
    else:
        ups = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        resumes = {u.name: extract_text(u) for u in ups} if ups else {}

with right:
    st.subheader("Job description")
    jd_up = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd")
    jd_text = extract_text(jd_up) if jd_up else st.text_area(
        "…or paste text", height=220, value=Path("data/sample_jd.txt").read_text())

if st.button("Analyse match", type="primary", disabled=not (resumes and jd_text.strip())):
    with st.spinner("Scoring…"):
        results = get_matcher().rank(resumes, jd_text)

    if len(results) > 1:
        st.subheader("Ranking")
        st.dataframe(pd.DataFrame([{
            "Resume": n, "Score": r.overall, "Verdict": r.verdict,
            "Skills": r.breakdown["skills"], "Semantic": r.breakdown["semantic"],
            "Keywords": r.breakdown["keywords"], "Missing required": ", ".join(r.missing_required)}
            for n, r in results]), hide_index=True, width="stretch")

    for name, r in results:
        st.divider()
        st.subheader(name)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f'<p class="score">{r.overall:.0f}</p>', unsafe_allow_html=True)
            st.write(f"**{r.verdict}** out of 100")
            for k, v in r.breakdown.items():
                st.progress(v / 100, text=f"{k.capitalize()}: {v:.0f}")
        with c2:
            st.markdown("**Required skills**")
            st.markdown(chips(r.matched_required, "ok") + " " + chips(r.missing_required, "miss"), unsafe_allow_html=True)
            st.markdown("**Preferred skills**")
            st.markdown(chips(r.matched_preferred, "ok") + " " + chips(r.missing_preferred, "miss"), unsafe_allow_html=True)
            st.markdown("**Other skills on resume**")
            st.markdown(chips(r.extra_skills, "extra"), unsafe_allow_html=True)
            if r.category_coverage:
                st.bar_chart(pd.Series(r.category_coverage, name="Coverage %"))

        st.markdown("**How to improve**")
        for t in r.suggestions:
            st.markdown(f"- {t}")

        with st.expander("Requirement-by-requirement evidence"):
            st.dataframe(pd.DataFrame([{"JD requirement": q.requirement, "Best resume evidence": q.best_evidence,
                                        "Match": round(q.score * 100)} for q in r.requirements]),
                         hide_index=True, width="stretch")

        st.download_button("Download report (JSON)", json.dumps(r.to_dict(), indent=2),
                           file_name=f"{Path(name).stem}_match.json", key=f"dl-{name}")
        st.caption(f"Encoder: {r.backend}")
