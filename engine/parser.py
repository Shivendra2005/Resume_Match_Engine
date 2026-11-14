"""Text extraction and cleaning for PDF, DOCX and TXT resumes / job descriptions."""
import io
import re
from pathlib import Path


def extract_text(source, filename: str | None = None) -> str:
    """Extract raw text from a file path or a file-like object (e.g. a Streamlit upload)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        data = path.read_bytes()
    else:
        data = source.read()
        filename = filename or getattr(source, "name", "upload.txt")

    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Normalise whitespace and strip emails/URLs/phone numbers so they don't skew similarity."""
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-()]{8,}\d", " ", text)
    text = re.sub(r"[^\w\s\+\#\.\-/]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


_PREFERRED = r"(preferred|nice to have|good to have|bonus|desirable)"
_REQUIRED = r"(required|requirements|must have|qualifications|what you.?ll need|minimum|responsibilities)"


def split_jd_sections(jd_text: str) -> dict[str, str]:
    """Heuristically split a JD into 'required' and 'preferred' blocks.
    Text before any header counts as required (it usually describes the core role)."""
    sections = {"required": [], "preferred": []}
    current = "required"
    for line in jd_text.splitlines():
        low = line.lower().strip()
        if 0 < len(low) < 60:  # header lines are short
            if re.search(_PREFERRED, low):
                current = "preferred"
            elif re.search(_REQUIRED, low):
                current = "required"
        sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}
