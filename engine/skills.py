"""Skill taxonomy and alias-aware skill extraction."""
import re

# canonical skill -> category + aliases (lowercase). Extend freely.
SKILL_TAXONOMY: dict[str, dict] = {
    # Programming
    "python": {"cat": "Programming", "aliases": ["python3"]},
    "java": {"cat": "Programming", "aliases": []},
    "c++": {"cat": "Programming", "aliases": ["cpp"]},
    "c": {"cat": "Programming", "aliases": ["embedded c"]},
    "javascript": {"cat": "Programming", "aliases": ["js", "es6"]},
    "typescript": {"cat": "Programming", "aliases": []},
    "sql": {"cat": "Programming", "aliases": ["mysql", "postgresql", "postgres", "sqlite"]},
    "matlab": {"cat": "Programming", "aliases": ["simulink"]},
    "oop": {"cat": "Programming", "aliases": ["object oriented programming", "object-oriented"]},
    # AI / ML
    "machine learning": {"cat": "AI/ML", "aliases": ["ml"]},
    "deep learning": {"cat": "AI/ML", "aliases": ["neural networks", "neural network", "cnn"]},
    "nlp": {"cat": "AI/ML", "aliases": ["natural language processing", "text mining"]},
    "computer vision": {"cat": "AI/ML", "aliases": ["image processing", "opencv"]},
    "reinforcement learning": {"cat": "AI/ML", "aliases": ["q-learning", "dqn"]},
    "llm": {"cat": "AI/ML", "aliases": ["llms", "large language models", "rag", "prompt engineering"]},
    "lstm": {"cat": "AI/ML", "aliases": ["rnn", "recurrent neural network"]},
    "transformers": {"cat": "AI/ML", "aliases": ["bert", "hugging face", "huggingface"]},
    "tensorflow": {"cat": "AI/ML", "aliases": ["keras"]},
    "pytorch": {"cat": "AI/ML", "aliases": ["torch"]},
    "scikit-learn": {"cat": "AI/ML", "aliases": ["sklearn", "scikit learn"]},
    "optimization": {"cat": "AI/ML", "aliases": ["pso", "genetic algorithm", "particle swarm"]},
    "pandas": {"cat": "Data", "aliases": []},
    "numpy": {"cat": "Data", "aliases": []},
    "data analysis": {"cat": "Data", "aliases": ["data analytics", "eda", "exploratory data analysis"]},
    "data visualization": {"cat": "Data", "aliases": ["matplotlib", "seaborn", "power bi", "tableau", "plotly"]},
    # Embedded / IoT / Wireless
    "embedded systems": {"cat": "Embedded/IoT", "aliases": ["embedded", "firmware", "microcontroller", "microcontrollers"]},
    "arduino": {"cat": "Embedded/IoT", "aliases": []},
    "raspberry pi": {"cat": "Embedded/IoT", "aliases": []},
    "iot": {"cat": "Embedded/IoT", "aliases": ["internet of things", "mqtt"]},
    "rtos": {"cat": "Embedded/IoT", "aliases": ["freertos"]},
    "uart/spi/i2c": {"cat": "Embedded/IoT", "aliases": ["uart", "spi", "i2c", "can bus"]},
    "5g/6g": {"cat": "Wireless", "aliases": ["5g", "6g", "5g nr", "b5g", "network slicing", "lte"]},
    "signal processing": {"cat": "Wireless", "aliases": ["dsp", "digital signal processing"]},
    "wireless communication": {"cat": "Wireless", "aliases": ["wireless communications", "beamforming", "mimo"]},
    # Web / Tools
    "react": {"cat": "Web", "aliases": ["react.js", "reactjs"]},
    "html/css": {"cat": "Web", "aliases": ["html", "css", "tailwind"]},
    "node.js": {"cat": "Web", "aliases": ["nodejs", "express"]},
    "rest api": {"cat": "Web", "aliases": ["rest apis", "restful", "fastapi", "flask"]},
    "git": {"cat": "Tools", "aliases": ["github", "gitlab", "version control"]},
    "docker": {"cat": "Tools", "aliases": ["kubernetes"]},
    "linux": {"cat": "Tools", "aliases": ["bash", "shell scripting", "unix"]},
    "cloud": {"cat": "Tools", "aliases": ["aws", "azure", "gcp", "google cloud"]},
    # Soft skills
    "communication": {"cat": "Soft skills", "aliases": ["communication skills"]},
    "teamwork": {"cat": "Soft skills", "aliases": ["collaboration", "team player"]},
    "problem solving": {"cat": "Soft skills", "aliases": ["problem-solving", "analytical skills"]},
}

# Single-letter skills are only accepted inside list-like context ("C, C++, Python")
_AMBIGUOUS = {"c"}


def _compile():
    pats = {}
    for skill, meta in SKILL_TAXONOMY.items():
        terms = sorted({skill, *meta["aliases"]}, key=len, reverse=True)
        alts = "|".join(re.escape(t) for t in terms)
        # custom boundaries so 'c++', 'node.js' work and 'c' doesn't match inside words
        pats[skill] = re.compile(rf"(?<![\w+#])(?:{alts})(?![\w+#])", re.IGNORECASE)
    return pats


_PATTERNS = _compile()
_C_LIST = re.compile(r"(?:^|[,/|•(:])\s*c\s*(?=[,/|)•]|\s+and\b|$)", re.IGNORECASE | re.MULTILINE)


# Knowing a child skill implies the parent (LSTM -> deep learning -> machine learning)
IMPLIES = {
    "lstm": ["deep learning"], "transformers": ["deep learning", "nlp"],
    "tensorflow": ["deep learning"], "pytorch": ["deep learning"],
    "deep learning": ["machine learning"], "reinforcement learning": ["machine learning"],
    "scikit-learn": ["machine learning"], "computer vision": ["machine learning"],
    "nlp": ["machine learning"], "llm": ["nlp"],
    "arduino": ["embedded systems"], "raspberry pi": ["embedded systems"],
}

# longest terms first, so "wireless communication" is consumed before "communication"
_ORDER = sorted(_PATTERNS, key=lambda k: -max(len(t) for t in [k, *SKILL_TAXONOMY[k]["aliases"]]))


def extract_skills(text: str, infer: bool = True) -> dict[str, int]:
    """Return {canonical_skill: mention_count}. Matched spans are masked so overlapping
    shorter skills don't double count. With infer=True, implied parent skills are added (count 0)."""
    found = {}
    work = text
    for skill in _ORDER:
        pat = _PATTERNS[skill]
        hits = pat.findall(work)
        if skill in _AMBIGUOUS:
            hits = [h for h in hits if h.lower() != "c"] + _C_LIST.findall(work)
        if hits:
            found[skill] = len(hits)
            work = pat.sub(lambda m: " " * len(m.group()), work)
    if infer:
        frontier = list(found)
        while frontier:
            for parent in IMPLIES.get(frontier.pop(), []):
                if parent not in found:
                    found[parent] = 0
                    frontier.append(parent)
    return found


def canonicalize(text: str) -> str:
    """Replace skill aliases with a single canonical token (e.g. 'keras' -> 'tensorflow')
    so lexical similarity understands synonyms."""
    for skill in _ORDER:
        token = re.sub(r"\W+", "_", skill)
        text = _PATTERNS[skill].sub(f" {token} ", text)
    return text


def category_of(skill: str) -> str:
    return SKILL_TAXONOMY.get(skill, {}).get("cat", "Other")
