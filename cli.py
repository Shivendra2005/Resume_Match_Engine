"""Command-line interface.

Examples:
  python cli.py --resume data/sample_resume.txt --jd data/sample_jd.txt
  python cli.py --resume data/*.txt --jd data/sample_jd.txt --rank
  python cli.py --resume cv.pdf --jd jd.txt --json report.json
"""
import argparse
import json
from pathlib import Path

from engine import ResumeMatcher
from engine.parser import extract_text


def bar(v: float, width: int = 30) -> str:
    n = round(v / 100 * width)
    return "█" * n + "░" * (width - n)


def print_report(name, r):
    print(f"\n=== {name} ===")
    print(f"Overall: {r.overall:5.1f}/100  {bar(r.overall)}  {r.verdict}")
    for k, v in r.breakdown.items():
        print(f"  {k:<9} {v:5.1f}  {bar(v, 20)}")
    print(f"\nMatched required : {', '.join(r.matched_required) or '-'}")
    print(f"Missing required : {', '.join(r.missing_required) or '-'}")
    print(f"Matched preferred: {', '.join(r.matched_preferred) or '-'}")
    print(f"Missing preferred: {', '.join(r.missing_preferred) or '-'}")
    print("\nSuggestions:")
    for t in r.suggestions:
        print(f"  • {t}")
    print(f"\n(encoder: {r.backend})")


def main():
    ap = argparse.ArgumentParser(description="AI-Based Resume Match Engine")
    ap.add_argument("--resume", nargs="+", required=True, help="resume file(s): pdf, docx or txt")
    ap.add_argument("--jd", required=True, help="job description file")
    ap.add_argument("--rank", action="store_true", help="rank multiple resumes")
    ap.add_argument("--json", help="write full result(s) to a JSON file")
    ap.add_argument("--model", help="Sentence-Transformers model name on the Hugging Face Hub")
    ap.add_argument("--no-neural", action="store_true", help="skip the embedding model, use TF-IDF fallback")
    args = ap.parse_args()

    matcher = ResumeMatcher(use_neural=not args.no_neural, model=args.model)
    jd = extract_text(args.jd)
    resumes = {Path(p).name: extract_text(p) for p in args.resume if Path(p).resolve() != Path(args.jd).resolve()}

    results = matcher.rank(resumes, jd)
    if args.rank and len(results) > 1:
        print("\nRANKING")
        for i, (name, r) in enumerate(results, 1):
            print(f" {i}. {name:<28} {r.overall:5.1f}  {r.verdict}")
    for name, r in results:
        print_report(name, r)

    if args.json:
        Path(args.json).write_text(json.dumps({n: r.to_dict() for n, r in results}, indent=2))
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()
