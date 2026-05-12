"""
Command-line runner for the local Kannada → English → LSTM pipeline.
Uses the same services as the Flask app (no duplicate predictor logic).
"""

from __future__ import annotations

import argparse
import sys

from config.settings import Settings
from services.pipeline import KannadaNewsAnalyzer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kannada fake news CLI (local NLLB + LSTM)")
    parser.add_argument("--text", type=str, help="Kannada news text to analyze")
    args = parser.parse_args(argv)

    settings = Settings()
    analyzer = KannadaNewsAnalyzer(settings)

    if args.text:
        texts = [args.text]
    else:
        print("Enter Kannada news text (blank line to quit).")
        texts = []
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            texts.append(line.strip())

    if not texts:
        print("No input provided.")
        return 1

    for kannada_text in texts:
        prediction, english = analyzer.combined_analysis(kannada_text)
        label = "FAKE" if prediction["is_fake"] else "ORIGINAL"
        print("---")
        print(f"English: {english}")
        print(f"Verdict: {label} (confidence {prediction['confidence']:.2%})")
        print(f"Category: {prediction['category']} ({prediction['category_confidence']:.0%})")
        print(f"Summary: {prediction['summary']}")
        print(f"Translation quality: {prediction['translation_quality']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
