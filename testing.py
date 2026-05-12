"""
Interactive LSTM-only tester (Kannada in, uses same preprocessing as production LSTM path
only when text is English-like; for raw Kannada use the full app or final.py pipeline).
"""

from __future__ import annotations

import sys

from services.classification.lstm_service import LSTMNewsClassifier
from config.settings import Settings


def main() -> None:
    settings = Settings()
    try:
        classifier = LSTMNewsClassifier(
            model_path=settings.LSTM_MODEL_PATH,
            tokenizer_path=settings.TOKENIZER_PATH,
            config_path=settings.MODEL_CONFIG_PATH,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Kannada News Fake Detection (LSTM on English-like tokens)")
    print("Type 'quit' to exit. For full Kn→En→LSTM use: python final.py")

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if len(user_input) < 5:
            print("Text too short.")
            continue
        prob, status = classifier.predict_fake_probability(user_input)
        is_fake = prob > 0.5
        conf = prob if is_fake else 1 - prob
        label = "FAKE" if is_fake else "ORIGINAL"
        print(f"  {label}  P(fake)={prob:.3f}  confidence={conf:.2%}  ({status})")


if __name__ == "__main__":
    main()
