"""
Interactive LSTM-only tester: Kannada in, same preprocessing as training/production.
"""

from __future__ import annotations

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

    print("Kannada News Fake Detection (LSTM on Kannada text, same as training)")
    print("Type 'quit' to exit. For full Kn→En + calibrated verdict use: python final.py")

    thr = float(settings.LSTM_FAKE_THRESHOLD)

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if len(user_input) < 5:
            print("Text too short.")
            continue
        prob, status = classifier.predict_fake_probability(user_input)
        is_fake = prob > thr
        label = "FAKE" if is_fake else "ORIGINAL"
        conf = prob if is_fake else 1 - prob
        print(f"  {label}  P(fake)={prob:.3f}  thr={thr}  confidence={conf:.2%}  ({status})")


if __name__ == "__main__":
    main()
