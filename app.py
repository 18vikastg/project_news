import logging
import os

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from config.settings import Settings
from database import get_db_connection, init_db
from models import User
from services.analysis.heuristics import infer_translation_quality
from services.pipeline import KannadaNewsAnalyzer

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = Settings()
app = Flask(__name__)
app.config["SECRET_KEY"] = settings.SECRET_KEY

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

init_db()

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


try:
    analyzer = KannadaNewsAnalyzer(settings)
    analyzer.warmup()
except Exception as e:
    logger.exception("Predictor init failed: %s", e)
    analyzer = None


@app.route("/health")
def health():
    ok = analyzer is not None
    return jsonify({"status": "ok" if ok else "degraded", "predictor_ready": ok}), 200 if ok else 503


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        existing_user = conn.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            (username, email),
        ).fetchone()

        if existing_user:
            flash("Username or email already exists")
            conn.close()
            return render_template("register.html")

        password_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
        conn.close()

        flash("Registration successful! Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user_data = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user_data and check_password_hash(user_data["password_hash"], password):
            user = User(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
            )
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid username or password")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db_connection()
    history = conn.execute(
        """SELECT * FROM analysis_history
           WHERE user_id = ?
           ORDER BY created_at DESC LIMIT 10""",
        (current_user.id,),
    ).fetchall()
    conn.close()

    return render_template("dashboard.html", history=history)


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    if analyzer is None:
        flash("Analysis service is not available. Check server logs and model files.")
        return redirect(url_for("dashboard"))

    kannada_text = request.form.get("kannada_text", "")

    if not kannada_text or len(kannada_text.strip()) < 5:
        flash("Please enter valid Kannada text (minimum 5 characters)")
        return redirect(url_for("dashboard"))

    try:
        prediction, english_text = analyzer.combined_analysis(kannada_text.strip())

        conn = get_db_connection()
        conn.execute(
            """INSERT INTO analysis_history
               (user_id, kannada_text, english_translation, is_fake, confidence,
                category, category_confidence, summary, analysis_method, translation_quality)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                current_user.id,
                kannada_text,
                english_text,
                prediction["is_fake"],
                prediction["confidence"],
                prediction["category"],
                prediction["category_confidence"],
                prediction["summary"],
                prediction["analysis_method"],
                prediction["translation_quality"],
            ),
        )
        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            kannada_text=kannada_text,
            english_text=english_text,
            prediction=prediction,
        )

    except Exception as e:
        logger.exception("Analyze failed: %s", e)
        flash(f"Analysis error: {str(e)}")
        return redirect(url_for("dashboard"))


@app.route("/api/translate", methods=["POST"])
@login_required
def translate_text():
    if analyzer is None:
        return (
            jsonify(
                {
                    "error": "service_unavailable",
                    "kannada_text": "",
                    "english_translation": "Model not loaded. Check server logs for Keras compatibility or missing weights.",
                    "quality": "poor",
                }
            ),
            503,
        )

    data = request.get_json(silent=True) or {}
    kannada_text = (data.get("text") or "").strip()

    if not kannada_text:
        return jsonify({"error": "No text provided"}), 400

    if len(kannada_text) < 2:
        return jsonify(
            {
                "kannada_text": kannada_text,
                "english_translation": "Please enter more text for translation",
                "quality": "poor",
            }
        )

    try:
        english_text = analyzer.translate_kannada_to_english(kannada_text)
        quality = infer_translation_quality(kannada_text, english_text)
        return jsonify(
            {
                "kannada_text": kannada_text,
                "english_translation": english_text,
                "quality": quality,
            }
        )
    except Exception as e:
        logger.exception("Translation API error: %s", e)
        return jsonify(
            {
                "kannada_text": kannada_text,
                "english_translation": "Translation temporarily unavailable",
                "quality": "poor",
            }
        )


def create_app():
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", host="0.0.0.0", port=port)
