from __future__ import annotations

import pytest

import database


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "test_news.db")
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    database.init_db()
    return path


@pytest.fixture
def stub_analyzer():
    class Stub:
        def translate_kannada_to_english(self, text: str) -> str:
            return "India won the cricket match yesterday."

        def combined_analysis(self, kannada_text: str):
            pred = {
                "is_fake": False,
                "confidence": 0.88,
                "category": "sports",
                "category_confidence": 0.7,
                "summary": "Sports-related snippet.",
                "analysis_method": "lstm_local",
                "translation_quality": "good",
            }
            return pred, self.translate_kannada_to_english(kannada_text)

        def warmup(self) -> None:
            pass

    return Stub()


@pytest.fixture
def app_client(temp_db, monkeypatch, stub_analyzer):
    import app as app_module

    monkeypatch.setattr(app_module, "analyzer", stub_analyzer)
    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    return app_module.app.test_client()


def register_and_login(client, username="testuser", password="secret123", email="t@example.com"):
    client.post(
        "/register",
        data={"username": username, "email": email, "password": password},
        follow_redirects=True,
    )
    client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


@pytest.fixture
def logged_in_client(app_client):
    register_and_login(app_client)
    return app_client
