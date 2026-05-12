def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["predictor_ready"] is True


def test_translate_api(logged_in_client):
    r = logged_in_client.post(
        "/api/translate",
        json={"text": "ನಮಸ್ಕಾರ"},
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 200
    data = r.get_json()
    assert "english_translation" in data
    assert "India" in data["english_translation"]


def test_analyze_saves_history(logged_in_client, temp_db):
    r = logged_in_client.post(
        "/analyze",
        data={"kannada_text": "ಕ್ರಿಕೆಟ್ ಸುದ್ದಿ ಇಲ್ಲಿ ಬರೆಯಿರಿ"},
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert b"ORIGINAL" in r.data or b"FAKE" in r.data

    import database

    conn = database.get_db_connection()
    row = conn.execute("SELECT COUNT(*) AS c FROM analysis_history").fetchone()
    conn.close()
    assert row["c"] >= 1


def test_analyze_short_text_redirect(logged_in_client):
    r = logged_in_client.post(
        "/analyze",
        data={"kannada_text": "ab"},
        follow_redirects=True,
    )
    assert r.status_code == 200


def test_translate_requires_login(app_client):
    r = app_client.post("/api/translate", json={"text": "hi"})
    assert r.status_code == 302
