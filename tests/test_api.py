import io


def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200 and b"Vision Lab" in res.data


def test_health_and_classes(client):
    health = client.get("/api/health").get_json()
    assert health["status"] == "ok" and health["input_size"] == 640
    classes = client.get("/api/classes").get_json()
    assert len(classes["classes"]) == 80 and "vehicles" in classes["presets"]


def test_detect_sample_photo(client, sample_bytes):
    res = client.post("/api/detect?conf=0.4", data=sample_bytes)
    body = res.get_json()
    assert res.status_code == 200
    assert body["counts"]["person"] >= 3 and body["counts"]["bus"] == 1
    assert body["total"] == len(body["detections"])
    assert "unique" not in body                       # tracking was not requested


def test_focus_preset(client, sample_bytes):
    body = client.post("/api/detect?focus=vehicles", data=sample_bytes).get_json()
    assert set(body["counts"]) == {"bus"}


def test_bad_inputs(client):
    assert client.post("/api/detect", data=b"").status_code == 400
    assert client.post("/api/detect", data=b"not an image").status_code == 400


def test_tracking_confirms_and_counts_unique_people(client, sample_bytes):
    for _ in range(4):
        body = client.post("/api/detect?track=1&session=t1", data=sample_bytes).get_json()
    assert body["unique"]["person"] >= 3
    assert all("id" in d for d in body["detections"])
    client.post("/api/session/reset?session=t1")
    after = client.post("/api/detect?track=1&session=t1", data=sample_bytes).get_json()
    assert after["unique_total"] == 0                 # fresh session, nothing confirmed yet


def test_sessions_are_isolated(client, sample_bytes):
    for _ in range(4):
        client.post("/api/detect?track=1&session=a", data=sample_bytes)
    other = client.post("/api/detect?track=1&session=b", data=sample_bytes).get_json()
    assert other["unique_total"] == 0


def test_stats_are_recorded(client, sample_bytes):
    client.post("/api/detect", data=sample_bytes)
    stats = client.get("/api/stats").get_json()
    assert stats["samples"] == 1
    assert stats["top_classes"][0]["label"] == "person"


def test_alert_events_and_csv_export(client):
    ok = client.post("/api/events", json={"kind": "alert", "label": "person", "count": 4,
                                          "detail": "4 in frame"})
    assert ok.status_code == 201
    assert client.post("/api/events", json={"kind": "alert", "label": "dragon"}).status_code == 400
    assert client.post("/api/events", json={"kind": "line", "label": "person"}).status_code == 400
    assert client.get("/api/events").get_json()[0]["label"] == "person"
    csv_text = client.get("/api/export/events.csv").data.decode()
    assert "alert" in csv_text and "person" in csv_text
    client.delete("/api/events")
    assert client.get("/api/events").get_json() == []


def test_snapshot_lifecycle(client, sample_bytes):
    res = client.post("/api/snapshots", data={"image": (io.BytesIO(sample_bytes), "s.jpg"),
                                              "summary": "5 objects"},
                      content_type="multipart/form-data")
    assert res.status_code == 201
    url = res.get_json()["url"]
    assert client.get(url).status_code == 200
    rows = client.get("/api/snapshots").get_json()
    assert rows[0]["summary"] == "5 objects"
    assert client.delete("/api/snapshots/%d" % rows[0]["id"]).status_code == 200
    assert client.delete("/api/snapshots/999").status_code == 404


def test_snapshot_rejects_non_images_and_bad_paths(client):
    res = client.post("/api/snapshots", data={"image": (io.BytesIO(b"hello"), "x.jpg")},
                      content_type="multipart/form-data")
    assert res.status_code == 400
    assert client.get("/snapshots/../../config.py").status_code == 404
