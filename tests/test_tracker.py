from backend.tracker import IoUTracker, crossing_direction


def det(x, y, label="person", size=0.1):
    return {"label": label, "box": [x, y, x + size, y + size]}


def test_same_object_keeps_its_id_while_moving():
    tr = IoUTracker()
    ids = []
    for step in range(6):
        d = [det(0.1 + step * 0.03, 0.4)]
        tr.update(d)
        ids.append(d[0]["id"])
    assert len(set(ids)) == 1


def test_unique_count_only_after_min_hits():
    tr = IoUTracker(min_hits=3)
    for step in range(2):
        tr.update([det(0.1 + step * 0.01, 0.1)])
    assert tr.unique.get("person", 0) == 0
    tr.update([det(0.12, 0.1)])
    assert tr.unique["person"] == 1


def test_two_objects_get_two_ids_and_labels_do_not_mix():
    tr = IoUTracker()
    for _ in range(4):
        d = [det(0.1, 0.1), det(0.7, 0.7), det(0.4, 0.4, label="car")]
        tr.update(d)
    assert len({x["id"] for x in d}) == 3
    assert tr.unique == {"person": 2, "car": 1}


def test_lost_track_is_forgotten_after_max_age():
    tr = IoUTracker(max_age=2)
    tr.update([det(0.1, 0.1)])
    for _ in range(4):
        tr.update([])
    assert tr.tracks == []


def test_line_crossing_directions():
    line = (0.5, 0.0, 0.5, 1.0)                       # vertical line in the middle
    assert crossing_direction(line, (0.4, 0.5), (0.6, 0.5)) == "backward"
    assert crossing_direction(line, (0.6, 0.5), (0.4, 0.5)) == "forward"


def test_no_crossing_when_staying_on_one_side_or_missing_the_segment():
    line = (0.5, 0.2, 0.5, 0.6)
    assert crossing_direction(line, (0.1, 0.5), (0.3, 0.5)) is None
    assert crossing_direction(line, (0.4, 0.9), (0.6, 0.9)) is None   # below the line's end
