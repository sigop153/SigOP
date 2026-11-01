from sigop.signals import (
    OVERFIT,
    PLATEAU,
    Decision,
    PlateauState,
    Rejected,
    classify,
    plateau,
    retire_signals,
    write_overfit,
    write_plateau,
)

TOLERANCE = 0.02


def test_train_up_val_down_past_tolerance_is_overfit():
    assert classify(+0.04, -0.05, TOLERANCE) == Decision(adopt=False, signal=OVERFIT)


def test_train_up_val_up_adopts():
    assert classify(+0.04, +0.05, TOLERANCE) == Decision(adopt=True, signal=None)


def test_train_up_val_flat_inside_tolerance_adopts():
    assert classify(+0.04, -0.01, TOLERANCE) == Decision(adopt=True, signal=None)


def test_train_flat_is_not_adopted():
    assert classify(0.0, +0.05, TOLERANCE) == Decision(adopt=False, signal=None)


def test_train_down_val_down_is_not_overfit():
    assert classify(-0.04, -0.05, TOLERANCE) == Decision(adopt=False, signal=None)


def test_tolerance_is_a_dead_band():
    assert classify(+0.04, -TOLERANCE + 1e-9, TOLERANCE).signal is None
    assert classify(+0.04, -TOLERANCE - 1e-9, TOLERANCE).signal == OVERFIT
    assert classify(+0.04, -TOLERANCE, TOLERANCE).signal == OVERFIT


def test_plateau_resets_when_train_improves():
    state = plateau(0.90, PlateauState(best_train=0.88, stale=3, signal=None), patience=5)
    assert state == PlateauState(best_train=0.90, stale=0, signal=None)


def test_plateau_counter_climbs_below_patience():
    state = plateau(0.88, PlateauState(best_train=0.88, stale=3, signal=None), patience=5)
    assert state == PlateauState(best_train=0.88, stale=4, signal=None)


def test_plateau_fires_at_patience_and_resets():
    state = plateau(0.88, PlateauState(best_train=0.88, stale=4, signal=None), patience=5)
    assert state == PlateauState(best_train=0.88, stale=0, signal=PLATEAU)


def test_overfit_outranks_plateau_and_counter_still_resets():
    state = plateau(0.88, PlateauState(best_train=0.88, stale=4, signal=OVERFIT), patience=5)
    assert state == PlateauState(best_train=0.88, stale=0, signal=OVERFIT)


def test_overfit_note_names_the_candidates_without_val_scores(tmp_path):
    path = write_overfit(tmp_path / "reports", 7, [Rejected("tag_vote", 0.04), Rejected("knn_v2", 0.02)], 50)
    text = path.read_text(encoding="utf-8")
    assert path.name == "iter07_SIGNAL_OVERFIT.md"
    assert "2 candidates" in text and "`tag_vote`: train score +4.0%" in text and "They were not adopted" in text
    assert "%" not in text.replace("+4.0%", "").replace("+2.0%", "")   # only the train deltas are numbers


def test_overfit_note_singular(tmp_path):
    text = write_overfit(tmp_path, 3, [Rejected("a", 0.02)], 50)
    assert "1 candidate from" in text.read_text(encoding="utf-8")


def test_plateau_note(tmp_path):
    path = write_plateau(tmp_path / "reports", 12, stale=5, best=0.34, incumbent="centroid_shortlist")
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert path.name == "iter12_SIGNAL_PLATEAU.md"
    assert "not improved for 5 iterations (currently 34.0% with `centroid_shortlist`)" in text


def test_retire_moves_older_signals_only(tmp_path):
    reports = tmp_path / "reports"
    write_plateau(reports, 5, 5, 0.3, "a")
    write_overfit(reports, 8, [Rejected("b", 0.01)], 50)
    assert retire_signals(reports, keep_iteration=8) == 1
    assert sorted(p.name for p in reports.glob("*.md")) == ["iter08_SIGNAL_OVERFIT.md"]
    assert (reports / ".archive" / "iter05_SIGNAL_PLATEAU.md").exists()
