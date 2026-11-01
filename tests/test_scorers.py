from sigop.scorers import AnswerTags, exact_scorer, extract_answer, normalize, set_scorer

DIAGNOSIS = AnswerTags(open="[DIAGNOSIS]", close="[/DIAGNOSIS]",
                       fallback_pattern=r"(?:diagnosis|final diagnosis|conclusion)[:：]\s*([^\n]+)")
CHARGES = AnswerTags(open="[罪名]", close="<eoa>")


def test_extract_between_tags():
    assert extract_answer("Reasoning...\n[DIAGNOSIS] diabetes [/DIAGNOSIS]\nDone.", DIAGNOSIS) == "diabetes"


def test_extract_runs_to_end_when_close_tag_missing():
    assert extract_answer("[罪名]盗窃;诈骗", CHARGES) == "盗窃;诈骗"


def test_extract_uses_fallback_pattern_without_tags():
    assert extract_answer("Final diagnosis: Malaria\nbecause of fever cycles", DIAGNOSIS) == "Malaria"


def test_extract_returns_whole_text_without_tags_or_fallback():
    assert extract_answer("  dengue  ", CHARGES) == "dengue"


def test_normalize_case_whitespace_and_trailing_punctuation():
    assert normalize("  Common   Cold!! ") == "common cold"


def test_exact_scorer_matches_after_normalisation():
    score = exact_scorer(DIAGNOSIS)
    assert score("[DIAGNOSIS]Common cold.[/DIAGNOSIS]", "common cold").correct
    assert score("[DIAGNOSIS]common cold[/DIAGNOSIS]", "pneumonia").correct is False


def test_exact_scorer_f1_is_one_or_zero():
    score = exact_scorer(DIAGNOSIS)
    assert score("[DIAGNOSIS]diabetes[/DIAGNOSIS]", "diabetes").f1 == 1.0
    assert score("[DIAGNOSIS]diabetes[/DIAGNOSIS]", "malaria").f1 == 0.0


def test_set_scorer_requires_the_whole_set():
    score = set_scorer(CHARGES)
    assert score("[罪名]盗窃;诈骗<eoa>", "诈骗;盗窃").correct
    partial = score("[罪名]盗窃<eoa>", "盗窃;诈骗")
    assert partial.correct is False
    assert abs(partial.f1 - 2 / 3) < 1e-9          # precision 1, recall 1/2


def test_set_scorer_accepts_any_listed_separator():
    score = set_scorer(CHARGES)
    assert score("[罪名]盗窃，诈骗<eoa>", "盗窃;诈骗").correct
    assert score("[罪名]盗窃、诈骗<eoa>", "盗窃;诈骗").correct


def test_set_scorer_empty_prediction_scores_zero():
    score = set_scorer(CHARGES)
    empty = score("", "盗窃")
    assert empty.correct is False and empty.f1 == 0.0
