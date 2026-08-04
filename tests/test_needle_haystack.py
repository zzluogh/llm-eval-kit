import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from my_math.needle_haystack import (
    generate_hay,
    estimate_tokens,
    generate_needle,
    insert_needle,
    check_needle_found,
    create_test_case,
    mock_model_query,
    run_test_suite,
    compute_accuracy,
)


def test_generate_hay_length():
    hay = generate_hay(100)
    words = hay.split()
    assert len(words) == 100


def test_generate_hay_all_lowercase():
    hay = generate_hay(50)
    assert all('a' <= c <= 'z' or c == ' ' for c in hay)


def test_estimate_tokens():
    text = "hello world python test"
    #text = "世界你好,这是python测试"
    tokens = estimate_tokens(text)
    assert tokens >= 5
    assert isinstance(tokens, int)


def test_generate_needle_format():
    needle = generate_needle()
    assert "secret key" in needle.lower()
    assert "NEEDLE-42" in needle


def test_insert_needle_start():
    hay = "one two three four five"
    result = insert_needle(hay, "NEEDLE", 0.0)
    assert result.startswith("NEEDLE")


def test_insert_needle_end():
    hay = "one two three four five"
    result = insert_needle(hay, "NEEDLE", 1.0)
    assert result.endswith("NEEDLE")


def test_insert_needle_middle():
    hay = "one two three four five"
    result = insert_needle(hay, "NEEDLE", 0.5)
    assert result.startswith("one two")
    assert "NEEDLE" in result
    assert result.endswith("four five")


def test_check_needle_found_yes():
    assert check_needle_found("The key is NEEDLE-42", "NEEDLE-42") is True


def test_check_needle_found_no():
    assert check_needle_found("The key is XXXX", "NEEDLE-42") is False


def test_create_test_case_structure():
    case = create_test_case(200, 0.3)
    assert "num_words" in case
    assert "doc" in case
    assert "needle" in case
    assert "position" in case
    assert case["position"] == 0.3
    assert case["num_words"] == 200


def test_create_test_case_needle_present():
    case = create_test_case(200, 0.5)
    assert case["needle"] in case["doc"]


def test_mock_model_query_finds_needle():
    doc = "aaaaa bbbbb secret key is NEEDLE-42 ccccc ddddd"
    output = mock_model_query(doc)
    assert "NEEDLE-42" in output


def test_mock_model_query_not_found():
    doc = "aaaaa bbbbb ccccc ddddd"
    output = mock_model_query(doc)
    assert "couldn't find" in output.lower()


def test_run_test_suite_all_pass():
    def always_find(doc):
        return "The secret key is NEEDLE-42"
    results = run_test_suite([100, 200], [0.0, 0.5, 1.0], always_find)
    assert len(results) == 6  # 2 lengths * 3 positions
    assert all(results.values())


def test_run_test_suite_all_fail():
    def never_find(doc):
        return "No idea"
    results = run_test_suite([100], [0.0, 0.5], never_find)
    assert not any(results.values())


def test_compute_accuracy_full():
    accuracy = compute_accuracy({"a": True, "b": True, "c": True})
    assert accuracy == 1.0


def test_compute_accuracy_half():
    accuracy = compute_accuracy({"a": True, "b": False})
    assert accuracy == 0.5


def test_compute_accuracy_empty():
    accuracy = compute_accuracy({})
    assert accuracy == 0.0


def test_insert_needle_word_count_preserved():
    """插针后单词数 = 原单词数 + 针的单词数"""
    hay = "a b c d e f g h i j"  # 10 words
    needle = "secret key is NEEDLE-42"  # 4 words
    result = insert_needle(hay, needle, 0.5)
    assert len(result.split()) == 14  # 10 + 4
