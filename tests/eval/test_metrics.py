"""Correctness tests for the evaluation metric implementations.

A quality harness that reports wrong scores is worse than none, so we pin
the metric math with hand-checked cases. These tests run by default and are
not gated on the ``eval`` marker.
"""
from __future__ import annotations

import math

import pytest

from tests.eval.metrics import (
    char_error_rate,
    exact_match,
    field_set_f1,
    word_error_rate,
)


# ---------------------------------------------------------------------------
# char_error_rate
# ---------------------------------------------------------------------------


def test_cer_identical_strings_is_zero():
    assert char_error_rate("hello world", "hello world") == 0.0


def test_cer_empty_reference_and_hypothesis_is_zero():
    assert char_error_rate("", "") == 0.0


def test_cer_single_substitution():
    # 1 substitution over 5 chars -> 0.2
    assert char_error_rate("abcde", "abXde") == pytest.approx(0.2)


def test_cer_total_deletion_is_one():
    assert char_error_rate("abcde", "") == 1.0


def test_cer_insertion_on_empty_reference_is_one():
    # By convention: any output for an empty reference is fully wrong.
    assert char_error_rate("", "abc") == 1.0


def test_cer_mixed_edits():
    # "kitten" -> "sitting": 3 edits (s/k, i/e, insert g) over 6 chars
    assert char_error_rate("kitten", "sitting") == pytest.approx(3 / 6)


# ---------------------------------------------------------------------------
# word_error_rate
# ---------------------------------------------------------------------------


def test_wer_identical_sequences_is_zero():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0


def test_wer_one_word_substitution():
    # 1/4 tokens wrong
    assert word_error_rate("the quick brown fox", "the quick red fox") == pytest.approx(0.25)


def test_wer_whitespace_normalized():
    assert word_error_rate("  a  b  c  ", "a b c") == 0.0


# ---------------------------------------------------------------------------
# field_set_f1
# ---------------------------------------------------------------------------


def test_f1_perfect_match():
    scores = field_set_f1(["Hypertension", "T2DM"], ["hypertension", "t2dm"])
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_f1_both_empty_is_one():
    assert field_set_f1([], None) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_f1_disjoint_is_zero():
    scores = field_set_f1(["a", "b"], ["c", "d"])
    assert scores == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_f1_partial_overlap():
    # ref={a,b,c}, hyp={b,c,d}: tp=2, precision=2/3, recall=2/3, f1=2/3
    scores = field_set_f1(["a", "b", "c"], ["b", "c", "d"])
    assert math.isclose(scores["precision"], 2 / 3)
    assert math.isclose(scores["recall"], 2 / 3)
    assert math.isclose(scores["f1"], 2 / 3)


def test_f1_hyp_missing_penalises_recall_not_precision():
    # ref={a,b}, hyp={a}: p=1.0, r=0.5, f1=2/3
    scores = field_set_f1(["a", "b"], ["a"])
    assert scores["precision"] == 1.0
    assert scores["recall"] == 0.5
    assert math.isclose(scores["f1"], 2 / 3)


def test_f1_ignores_whitespace_and_case():
    scores = field_set_f1(["  Diabetes  "], ["DIABETES"])
    assert scores["f1"] == 1.0


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


def test_exact_match_case_and_whitespace_insensitive():
    assert exact_match("  MRN-12345  ", "mrn-12345") is True


def test_exact_match_rejects_different_values():
    assert exact_match("MRN-1", "MRN-2") is False


def test_exact_match_none_equivalent_to_empty():
    assert exact_match(None, "") is True
    assert exact_match(None, None) is True
