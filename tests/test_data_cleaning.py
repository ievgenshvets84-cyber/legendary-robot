"""Tests für die Datenbereinigung."""

from __future__ import annotations

from training.data_cleaning import TextCleaner


def test_removes_empty_and_whitespace():
    cleaner = TextCleaner()
    lines, stats = cleaner.clean(["", "   ", "gültiger text"])
    assert lines == ["gültiger text"]
    assert stats.removed_empty == 2
    assert stats.kept == 1


def test_deduplicates():
    cleaner = TextCleaner(dedupe=True)
    lines, stats = cleaner.clean(["hallo welt", "hallo welt", "anderer text"])
    assert len(lines) == 2
    assert stats.removed_duplicate == 1


def test_collapse_whitespace():
    cleaner = TextCleaner(collapse_whitespace=True)
    lines, _ = cleaner.clean(["viel    leerraum\tund\ntabs"])
    assert lines == ["viel leerraum und tabs"]


def test_removes_control_chars():
    cleaner = TextCleaner(strip_control_chars=True)
    lines, _ = cleaner.clean(["text\x00mit\x07steuerzeichen"])
    assert lines == ["textmitsteuerzeichen"]


def test_repetitive_filter():
    cleaner = TextCleaner(max_repetition_ratio=0.5)
    lines, stats = cleaner.clean(["aaaaaaaaaaaaaaaaaaaa", "normaler satz hier"])
    assert "normaler satz hier" in lines
    assert stats.removed_repetitive == 1


def test_min_max_length():
    cleaner = TextCleaner(min_chars=3, max_chars=10)
    lines, stats = cleaner.clean(["ab", "passt", "viel zu langer text hier"])
    assert lines == ["passt"]
    assert stats.removed_short == 1
    assert stats.removed_long == 1


def test_stats_dict_consistent():
    cleaner = TextCleaner()
    _, stats = cleaner.clean(["a text", "a text", ""])
    d = stats.as_dict()
    assert d["total"] == 3
    assert d["kept"] + d["removed"] == d["total"]
