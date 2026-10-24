import pytest

from aip_predictor.fasta import records_from_text, validate_sequence


def test_records_from_fasta_text() -> None:
    records = records_from_text(">first\nACDE\n>second\nKLMN\n")
    assert [(record.identifier, record.sequence) for record in records] == [
        ("first", "ACDE"),
        ("second", "KLMN"),
    ]


def test_records_from_plain_lines() -> None:
    records = records_from_text("acde\nKLMN")
    assert [record.identifier for record in records] == ["sequence_1", "sequence_2"]
    assert [record.sequence for record in records] == ["ACDE", "KLMN"]


def test_invalid_amino_acid_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported amino-acid"):
        validate_sequence("ACDX")

