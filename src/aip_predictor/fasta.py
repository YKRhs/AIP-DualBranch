from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class SequenceRecord:
    identifier: str
    sequence: str


def validate_sequence(sequence: str, identifier: str = "sequence") -> str:
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError(f"{identifier}: sequence is empty")
    invalid = sorted(set(cleaned).difference(STANDARD_AA))
    if invalid:
        raise ValueError(
            f"{identifier}: unsupported amino-acid letters: {', '.join(invalid)}. "
            "Only the 20 standard amino-acid letters are accepted."
        )
    return cleaned


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    identifier: str | None = None
    chunks: list[str] = []

    def finish() -> None:
        if identifier is None:
            return
        records.append(
            SequenceRecord(identifier, validate_sequence("".join(chunks), identifier))
        )

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                identifier = line[1:].strip() or f"sequence_{len(records) + 1}"
                chunks = []
            else:
                if identifier is None:
                    raise ValueError("FASTA must start with a line beginning with '>'")
                chunks.append(line)
    finish()
    if not records:
        raise ValueError("No sequences were found in the FASTA file")
    if len({record.identifier for record in records}) != len(records):
        raise ValueError("FASTA identifiers must be unique")
    return records


def records_from_text(text: str) -> list[SequenceRecord]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("No sequence was entered")
    if stripped.startswith(">"):
        records: list[SequenceRecord] = []
        identifier: str | None = None
        chunks: list[str] = []
        for raw_line in stripped.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append(
                        SequenceRecord(
                            identifier,
                            validate_sequence("".join(chunks), identifier),
                        )
                    )
                identifier = line[1:].strip() or f"sequence_{len(records) + 1}"
                chunks = []
            else:
                if identifier is None:
                    raise ValueError("Invalid FASTA text")
                chunks.append(line)
        if identifier is not None:
            records.append(
                SequenceRecord(identifier, validate_sequence("".join(chunks), identifier))
            )
        return records
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return [
        SequenceRecord(f"sequence_{index}", validate_sequence(line, f"sequence_{index}"))
        for index, line in enumerate(lines, start=1)
    ]


def write_fasta(records: list[SequenceRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.identifier}\n{record.sequence}\n")
