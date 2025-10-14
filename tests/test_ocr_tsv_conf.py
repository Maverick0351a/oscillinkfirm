from __future__ import annotations

from oscillink.ingest.ocr import _tsv_avg_conf


def test_tsv_avg_conf_basic():
    # Minimal realistic TSV: header + a few rows; negative confidences ignored
    header = "\t".join([
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    ])
    rows = [
        "\t".join(["1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "95", ""]),
        "\t".join(["5", "1", "1", "1", "1", "1", "10", "10", "100", "20", "-1", "foo"]),  # ignored
        "\t".join(["5", "1", "1", "1", "1", "2", "20", "10", "100", "20", "80", "bar"]),
        "\t".join(["5", "1", "1", "1", "1", "3", "30", "10", "100", "20", "70", "baz"]),
    ]
    tsv = "\n".join([header] + rows)
    avg = _tsv_avg_conf(tsv)
    # Average of 0.95, 0.80, 0.70 = 0.81666...
    assert avg is not None
    assert abs(avg - ((0.95 + 0.80 + 0.70) / 3.0)) < 1e-6


def test_tsv_avg_conf_empty_or_bad():
    assert _tsv_avg_conf("") is None
    # Header only -> no values
    header_only = "\t".join(["level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"])  # noqa: E501
    assert _tsv_avg_conf(header_only) is None
