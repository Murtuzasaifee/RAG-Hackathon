from __future__ import annotations

from app.ingestion.parser import _PageGeometry, _normalize_bbox


def test_normalize_bbox_uses_page_geometry():
    bbox = _normalize_bbox([1.0, 2.0, 5.0, 8.0], _PageGeometry(width=10.0, height=20.0))

    assert bbox == [0.1, 0.1, 0.5, 0.4]


def test_normalize_bbox_clamps_values():
    bbox = _normalize_bbox([-1.0, 2.0, 12.0, 24.0], _PageGeometry(width=10.0, height=20.0))

    assert bbox == [0.0, 0.1, 1.0, 1.0]


def test_normalize_bbox_keeps_original_without_geometry():
    bbox = [1.0, 2.0, 5.0, 8.0]

    assert _normalize_bbox(bbox, None) == bbox
