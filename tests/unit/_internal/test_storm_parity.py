"""Cross-transport parity checks for storm simulation metrics.

Compares dispersion_width and request_amplification across sync (DX-0166),
async (DX-0167), and gRPC (DX-0168) storm tests. Each test writes a JSON
metrics file; this module reads all three and asserts they are within
acceptable bounds of each other.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_DIR = pathlib.Path(__file__).parent
_SYNC_PATH = _DIR / "_storm_parity_metrics_sync.json"
_ASYNC_PATH = _DIR / "_storm_parity_metrics_async.json"
_GRPC_PATH = _DIR / "_storm_parity_metrics_grpc.json"


def _load(path: pathlib.Path) -> dict[str, object]:
    if not path.exists():
        pytest.xfail(f"{path.name} not yet generated — run the corresponding storm test first")
    with path.open() as f:
        data: dict[str, object] = json.load(f)
    return data


def test_dispersion_widths_within_2x() -> None:
    sync_m = _load(_SYNC_PATH)
    async_m = _load(_ASYNC_PATH)
    grpc_m = _load(_GRPC_PATH)

    sync_w = float(sync_m["dispersion_width"])  # type: ignore[arg-type]
    async_w = float(async_m["dispersion_width"])  # type: ignore[arg-type]
    grpc_w = float(grpc_m["dispersion_width"])  # type: ignore[arg-type]

    assert async_w <= sync_w * 2.0, f"async dispersion {async_w:.3f} > 2x sync {sync_w:.3f}"
    assert sync_w <= async_w * 2.0, f"sync dispersion {sync_w:.3f} > 2x async {async_w:.3f}"
    assert grpc_w <= sync_w * 2.0, f"gRPC dispersion {grpc_w:.3f} > 2x sync {sync_w:.3f}"
    assert sync_w <= grpc_w * 2.0, f"sync dispersion {sync_w:.3f} > 2x gRPC {grpc_w:.3f}"


def test_amplifications_within_1_5x() -> None:
    sync_m = _load(_SYNC_PATH)
    async_m = _load(_ASYNC_PATH)
    grpc_m = _load(_GRPC_PATH)

    sync_a = float(sync_m["request_amplification"])  # type: ignore[arg-type]
    async_a = float(async_m["request_amplification"])  # type: ignore[arg-type]
    grpc_a = float(grpc_m["request_amplification"])  # type: ignore[arg-type]

    assert async_a <= sync_a * 1.5, f"async amp {async_a:.3f} > 1.5x sync {sync_a:.3f}"
    assert sync_a <= async_a * 1.5, f"sync amp {sync_a:.3f} > 1.5x async {async_a:.3f}"
    assert grpc_a <= sync_a * 1.5, f"gRPC amp {grpc_a:.3f} > 1.5x sync {sync_a:.3f}"
    assert sync_a <= grpc_a * 1.5, f"sync amp {sync_a:.3f} > 1.5x gRPC {grpc_a:.3f}"
