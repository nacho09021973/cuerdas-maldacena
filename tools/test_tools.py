#!/usr/bin/env python3
"""
Unit tests for infra tooling that do not require heavy deps.

These tests avoid numpy/h5py imports to stay runnable in constrained envs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.assert_artifacts import main as assert_artifacts_main
from tools.stage_utils import StageContext, ensure_no_writes_outside_run


def test_assert_artifacts_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_dir": str(run_dir),
        "artifacts": {"a": "nonexistent/foo.txt"},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))

    # Call CLI entry
    rc = assert_artifacts_main(["--run-dir", str(run_dir)])
    assert rc == 2


def test_ensure_no_writes_outside_run(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "exp"
    run_root.mkdir(parents=True)
    inside = run_root / "01_stage" / "file.txt"
    inside.parent.mkdir(parents=True)
    inside.write_text("ok")

    ensure_no_writes_outside_run(run_root, [inside])

    outside = tmp_path / "outside.txt"
    outside.write_text("nope")

    with pytest.raises(ValueError):
        ensure_no_writes_outside_run(run_root, [outside])


def test_stage_context_summary_and_manifest(tmp_path: Path) -> None:
    class Args:
        experiment = "exp"

    ctx = StageContext.from_args(Args(), stage_number="01", stage_slug="demo")
    ctx.run_root = tmp_path / ctx.run_root  # relocate under tmp
    ctx.stage_dir = ctx.run_root / ctx.stage_dir.name
    ctx.stage_dir.mkdir(parents=True, exist_ok=True)

    artifact = ctx.stage_dir / "output.txt"
    artifact.write_text("data")
    ctx.record_artifact(artifact)

    ctx.write_manifest(outputs={"artifact": str(artifact.relative_to(ctx.run_root))})
    summary_path = ctx.write_summary(status="ok", exit_code=0)

    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "ok"
    assert summary["exit_code"] == 0
    assert summary["artifacts_written"] == [str(artifact.relative_to(ctx.run_root))]
