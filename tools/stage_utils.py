#!/usr/bin/env python3
"""
Utilities to standardize stage execution for CUERDAS pipeline entrypoints.

This module is intentionally infra-only: it handles run directory layout,
stage summaries, manifest updates, and guardrails (no-writes gate) without
touching scientific logic.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from cuerdas_io import update_run_manifest, write_run_manifest


STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_INCOMPLETE = "incomplete"
STATUS_ERROR = "error"

EXIT_OK = 0
EXIT_WARNING = 1
EXIT_INCOMPLETE = 2
EXIT_ERROR = 3


def _iso_now() -> str:
    return datetime.utcnow().isoformat()


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return None


def _python_env() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }


@dataclass
class StageContext:
    stage_name: str
    stage_number: str
    stage_slug: str
    experiment: str
    run_root: Path
    stage_dir: Path
    started_at: str = field(default_factory=_iso_now)
    artifacts_written: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args, stage_number: str, stage_slug: str) -> "StageContext":
        experiment = getattr(args, "experiment", None) or "default_experiment"
        run_root = Path("runs") / experiment
        stage_dir = run_root / f"{stage_number}_{stage_slug}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            stage_name=f"{stage_number}_{stage_slug}",
            stage_number=stage_number,
            stage_slug=stage_slug,
            experiment=experiment,
            run_root=run_root,
            stage_dir=stage_dir,
        )

    # -------------------- artifact + manifest helpers --------------------
    def record_artifact(self, path: Path) -> None:
        path = path.resolve()
        if not str(path).startswith(str(self.run_root.resolve())):
            raise ValueError(
                f"Artifact outside run root: {path} (run_root={self.run_root})"
            )
        rel = path.relative_to(self.run_root)
        self.artifacts_written.append(str(rel))

    def write_manifest(
        self,
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Path:
        payload: Dict[str, Dict[str, str]] = {}
        if inputs:
            payload.update(inputs)
        if outputs:
            payload.update(outputs)

        meta = {
            "stage": self.stage_name,
            "git_sha": _git_sha(),
            "env": _python_env(),
        }
        if metadata:
            meta.update(metadata)

        manifest_path = self.run_root / "run_manifest.json"
        if manifest_path.exists():
            return update_run_manifest(self.run_root, payload, section="artifacts")
        return write_run_manifest(self.run_root, payload, metadata=meta)

    # ----------------------------- summaries -----------------------------
    def write_summary(
        self,
        status: str,
        exit_code: int,
        error_message: Optional[str] = None,
    ) -> Path:
        finished_at = _iso_now()
        summary = {
            "stage_name": self.stage_name,
            "status": status,
            "exit_code": exit_code,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "artifacts_written": self.artifacts_written,
        }
        if error_message:
            summary["error_message"] = error_message

        summary_path = self.stage_dir / "stage_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary_path

    # --------------------------- validations ----------------------------
    def require_within_run(self, paths: Iterable[Path]) -> None:
        base = self.run_root.resolve()
        for p in paths:
            resolved = p.resolve()
            if not str(resolved).startswith(str(base)):
                raise ValueError(
                    f"Detected write outside run root: {resolved} (run_root={base})"
                )

    def consolidate_artifacts(self, artifacts: Iterable[Path]) -> None:
        for p in artifacts:
            self.record_artifact(p)


def ensure_no_writes_outside_run(run_root: Path, candidates: Iterable[Path]) -> None:
    root = run_root.resolve()
    for path in candidates:
        resolved = path.resolve()
        if not str(resolved).startswith(str(root)):
            raise ValueError(
                f"Detected write outside run root: {resolved} (run_root={root})"
            )

