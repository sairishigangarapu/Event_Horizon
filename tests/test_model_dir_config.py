"""Regression tests for F-003: MODEL_DIR configuration consistency.

The configuration strategy is:
- config.py class default: mc_v2 (trained) — used when no .env and no env var
- .env.example (tracked): mc_v2 — template for new users
- .env (gitignored, user-specific): user's explicit choice — takes precedence
- hosted_server.py: setdefault to mc_v2 — safety net for uvicorn startup
- start_review.py / verify_random_flow.py: explicit mc_v2 override
"""
import json
import os
from pathlib import Path

import pytest

from crowd_reliability.config import Settings, settings
from crowd_reliability.ml.pipeline import load_manifest


def test_class_default_model_dir_is_mc_v2():
    """J-4 regression: the Settings class default model_dir must be mc_v2 (trained).

    This checks the class definition default, not the .env-overridden effective
    value. When no .env file and no env var exist, mc_v2 should be the default.
    """
    fields = Settings.model_fields
    assert fields["model_dir"].default == "./models/candidates/mc_v2", (
        f"config.py default should be mc_v2, got: {fields['model_dir'].default}"
    )


def test_env_example_model_dir_is_mc_v2():
    """J-4 regression: .env.example template must point to mc_v2 (trained)."""
    env_example = Path(".env.example").read_text()
    assert "MODEL_DIR=./models/candidates/mc_v2" in env_example, (
        ".env.example must document mc_v2 as the default model for new users"
    )


def test_explicit_env_var_takes_precedence(monkeypatch):
    """J-4 regression: an explicit MODEL_DIR env var overrides the class default."""
    monkeypatch.setenv("MODEL_DIR", "./models/production")
    # Use _env_file=None to bypass .env, testing pure env-var precedence
    s = Settings(_env_file=None)
    assert s.model_dir == "./models/production"


def test_mc_v2_manifest_is_trained():
    """J-4 verification: the mc_v2 manifest has trained/synthetic_validated status."""
    manifest = load_manifest("./models/candidates/mc_v2")
    assert manifest.tree_version == "dt_bank_context_v3"
    statuses = manifest.training_status
    assert all(v in ("synthetic_validated", "trained") for v in statuses.values()), (
        f"Expected all stages trained/validated, got: {statuses}"
    )


def test_production_manifest_is_untrained():
    """J-4 documentation: production models are untrained_initial.

    This test makes the configuration distinction explicit: production
    models are a separate, untrained artifact set. The class default
    must not point here for local development or demos.
    """
    manifest = load_manifest("./models/production")
    assert manifest.tree_version == "dt_bank_v2"
    statuses = manifest.training_status
    assert any("untrained" in str(v) for v in statuses.values()), (
        f"Expected some untrained stages in production, got: {statuses}"
    )


def test_effective_model_dir_reports_what_is_loaded():
    """J-4 verification: report the actual effective model directory and manifest.

    This test does not assert a specific value — it documents what .env/env
    configuration currently selects, so the effective model is always visible.
    """
    manifest = load_manifest(settings.model_dir)
    statuses = manifest.training_status
    all_trained = all(v in ("synthetic_validated", "trained") for v in statuses.values())
    print(f"\nEffective model_dir: {settings.model_dir}")
    print(f"Loaded manifest tree_version: {manifest.tree_version}")
    print(f"Training status: {statuses}")
    print(f"All stages trained/validated: {all_trained}")
    if not all_trained:
        print("WARNING: Effective model has untrained stages — check .env MODEL_DIR")
