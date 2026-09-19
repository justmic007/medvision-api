"""Tests for the analysis orchestrator.

Mocks the four services so the test is fast and network/model-independent —
it verifies the orchestration logic (top-N heatmaps/literature, assembly,
disclaimer), not the underlying services (which have their own tests).
"""
from unittest.mock import MagicMock

import numpy as np
import torch

from app.services.classifier import ClassificationResult, Finding
from app.services.literature import Article
from app.services.orchestrator import AnalysisResult, Orchestrator


import pytest


@pytest.fixture(autouse=True)
def _no_mlflow(monkeypatch):
    """Stub MLflow logging so orchestrator tests never write to ./mlruns."""
    monkeypatch.setattr(
        "app.services.orchestrator.log_inference", lambda *a, **k: "test-run"
    )


def _fake_finding(name, prob, thr=0.1):
    return Finding(name=name, probability=prob, threshold=thr, present=True)


def _build_orchestrator():
    # Five present findings, prob-descending.
    findings = [
        _fake_finding("Nodule", 0.9),
        _fake_finding("Mass", 0.8),
        _fake_finding("Effusion", 0.7),
        _fake_finding("Fracture", 0.6),
        _fake_finding("Edema", 0.5),
    ]
    classification = ClassificationResult(findings=findings, model_name="test-model")

    pre = MagicMock()
    pre.process.return_value = torch.zeros(1, 1, 224, 224)

    clf = MagicMock()
    clf.predict.return_value = classification

    exp = MagicMock()
    exp.heatmap.return_value = np.zeros((224, 224), dtype=np.float32)

    ret = MagicMock()
    ret.retrieve.return_value = [
        Article(pmid="1", title="t", journal="j", year="2020")
    ]

    orch = Orchestrator(preprocessor=pre, classifier=clf, explainer=exp, retriever=ret)
    return orch, exp, ret


def test_returns_analysis_result(monkeypatch):
    # Stub the base64 renderer so no matplotlib work happens.
    monkeypatch.setattr(
        "app.services.orchestrator.render_overlay_base64", lambda *a, **k: "FAKEB64"
    )
    orch, _, _ = _build_orchestrator()
    result = orch.analyze("fake.jpg")
    assert isinstance(result, AnalysisResult)
    assert result.model_name == "test-model"
    assert result.num_present == 5
    assert "diagnostic" in result.disclaimer.lower()


def test_top_n_limits_heatmaps_and_literature(monkeypatch):
    monkeypatch.setattr(
        "app.services.orchestrator.render_overlay_base64", lambda *a, **k: "FAKEB64"
    )
    orch, exp, ret = _build_orchestrator()
    result = orch.analyze("fake.jpg", heatmap_top_n=2, literature_top_n=3)

    # Only top-2 have heatmaps; top-3 have articles.
    with_heatmap = [f for f in result.findings if f.heatmap_base64]
    with_articles = [f for f in result.findings if f.articles]
    assert len(with_heatmap) == 2
    assert len(with_articles) == 3
    # The explainer and retriever were called the right number of times.
    assert exp.heatmap.call_count == 2
    assert ret.retrieve.call_count == 3


def test_findings_below_top_n_still_reported(monkeypatch):
    monkeypatch.setattr(
        "app.services.orchestrator.render_overlay_base64", lambda *a, **k: "FAKEB64"
    )
    orch, _, _ = _build_orchestrator()
    result = orch.analyze("fake.jpg", heatmap_top_n=2, literature_top_n=2)
    # All 5 present findings are reported, even those without heatmap/literature.
    assert result.num_present == 5
    assert result.findings[-1].heatmap_base64 is None
    assert result.findings[-1].articles == []
