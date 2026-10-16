"""ML client tests: contract-shape parsing + mock service (no network)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.asyncio()
async def test_mock_predict_uses_feature_order():
    from app.services.ml_service import FEATURE_ORDER, MockMLService

    ml = MockMLService()
    feats = {k: 0.5 for k in FEATURE_ORDER}
    r = await ml.predict("CD274", "CTLA4", feats)
    assert r.prediction_score == pytest.approx(0.93)
    assert set(r.features.keys()) == set(FEATURE_ORDER)
    assert isinstance(r.feature_importance, dict)
    assert r.model.startswith("mock-")


@pytest.mark.asyncio()
async def test_mock_rank_orders_desc():
    from app.services.ml_service import FEATURE_ORDER, MockMLService

    ml = MockMLService()
    cands = [
        {"target1": "EGFR", "target2": "ERBB2", "features": {k: 0.1 for k in FEATURE_ORDER}},
        {"target1": "CD274", "target2": "CTLA4", "features": {k: 0.1 for k in FEATURE_ORDER}},
    ]
    ranked = await ml.rank_candidates(cands)
    assert [x.rank for x in ranked.results] == [1, 2]
    scores = [x.prediction_score for x in ranked.results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio()
async def test_real_client_sends_features_and_parses_dict_importance():
    import respx
    from httpx import Response

    from app.services.ml_service import FEATURE_ORDER, MLService

    feats = {k: 0.25 for k in FEATURE_ORDER}
    payload = {
        "target1": "CD274",
        "target2": "CTLA4",
        "prediction_score": 0.81,
        "rank": 0,
        "features": feats,
        "feature_importance": {"safety_score": 0.6, "double_positive": 0.4},
        "model": "xgboost_pairwise",
    }
    with respx.mock(base_url="http://ml:8001") as mock:
        route = mock.post("/predict").mock(return_value=Response(200, json=payload))
        ml = MLService("http://ml:8001", timeout=5.0)
        r = await ml.predict("cd274", "ctla4", feats)
        assert route.called
        sent = route.calls[0].request
        import json as _json

        body = _json.loads(sent.content.decode())
        assert body["target1"] == "CD274" and "features" in body  # must include features (regression)
        assert r.prediction_score == pytest.approx(0.81)
        assert r.sorted_importance()[0]["feature"] == "safety_score"
        await ml.close()


@pytest.mark.asyncio()
async def test_real_predict_rejects_nan_features_without_network():
    """Non-finite features -> structured 503 before any HTTP call (no zero-fill, no mock)."""
    from app.services.ml_service import FEATURE_ORDER, MLService, MLServiceError

    ml = MLService("http://ml:8001", timeout=5.0)
    try:
        with pytest.raises(MLServiceError) as exc:
            await ml.predict("CD274", "CTLA4", {k: float("nan") for k in FEATURE_ORDER})
        assert exc.value.status_code == 503
        assert "feature" in str(exc.value).lower()
    finally:
        await ml.close()


@pytest.mark.asyncio()
async def test_real_predict_rejects_inf_and_missing_keys():
    from app.services.ml_service import FEATURE_ORDER, MLService, MLServiceError

    ml = MLService("http://ml:8001", timeout=5.0)
    try:
        inf_feats = {k: 0.5 for k in FEATURE_ORDER}
        inf_feats["safety_score"] = float("inf")
        with pytest.raises(MLServiceError) as exc:
            await ml.predict("CD274", "CTLA4", inf_feats)
        assert exc.value.status_code == 503
        partial = {k: 0.5 for k in FEATURE_ORDER[:5]}  # missing keys
        with pytest.raises(MLServiceError) as exc2:
            await ml.predict("CD274", "CTLA4", partial)
        assert exc2.value.status_code == 503
    finally:
        await ml.close()


@pytest.mark.asyncio()
async def test_real_rank_rejects_nan_candidate_without_network():
    import respx

    from app.services.ml_service import FEATURE_ORDER, MLService, MLServiceError

    ml = MLService("http://ml:8001", timeout=5.0)
    cands = [
        {"target1": "CD274", "target2": "CTLA4", "features": {k: 0.5 for k in FEATURE_ORDER}},
        {"target1": "EGFR", "target2": "ERBB2", "features": {k: float("nan") for k in FEATURE_ORDER}},
    ]
    with respx.mock(base_url="http://ml:8001", assert_all_called=False) as mock:
        route = mock.post("/rank").mock(side_effect=AssertionError("must not call ML with NaN"))
        with pytest.raises(MLServiceError) as exc:
            await ml.rank_candidates(cands, top_k=2)
        assert exc.value.status_code == 503
        assert not route.called
    await ml.close()
