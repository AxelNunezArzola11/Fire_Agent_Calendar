"""Skill-harness tests (synthetic frames; numpy/xgboost only)."""

import numpy as np

import forecast_skill as skill


def _frame(n_pos=20, n_neg=40, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pos + n_neg):
        pos = i < n_pos
        rows.append({
            "hist_fire_count_24h": 3 if pos else 0,
            "hist_fire_count_7d": 5 if pos else 0,
            "temp_24h_mean": float(rng.normal(35 if pos else 22, 3)),
            "humidity_24h_mean": float(rng.normal(25 if pos else 60, 8)),
            "wind_24h_max": float(rng.normal(30, 8)),
            "precip_24h_sum": float(rng.normal(0 if pos else 4, 2)),
            "soil_moisture_now": float(rng.normal(0.2, 0.05)),
        })
    import pandas as pd

    return pd.DataFrame(rows)


class TestSkillHarness:
    def test_runs_and_bounds(self):
        out = skill.evaluate(_frame())
        assert out["eligible"] is True
        assert (out["n_pos"], out["n_neg"]) == (20, 40)
        for key in ("deterministic", "dummy_base_rate"):
            assert 0.0 <= out[key]["brier"] <= 1.0
            assert out[key]["logloss"] >= 0.0
        assert out["dummy_base_rate"]["base_rate"] == 20 / 60
        assert out["xgb_oof"]["status"] == "unavailable"
        assert "sklearn" in out["xgb_oof"]["reason"]

    def test_deterministic(self):
        first = skill.evaluate(_frame())
        second = skill.evaluate(_frame())
        assert first["deterministic"] == second["deterministic"]
        assert first["reliability_deterministic"] == \
            second["reliability_deterministic"]

    def test_engine_falls_back_without_sklearn(self):
        import forecast_engine as engine

        probs, used, clf, _ = engine._get_model_and_predictions(_frame())
        assert used == "Deterministic"
        assert clf is None
        assert len(probs) == 60
        assert bool((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_ineligible_small_frame(self):
        out = skill.evaluate(_frame(n_pos=1, n_neg=2))
        assert out["eligible"] is False
        assert "need >=" in out["reason"]

    def test_reliability_covers_all_rows(self):
        out = skill.evaluate(_frame())
        table = out["reliability_deterministic"]
        assert len(table) == 10
        assert sum(row["n"] for row in table) == 60

    def test_stratified_folds_balanced(self):
        labels = np.array([0] * 40 + [1] * 20)
        folds = skill.stratified_folds(labels, k=5, seed=42)
        for fold in range(5):
            sel = labels[folds == fold]
            assert set(sel.tolist()) == {0, 1}

    def test_out_of_fold_not_in_sample(self, monkeypatch):
        # Holdout invariant: no fold's model ever trains on the rows it
        # predicts (OOF estimates fitting quality, never in-sample fit).
        n = 30
        rng = np.random.default_rng(11)
        X = np.column_stack([np.arange(n, dtype=float),
                             rng.normal(size=n)])
        labels = np.array([0, 1] * (n // 2))
        fits: list = []
        preds: list = []

        class _SpyClf:
            def __init__(self, **kwargs):
                self._train_ids = frozenset()

            def fit(self, Xtr, ytr):
                self._train_ids = frozenset(Xtr[:, 0].tolist())
                fits.append(self._train_ids)
                return self

            def predict_proba(self, Xte):
                te_ids = frozenset(Xte[:, 0].tolist())
                preds.append(te_ids)
                assert te_ids.isdisjoint(self._train_ids)
                half = np.full(len(Xte), 0.5)
                return np.column_stack([half, half])

        monkeypatch.setattr(skill, "XGBClassifier", _SpyClf)
        oof = skill._oof_predict(X, labels, k=5, seed=42)
        assert oof.shape == (n,)
        assert sorted(i for s in preds for i in s) == list(range(n))
        assert len(fits) == 5
