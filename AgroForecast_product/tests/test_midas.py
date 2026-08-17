"""Проверка корректности MIDAS на синтетических данных с известным ответом."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.midas import (  # noqa: E402
    PanelMIDAS,
    almon_weights,
    beta_weights,
    bootstrap_weights,
)


def test_weights_normalised() -> None:
    for K in (1, 3, 6, 12, 24):
        for fn, th in ((almon_weights, np.array([1.5, -3.0])),
                       (beta_weights, np.array([0.5, 1.2]))):
            w = fn(th, K)
            assert w.shape == (K,), f"{fn.__name__}: форма {w.shape}"
            assert abs(w.sum() - 1.0) < 1e-10, f"{fn.__name__}: сумма {w.sum()}"
            assert (w >= 0).all(), f"{fn.__name__}: отрицательные веса"
    # Устойчивость к экстремальным параметрам (переполнение exp)
    for th in ([500.0, 500.0], [-500.0, -500.0], [0.0, 0.0]):
        assert np.isfinite(almon_weights(np.array(th), 12)).all()
        assert np.isfinite(beta_weights(np.array(th), 12)).all()
    print("  ok: веса нормированы, неотрицательны, устойчивы")


def _make_panel(K=12, n_reg=40, n_yr=20, beta_true=3.0, noise=0.25, seed=0):
    """Панель с ИЗВЕСТНОЙ бета-весовой функцией."""
    rng = np.random.default_rng(seed)
    theta_true = np.array([[0.7, 1.4]])
    w_true = beta_weights(theta_true[0], K)
    N = n_reg * n_yr
    groups = np.repeat([f"R{i:02d}" for i in range(n_reg)], n_yr)
    fe = np.repeat(rng.normal(0, 4.0, n_reg), n_yr)
    X = rng.normal(0, 1, (N, 1, K))
    Z = rng.normal(0, 1, (N, 1))
    y = fe + beta_true * (X[:, 0, :] @ w_true) + 1.5 * Z[:, 0] + rng.normal(0, noise, N)
    return X, y, groups, Z, w_true, theta_true


def test_recovers_known_weights() -> None:
    X, y, groups, Z, w_true, _ = _make_panel()
    fit = PanelMIDAS(scheme="beta", n_starts=10, seed=1).fit(X, y, groups, Z)
    w_hat = fit.weights()["hf0"]
    err = np.abs(w_hat - w_true).max()
    b = fit.beta()["hf0"]
    assert err < 0.02, f"максимальное отклонение весов {err:.4f}"
    assert abs(b - 3.0) < 0.15, f"β = {b:.3f}, ожидалось 3.0"
    assert fit.r2_within > 0.98, f"R²_within = {fit.r2_within:.3f}"
    print(f"  ok: веса восстановлены (max |Δw| = {err:.4f}), β = {b:.3f}, "
          f"R²_within = {fit.r2_within:.4f}")


def test_umidas_matches_ols() -> None:
    """U-MIDAS обязан совпасть с МНК на развёрнутых лагах."""
    X, y, groups, Z, _, _ = _make_panel(K=6, n_reg=20, n_yr=15, seed=3)
    fit = PanelMIDAS(scheme="umidas").fit(X, y, groups, Z)
    N, V, K = X.shape
    D = np.hstack([X.reshape(N, V * K), Z])
    yd, Dd = y.copy(), D.copy()
    for g in np.unique(groups):
        m = groups == g
        yd[m] -= y[m].mean()
        Dd[m] -= D[m].mean(axis=0)
    coef_ref, *_ = np.linalg.lstsq(Dd, yd, rcond=None)
    assert np.abs(fit.coef - coef_ref).max() < 1e-8
    print("  ok: U-MIDAS совпал с эталонным МНК до 1e-8")


def test_parametric_beats_umidas_out_of_sample() -> None:
    """При верной параметрической форме ограничение должно помогать вне выборки."""
    X, y, groups, Z, _, _ = _make_panel(K=24, n_reg=25, n_yr=12, noise=1.5, seed=7)
    n_tr = int(len(y) * 0.7)
    tr = slice(0, n_tr)
    te = slice(n_tr, len(y))
    errs = {}
    for scheme in ("beta", "umidas"):
        m = PanelMIDAS(scheme=scheme, n_starts=8, seed=5)
        m.fit(X[tr], y[tr], groups[tr], Z[tr])
        p = m.predict(X[te], groups[te], Z[te])
        errs[scheme] = float(np.sqrt(np.mean((p - y[te]) ** 2)))
    assert errs["beta"] < errs["umidas"], errs
    print(f"  ok: RMSE вне выборки beta = {errs['beta']:.3f} < "
          f"umidas = {errs['umidas']:.3f} (K=24, экономия параметров работает)")


def test_predict_unknown_region() -> None:
    X, y, groups, Z, _, _ = _make_panel(K=6, n_reg=10, n_yr=12, seed=11)
    m = PanelMIDAS(scheme="almon", n_starts=6, seed=2)
    m.fit(X, y, groups, Z)
    p = m.predict(X[:5], np.array(["НЕИЗВЕСТНЫЙ"] * 5), Z[:5])
    assert np.isfinite(p).all(), "прогноз для нового субъекта дал NaN"
    print("  ok: прогноз для субъекта вне обучения конечен")


def test_rejects_nan() -> None:
    X, y, groups, Z, _, _ = _make_panel(K=4, n_reg=8, n_yr=10, seed=13)
    X[0, 0, 0] = np.nan
    try:
        PanelMIDAS().fit(X, y, groups, Z)
    except ValueError as exc:
        assert "пропуск" in str(exc).lower()
        print("  ok: пропуски отвергаются явной ошибкой, а не тихо заполняются")
        return
    raise AssertionError("NaN не был отвергнут")


def test_bootstrap_covers_truth() -> None:
    X, y, groups, Z, w_true, _ = _make_panel(K=8, n_reg=30, n_yr=15, noise=1.0, seed=17)
    ci = bootstrap_weights(X, y, groups, Z, scheme="beta", n_boot=60, seed=5, n_starts=3)
    inside = ((w_true >= ci["lower"][0] - 1e-9) & (w_true <= ci["upper"][0] + 1e-9)).mean()
    assert ci["n_ok"] >= 40, f"успешных итераций всего {ci['n_ok']}"
    assert inside >= 0.75, f"истинные веса покрыты лишь в {inside:.0%} точек"
    print(f"  ok: кластерный бутстрэп ({ci['n_ok']} итераций) покрывает "
          f"{inside:.0%} истинных весов")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"MIDAS — {len(tests)} проверок\n")
    for t in tests:
        print(f"* {t.__name__}")
        t()
    print("\nВсе проверки пройдены.")
