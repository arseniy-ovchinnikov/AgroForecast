"""Панельная MIDAS-модель (Mixed Data Sampling).

Назначение
----------
Связать низкочастотную целевую переменную (годовая урожайность или
коэффициент уборки) с высокочастотными внутрисезонными регрессорами
(16-дневные NDVI/EVI, месячные метеопоказатели ERA5) БЕЗ предварительного
усреднения последних. Отказ от временнóго агрегирования — единственное
содержательное отличие MIDAS от обычной регрессии на сезонном среднем,
и именно оно проверяется гипотезой H1.

Спецификация
------------
Для субъекта i и года t:

    y_it = α_i + Σ_v β_v · Σ_{j=0}^{K-1} w_j(θ_v) · x_{v,i,t,j} + γ' z_it + ε_it

где
    x_{v,i,t,j} — значение высокочастотной переменной v в периоде j сезона t;
    w_j(θ_v)    — весовая функция, Σ_j w_j = 1, w_j ≥ 0;
    z_it        — низкочастотные контроли (лаги урожайности и т. п.);
    α_i         — фиксированный эффект субъекта.

Весовые схемы
-------------
``almon``  экспоненциальный полином Альмона, 2 параметра:
           w_j ∝ exp(θ₁·u_j + θ₂·u_j²),  u_j = (j+1)/K
``beta``   бета-функция Гизельса, 2 параметра:
           w_j ∝ u_j^(a−1)·(1−u_j)^(b−1),  a = exp(θ₁), b = exp(θ₂)
``umidas`` без ограничений: каждый лаг получает собственный коэффициент
           (U-MIDAS). Служит верхней границей гибкости и одновременно
           проверкой того, что параметрическое ограничение не вредит.

Оценивание
----------
Концентрированный нелинейный МНК: при фиксированном θ модель линейна по
(β, γ), поэтому внутренний шаг — обычный МНК на внутригрупповых отклонениях
(within-преобразование), а внешний — оптимизация только по θ (2 параметра
на переменную). Многократный старт из разных начальных точек защищает от
локальных минимумов.

Доверительные интервалы для весов — кластерный бутстрэп по субъектам:
ресемплируются целые регионы, а не отдельные наблюдения, что сохраняет
внутрирегиональную зависимость.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Весовые функции
# ---------------------------------------------------------------------------
def almon_weights(theta: np.ndarray, K: int) -> np.ndarray:
    """Экспоненциальный полином Альмона, нормированный к единице."""
    u = (np.arange(K, dtype=float) + 1.0) / K
    z = theta[0] * u + theta[1] * u * u
    z = z - z.max()                      # защита от переполнения exp
    w = np.exp(z)
    s = w.sum()
    return w / s if s > _EPS else np.full(K, 1.0 / K)


def beta_weights(theta: np.ndarray, K: int) -> np.ndarray:
    """Бета-весовая функция Гизельса, нормированная к единице.

    Параметризация a = exp(θ₁), b = exp(θ₂) гарантирует положительность
    без ограничений на области определения оптимизатора.
    """
    if K == 1:
        return np.ones(1)
    u = np.linspace(_EPS, 1.0 - _EPS, K)
    a = np.exp(np.clip(theta[0], -8.0, 8.0))
    b = np.exp(np.clip(theta[1], -8.0, 8.0))
    log_w = (a - 1.0) * np.log(u) + (b - 1.0) * np.log1p(-u)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    s = w.sum()
    return w / s if s > _EPS else np.full(K, 1.0 / K)


WEIGHT_FUNCTIONS = {"almon": almon_weights, "beta": beta_weights}
N_THETA = 2


# ---------------------------------------------------------------------------
# Результат
# ---------------------------------------------------------------------------
@dataclass
class MidasFit:
    """Оценённая модель и всё, что нужно для прогноза и интерпретации."""

    scheme: str
    hf_names: List[str]
    lf_names: List[str]
    K: int
    theta: Optional[np.ndarray]          # (V, 2) или None для umidas
    coef: np.ndarray                     # коэффициенты при [агрегаты, контроли]
    intercept: float
    group_effects: Dict[str, float] = field(default_factory=dict)
    rss: float = np.nan
    r2_within: float = np.nan
    n_obs: int = 0
    n_starts_converged: int = 0

    def weights(self) -> Dict[str, np.ndarray]:
        """Нормированные веса по каждой высокочастотной переменной."""
        if self.scheme == "umidas":
            out: Dict[str, np.ndarray] = {}
            for v, name in enumerate(self.hf_names):
                block = self.coef[v * self.K : (v + 1) * self.K]
                total = np.abs(block).sum()
                out[name] = block / total if total > _EPS else block
            return out
        fn = WEIGHT_FUNCTIONS[self.scheme]
        return {name: fn(self.theta[v], self.K) for v, name in enumerate(self.hf_names)}

    def beta(self) -> Dict[str, float]:
        """Масштабные коэффициенты β при агрегатах (для umidas — сумма блока)."""
        if self.scheme == "umidas":
            return {
                name: float(self.coef[v * self.K : (v + 1) * self.K].sum())
                for v, name in enumerate(self.hf_names)
            }
        return {name: float(self.coef[v]) for v, name in enumerate(self.hf_names)}


# ---------------------------------------------------------------------------
# Оценивание
# ---------------------------------------------------------------------------
class PanelMIDAS:
    """MIDAS с фиксированными эффектами субъектов.

    Args:
        scheme: 'almon' | 'beta' | 'umidas'.
        fixed_effects: применять внутригрупповое преобразование.
        n_starts: число случайных стартов оптимизатора (для umidas игнорируется).
        max_nfev: предел вызовов функции невязок на один старт.
        seed: зерно генератора начальных точек.
        min_periods_parametric: минимальное K, при котором двухпараметрическая
            весовая функция идентифицируема (см. ниже).

    Идентифицируемость
    ------------------
    Нормировка Σ w_j = 1 оставляет K−1 свободных весов, тогда как схемы
    ``almon`` и ``beta`` имеют 2 параметра. При K = 2 свободный вес всего один,
    и θ не идентифицируется: бесконечно много пар (θ₁, θ₂) дают одни и те же
    веса. При K = 3 задача формально точно идентифицирована, но численно
    неустойчива. Поэтому при K < ``min_periods_parametric`` (по умолчанию 4)
    оценивание параметрической схемы отклоняется явной ошибкой: для коротких
    сезонных отсечек следует применять U-MIDAS, где веса свободны.
    """

    def __init__(
        self,
        scheme: str = "beta",
        fixed_effects: bool = True,
        n_starts: int = 8,
        max_nfev: int = 4000,
        seed: int = 42,
        min_periods_parametric: int = 4,
    ) -> None:
        if scheme not in set(WEIGHT_FUNCTIONS) | {"umidas"}:
            raise ValueError(f"Неизвестная весовая схема: {scheme}")
        self.scheme = scheme
        self.fixed_effects = fixed_effects
        self.n_starts = n_starts
        self.max_nfev = max_nfev
        self.seed = seed
        self.min_periods_parametric = min_periods_parametric
        self.fit_: Optional[MidasFit] = None

    # ------------------------------------------------------------- служебное
    @staticmethod
    def _group_codes(groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Целочисленные коды групп и их размеры (для векторного демининга)."""
        labels, codes = np.unique(groups, return_inverse=True)
        counts = np.bincount(codes, minlength=labels.size).astype(float)
        return labels, codes, counts

    @staticmethod
    def _demean_fast(
        y: np.ndarray,
        D: np.ndarray,
        codes: np.ndarray,
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Внутригрупповое преобразование через bincount, без циклов по группам.

        Возвращает (y_демин., D_демин., средние y по группам, средние D по группам).
        Оптимизация принципиальна: функция вызывается на каждой итерации
        нелинейного оптимизатора, то есть тысячи раз за одну оценку.
        """
        G = counts.size
        y_mean = np.bincount(codes, weights=y, minlength=G) / counts
        D_mean = np.empty((G, D.shape[1]), dtype=float)
        for c in range(D.shape[1]):
            D_mean[:, c] = np.bincount(codes, weights=D[:, c], minlength=G) / counts
        return y - y_mean[codes], D - D_mean[codes], y_mean, D_mean

    def _aggregate(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Свёртка (N, V, K) → (N, V) с весами по каждой переменной."""
        fn = WEIGHT_FUNCTIONS[self.scheme]
        K = X.shape[2]
        out = np.empty((X.shape[0], X.shape[1]), dtype=float)
        for v in range(X.shape[1]):
            out[:, v] = X[:, v, :] @ fn(theta[v], K)
        return out

    @staticmethod
    def _ols(D: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(D, y, rcond=None)
        return coef

    # ------------------------------------------------------------------- API
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        Z: Optional[np.ndarray] = None,
        hf_names: Optional[Sequence[str]] = None,
        lf_names: Optional[Sequence[str]] = None,
    ) -> MidasFit:
        """Оценивает модель.

        Args:
            X: (N, V, K) высокочастотные регрессоры; j = 0 — начало сезона.
            y: (N,) целевая переменная.
            groups: (N,) идентификаторы субъектов для фиксированных эффектов.
            Z: (N, P) низкочастотные контроли (может быть None).
            hf_names, lf_names: имена для интерпретации результата.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"X должен быть 3-мерным (N, V, K), получено {X.shape}")
        N, V, K = X.shape
        if y.shape[0] != N:
            raise ValueError(f"Размерности X ({N}) и y ({y.shape[0]}) не совпадают")
        Z = np.zeros((N, 0)) if Z is None else np.asarray(Z, dtype=float)
        hf_names = list(hf_names) if hf_names else [f"hf{v}" for v in range(V)]
        lf_names = list(lf_names) if lf_names else [f"lf{p}" for p in range(Z.shape[1])]

        if self.scheme != "umidas" and K < self.min_periods_parametric:
            raise ValueError(
                f"Схема «{self.scheme}» неидентифицируема при K = {K}: нормировка "
                f"оставляет {K - 1} свободных весов против {N_THETA} параметров. "
                f"Для K < {self.min_periods_parametric} используйте scheme='umidas'"
            )

        if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(Z).all():
            raise ValueError(
                "MIDAS не принимает пропуски: очистите X, y, Z до вызова fit "
                "(молчаливое заполнение здесь недопустимо)"
            )

        groups = np.asarray(groups)
        tss_ref = None

        # ---------- U-MIDAS: линейная задача, оптимизация не нужна ----------
        if self.scheme == "umidas":
            D = np.hstack([X.reshape(N, V * K), Z])
            labels, codes, counts = self._group_codes(groups)
            if self.fixed_effects:
                y_d, D_d, ym, Dm = self._demean_fast(y, D, codes, counts)
            else:
                y_d, D_d = y - y.mean(), D - D.mean(axis=0)
                ym = Dm = None
            coef = self._ols(D_d, y_d)
            resid = y_d - D_d @ coef
            rss = float(resid @ resid)
            tss_ref = float(y_d @ y_d)
            self.fit_ = MidasFit(
                scheme="umidas", hf_names=hf_names, lf_names=lf_names, K=K,
                theta=None, coef=coef, intercept=float(y.mean()),
                group_effects=(
                    {str(g): float(ym[i] - Dm[i] @ coef) for i, g in enumerate(labels)}
                    if self.fixed_effects else {}
                ),
                rss=rss, r2_within=1.0 - rss / tss_ref if tss_ref > _EPS else np.nan,
                n_obs=N, n_starts_converged=1,
            )
            return self.fit_

        # ---------- Параметрические схемы: концентрированный НМНК -----------
        from scipy.optimize import least_squares

        labels, codes, counts = self._group_codes(groups)

        def residuals(theta_flat: np.ndarray) -> np.ndarray:
            theta = theta_flat.reshape(V, N_THETA)
            D = np.hstack([self._aggregate(X, theta), Z])
            if self.fixed_effects:
                y_d, D_d, _, _ = self._demean_fast(y, D, codes, counts)
            else:
                y_d, D_d = y - y.mean(), D - D.mean(axis=0)
            coef = self._ols(D_d, y_d)
            return y_d - D_d @ coef

        rng = np.random.default_rng(self.seed)
        # Первый старт — «плоские» веса, остальные случайные.
        starts = [np.zeros(V * N_THETA)]
        for _ in range(max(0, self.n_starts - 1)):
            starts.append(rng.normal(0.0, 1.0, V * N_THETA))

        best, best_rss, converged = None, np.inf, 0
        for start in starts:
            try:
                res = least_squares(residuals, start, method="lm", max_nfev=self.max_nfev)
            except Exception as exc:  # noqa: BLE001 — старт может разойтись, это нормально
                logger.debug("Старт MIDAS разошёлся: %s: %s", type(exc).__name__, exc)
                continue
            converged += 1
            rss = float(res.fun @ res.fun)
            if rss < best_rss:
                best_rss, best = rss, res.x

        if best is None:
            raise RuntimeError(
                f"MIDAS ({self.scheme}): ни один из {len(starts)} стартов не сошёлся"
            )

        theta = best.reshape(V, N_THETA)
        D = np.hstack([self._aggregate(X, theta), Z])
        if self.fixed_effects:
            y_d, D_d, ym, Dm = self._demean_fast(y, D, codes, counts)
        else:
            y_d, D_d = y - y.mean(), D - D.mean(axis=0)
            ym = Dm = None
        coef = self._ols(D_d, y_d)
        resid = y_d - D_d @ coef
        tss_ref = float(y_d @ y_d)

        self.fit_ = MidasFit(
            scheme=self.scheme, hf_names=hf_names, lf_names=lf_names, K=K,
            theta=theta, coef=coef, intercept=float(y.mean()),
            group_effects=(
                {str(g): float(ym[i] - Dm[i] @ coef) for i, g in enumerate(labels)}
                if self.fixed_effects else {}
            ),
            rss=float(resid @ resid),
            r2_within=1.0 - float(resid @ resid) / tss_ref if tss_ref > _EPS else np.nan,
            n_obs=N, n_starts_converged=converged,
        )
        return self.fit_

    def predict(
        self, X: np.ndarray, groups: np.ndarray, Z: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Прогноз. Для субъектов, отсутствовавших в обучении, берётся общая константа."""
        if self.fit_ is None:
            raise RuntimeError("Модель не обучена")
        f = self.fit_
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        Z = np.zeros((N, 0)) if Z is None else np.asarray(Z, dtype=float)

        if f.scheme == "umidas":
            D = np.hstack([X.reshape(N, X.shape[1] * X.shape[2]), Z])
        else:
            D = np.hstack([self._aggregate(X, f.theta), Z])

        base = D @ f.coef
        if not self.fixed_effects or not f.group_effects:
            return base + f.intercept
        fallback = float(np.mean(list(f.group_effects.values())))
        alpha = np.array([f.group_effects.get(str(g), fallback) for g in groups])
        return base + alpha


# ---------------------------------------------------------------------------
# Кластерный бутстрэп доверительных интервалов для весов
# ---------------------------------------------------------------------------
def bootstrap_weights(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    Z: Optional[np.ndarray],
    scheme: str = "beta",
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 42,
    n_starts: int = 3,
) -> Dict[str, np.ndarray]:
    """Доверительные интервалы весов кластерным бутстрэпом по субъектам.

    Ресемплируются целые регионы: внутрирегиональная зависимость наблюдений
    сохраняется, поэтому интервалы не занижаются.

    Returns:
        {'lower': (V, K), 'median': (V, K), 'upper': (V, K), 'n_ok': int}
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    draws: List[np.ndarray] = []

    for b in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in picked])
        # Уникальные метки, иначе один и тот же регион слипнется в один эффект
        gb = np.concatenate(
            [np.full((groups == g).sum(), f"{g}__{k}") for k, g in enumerate(picked)]
        )
        try:
            fit = PanelMIDAS(scheme=scheme, n_starts=n_starts, seed=seed + b).fit(
                X[idx], y[idx], gb, None if Z is None else Z[idx]
            )
            draws.append(np.vstack([w for w in fit.weights().values()]))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Бутстрэп-итерация %d отброшена: %s", b, exc)

    if not draws:
        raise RuntimeError("Бутстрэп не дал ни одной успешной итерации")

    stack = np.stack(draws)  # (B, V, K)
    return {
        "lower": np.quantile(stack, alpha / 2, axis=0),
        "median": np.quantile(stack, 0.5, axis=0),
        "upper": np.quantile(stack, 1 - alpha / 2, axis=0),
        "n_ok": len(draws),
    }
