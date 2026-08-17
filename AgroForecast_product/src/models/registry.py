"""Фабрика моделей и единый интерфейс обучения/прогноза.

Модели различаются отношением к пропускам и категориальным признакам,
поэтому обёртка приводит их к одному контракту:

    fit(X_train, y_train) -> None
    predict(X) -> np.ndarray

* CatBoost      — нативно работает с NaN и категориальным `region`;
* HistGradientBoosting — нативно работает с NaN, `region` кодируется порядково;
* RandomForest  — требует импутации; выполняется явно медианой ОБУЧАЮЩЕЙ
                  выборки (никакой утечки из теста).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

REGION_COLUMN = "region"


@dataclass
class ModelWrapper:
    """Единый интерфейс поверх разных реализаций."""

    name: str
    params: Dict[str, Any]
    feature_columns: List[str]
    use_region: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _medians: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _region_codes: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ утилиты
    def _prepare(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        X = df[self.feature_columns].copy()

        if self.use_region:
            if self.name == "catboost":
                X[REGION_COLUMN] = df[REGION_COLUMN].astype(str)
            else:
                if fitting:
                    self._region_codes = {
                        r: i for i, r in enumerate(sorted(df[REGION_COLUMN].unique()))
                    }
                X[REGION_COLUMN] = (
                    df[REGION_COLUMN].map(self._region_codes).astype("float64")
                )

        if self.name == "random_forest":
            numeric = X.select_dtypes(include=[np.number]).columns
            if fitting:
                self._medians = X[numeric].median()
            X[numeric] = X[numeric].fillna(self._medians)
            # Признак, полностью пустой в обучении, останется NaN — заменяем на 0
            X[numeric] = X[numeric].fillna(0.0)
        return X

    # -------------------------------------------------------------------- API
    def fit(self, train: pd.DataFrame, target: str) -> "ModelWrapper":
        X = self._prepare(train, fitting=True)
        y = train[target].to_numpy(dtype=float)

        if self.name == "catboost":
            from catboost import CatBoostRegressor, Pool

            cat_features = [REGION_COLUMN] if self.use_region else []
            self._model = CatBoostRegressor(**self.params)
            self._model.fit(Pool(X, y, cat_features=cat_features))
        elif self.name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            self._model = RandomForestRegressor(**self.params)
            self._model.fit(X, y)
        elif self.name == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            self._model = HistGradientBoostingRegressor(**self.params)
            self._model.fit(X, y)
        else:
            raise ValueError(f"Неизвестная модель: {self.name}")
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"Модель {self.name} не обучена")
        X = self._prepare(data, fitting=False)
        return np.asarray(self._model.predict(X), dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        """Важность признаков (если поддерживается реализацией)."""
        cols = list(self.feature_columns) + ([REGION_COLUMN] if self.use_region else [])
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        if hasattr(self._model, "get_feature_importance"):
            values = self._model.get_feature_importance()
        elif hasattr(self._model, "feature_importances_"):
            values = self._model.feature_importances_
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame({"feature": cols[: len(values)], "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.name == "catboost":
            self._model.save_model(str(path))
        else:
            import joblib

            joblib.dump(self._model, path)
        logger.info("Модель %s сохранена: %s", self.name, path)


def build_model(
    name: str,
    config_models: Dict[str, Dict[str, Any]],
    feature_columns: Sequence[str],
    use_region: bool = True,
) -> ModelWrapper:
    """Создаёт обёртку модели с параметрами из конфига."""
    if name not in config_models:
        raise KeyError(f"В конфиге нет параметров для модели «{name}»")
    return ModelWrapper(
        name=name,
        params=dict(config_models[name]),
        feature_columns=list(feature_columns),
        use_region=use_region,
    )
