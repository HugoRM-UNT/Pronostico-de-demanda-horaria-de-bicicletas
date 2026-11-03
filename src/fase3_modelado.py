"""Etapas 7-16: pipelines, validación temporal, ajuste y evaluación."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from fase1_xy import agregar_calendario, construir_X_y
from fase2_eda import corte_temporal

N_SPLITS = 5
RANDOM_STATE = 42
# Streamlit Cloud suele tener límites estrictos de RAM/CPU. Un solo worker
# evita que GridSearchCV cree procesos adicionales por cada sesión.
N_JOBS = 1
NUMERICAS = [
    "temp", "atemp", "humidity", "windspeed", "year", "days_since_start",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
]
CATEGORICAS = ["season", "holiday", "workingday", "weather", "hour", "month", "dayofweek"]


def preprocesador() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERICAS),
        ("categorical", categorical, CATEGORICAS),
    ])


def preprocesador_arbol() -> ColumnTransformer:
    """Preprocesador denso para HistGradientBoostingRegressor."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERICAS),
        ("categorical", categorical, CATEGORICAS),
    ])


def agregar_features_modelado(df: pd.DataFrame, inicio: pd.Timestamp) -> pd.DataFrame:
    """Añade señales temporales disponibles al momento de predecir."""
    result = agregar_calendario(df)
    horas = result["hour"].astype(float)
    dias = result["dayofweek"].astype(float)
    meses = result["month"].astype(float)
    result["days_since_start"] = (
        (result["datetime"] - inicio).dt.total_seconds() / 86400
    )
    result["hour_sin"] = np.sin(2 * np.pi * horas / 24)
    result["hour_cos"] = np.cos(2 * np.pi * horas / 24)
    result["dayofweek_sin"] = np.sin(2 * np.pi * dias / 7)
    result["dayofweek_cos"] = np.cos(2 * np.pi * dias / 7)
    result["month_sin"] = np.sin(2 * np.pi * meses / 12)
    result["month_cos"] = np.cos(2 * np.pi * meses / 12)
    return result


def crear_pipelines() -> dict[str, Pipeline]:
    modelos = {
        "LinearRegression": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR(),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", preprocesador_arbol()),
            ("model", HistGradientBoostingRegressor(
                max_iter=350,
                learning_rate=0.05,
                max_leaf_nodes=63,
                l2_regularization=2.0,
                random_state=RANDOM_STATE,
            )),
        ]),
        "DummyRegressor": DummyRegressor(strategy="mean"),
    }
    pipelines = {
        nombre: Pipeline([("preprocessor", preprocesador()), ("model", modelo)])
        for nombre, modelo in modelos.items()
        if nombre != "HistGradientBoosting"
    }
    pipelines["HistGradientBoosting"] = modelos["HistGradientBoosting"]
    return pipelines


def particionar(df: pd.DataFrame):
    train, test = corte_temporal(df)
    inicio = train["datetime"].min()
    x_train, y_train = construir_X_y(agregar_features_modelado(train, inicio))
    x_test, y_test = construir_X_y(agregar_features_modelado(test, inicio))
    return x_train, x_test, y_train, y_test, train, test


def validar_modelos(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    filas = []
    for nombre, pipeline in crear_pipelines().items():
        inicio = time.perf_counter()
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
            return_train_score=False,
        )
        fold_mae = -scores["test_mae"]
        filas.append({
            "Modelo": nombre,
            "MAE fold 1": fold_mae[0],
            "MAE fold 2": fold_mae[1],
            "MAE fold 3": fold_mae[2],
            "MAE fold 4": fold_mae[3],
            "MAE fold 5": fold_mae[4],
            "MAE medio": fold_mae.mean(),
            "MAE desviación": fold_mae.std(),
            "RMSE medio": -scores["test_rmse"].mean(),
            "R² medio": scores["test_r2"].mean(),
            "Tiempo (s)": time.perf_counter() - inicio,
        })
    return pd.DataFrame(filas).sort_values("MAE medio").reset_index(drop=True)


def ajustar_hiperparametros(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    base = crear_pipelines()
    grids = {
        "KNN": {
            "model__n_neighbors": [5, 15, 30],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "SVR": {
            "model__C": [10, 50],
            "model__epsilon": [0.1, 0.3],
            "model__kernel": ["rbf"],
        },
    }
    resultados = []
    top = {}
    buscadores = {}
    for nombre, grid in grids.items():
        inicio = time.perf_counter()
        search = GridSearchCV(
            base[nombre], grid, cv=cv, scoring="neg_mean_absolute_error",
            n_jobs=N_JOBS, pre_dispatch=N_JOBS, refit=True, return_train_score=False,
        )
        search.fit(x_train, y_train)
        buscadores[nombre] = search
        top[nombre] = (
            pd.DataFrame(search.cv_results_)
            .sort_values("rank_test_score")
            .head(5)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
            .assign(MAE=lambda d: -d["mean_test_score"])
            .drop(columns="mean_test_score")
        )
        resultados.append({
            "Modelo": nombre,
            "MAE CV óptimo": -search.best_score_,
            "Mejores parámetros": str(search.best_params_),
            "Tiempo (s)": time.perf_counter() - inicio,
        })
    return {"tabla": pd.DataFrame(resultados), "searches": buscadores, "top": top}


def entrenar_experimento(df: pd.DataFrame) -> dict[str, Any]:
    """Ejecuta CV y grids usando exclusivamente el 80 % inicial."""
    x_train, x_test, y_train, y_test, train, test = particionar(df)
    cv = validar_modelos(x_train, y_train)
    grids = ajustar_hiperparametros(x_train, y_train)
    combinada = cv[["Modelo", "MAE medio", "MAE desviación"]].merge(
        grids["tabla"][["Modelo", "MAE CV óptimo", "Mejores parámetros"]],
        on="Modelo",
        how="outer",
    ).sort_values("MAE medio", na_position="last")
    comparacion = cv[["Modelo", "MAE medio"]].copy()
    for nombre in ("KNN", "SVR"):
        comparacion.loc[comparacion["Modelo"] == nombre, "MAE medio"] = (
            grids["tabla"].loc[grids["tabla"]["Modelo"] == nombre, "MAE CV óptimo"].iloc[0]
        )
    ganador = comparacion.sort_values("MAE medio").iloc[0]["Modelo"]
    modelo = (
        grids["searches"][ganador].best_estimator_
        if ganador in grids["searches"]
        else crear_pipelines()[ganador]
    )
    return {
        "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test,
        "train": train, "test": test, "cv": cv, "grids": grids["tabla"],
        "grid_top": grids["top"],
        "comparison": combinada, "selection": comparacion.sort_values("MAE medio"),
        "winner": ganador, "model": modelo,
    }


def evaluar(experimento: dict[str, Any]) -> dict[str, Any]:
    """Ajusta el ganador una vez al train y abre el test solo para esta evaluación."""
    model = experimento["model"]
    model.fit(experimento["x_train"], experimento["y_train"])
    pred = np.clip(model.predict(experimento["x_test"]), 0, None)
    y = experimento["y_test"]
    metrics = {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R²": r2_score(y, pred),
    }
    detail = agregar_calendario(experimento["test"]).copy()
    detail["prediccion"] = pred
    detail["residual"] = detail["count"] - detail["prediccion"]
    detail["franja"] = pd.cut(
        detail["hour"], [-1, 5, 10, 15, 19, 23],
        labels=["Madrugada (0-5)", "Mañana (6-10)", "Mediodía (11-15)",
                "Pico tarde (16-19)", "Noche (20-23)"],
    )
    by_hour = detail.groupby("hour", observed=False)[["count", "prediccion"]].apply(
        lambda g: pd.Series({"n": len(g), "MAE": mean_absolute_error(g["count"], g["prediccion"])})
    ).reset_index()
    by_weather = detail.groupby("weather", observed=False)[["count", "prediccion"]].apply(
        lambda g: pd.Series({"n": len(g), "MAE": mean_absolute_error(g["count"], g["prediccion"])})
    ).reset_index()
    return {"model": model, "pred": pred, "metrics": metrics, "detail": detail,
            "by_hour": by_hour, "by_weather": by_weather}


def cinco_predicciones(experimento: dict[str, Any], evaluacion: dict[str, Any]) -> pd.DataFrame:
    """Genera cinco escenarios futuros; no se usan para medir exactitud."""
    ultimo = experimento["train"].iloc[-1].copy()
    fechas = pd.date_range(ultimo["datetime"] + pd.Timedelta(hours=1), periods=5, freq="h")
    nuevos = pd.DataFrame([ultimo.to_dict()] * 5)
    nuevos["datetime"] = fechas
    nuevos["count"] = np.nan
    nuevos = agregar_calendario(nuevos)
    x_nuevos, _ = construir_X_y(
        agregar_features_modelado(nuevos.assign(count=0), experimento["train"]["datetime"].min())
    )
    return nuevos[["datetime", "weather", "temp", "humidity"]].assign(
        prediccion=np.clip(evaluacion["model"].predict(x_nuevos), 0, None)
    )
