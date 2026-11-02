"""Fase 1 (etapas 3-4): carga del dataset y construcción de X e y sin fuga.

Uso:  python src/fase1_xy.py
"""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATA_PATH = RAIZ / "data" / "train.csv"
FALLBACK_DATA_PATH = RAIZ / "train.csv"
TARGET = "count"
COLS_FUGA = ["casual", "registered"]          # suman count -> prohibidas
COLS_CALENDARIO = ["hour", "month", "dayofweek", "year"]


def cargar_datos(path=None) -> pd.DataFrame:
    """Lee train.csv y lo ordena cronológicamente (requisito de validación temporal)."""
    if path is None:
        path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descarga train.csv de Kaggle (ver data/LEEME_DATOS.md)."
        )
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError(
            f"El archivo encontrado en {path} no es el train.csv de Bike Sharing Demand: "
            "falta la columna 'datetime'. Descarga el train.csv correcto de Kaggle y "
            "colócalo en data/train.csv."
        )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def ruta_datos() -> Path:
    """Devuelve la ubicación documentada o alternativa del CSV."""
    return DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH


def agregar_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Descompone datetime en variables de calendario (no consulta valores futuros)."""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["year"] = df["datetime"].dt.year
    return df


def construir_X_y(df: pd.DataFrame):
    """y = count ; X = predictores sin fuga (sin count, casual, registered, datetime)."""
    df = agregar_calendario(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET, *COLS_FUGA, "datetime"])
    return X, y


def verificar(X: pd.DataFrame, y: pd.Series) -> dict:
    """Comprobaciones exigidas en la etapa 4."""
    resultados = {
        "indices_coinciden": X.index.equals(y.index),
        "sin_fuga": not ({TARGET, *COLS_FUGA} & set(X.columns)),
        "sin_datetime_crudo": "datetime" not in X.columns,
        "y_sin_nulos": bool(y.notna().all()),
    }
    assert all(resultados.values()), f"Falló alguna comprobación: {resultados}"
    return resultados


def tabla_exclusiones() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("count", "Es el objetivo (y)"),
            ("casual", "Fuga: casual + registered = count"),
            ("registered", "Fuga: casual + registered = count"),
            ("datetime", "Se reemplaza por hour, month, dayofweek, year"),
        ],
        columns=["Columna excluida de X", "Razón"],
    )


if __name__ == "__main__":
    df = cargar_datos()
    print(f"Filas x columnas : {df.shape}")
    print(f"Rango temporal   : {df['datetime'].min()}  ->  {df['datetime'].max()}")
    print(f"Columnas         : {list(df.columns)}\n")
    X, y = construir_X_y(df)
    print(f"X: {X.shape}   y: {y.shape}")
    print(f"Columnas de X    : {list(X.columns)}")
    print("Comprobaciones   :", verificar(X, y))
    print("\n", tabla_exclusiones().to_string(index=False))
