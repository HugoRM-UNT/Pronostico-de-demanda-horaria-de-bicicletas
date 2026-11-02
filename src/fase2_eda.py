"""Fase 2 (etapas 5-6): calidad de datos, EDA y partición temporal.

Reglas de este módulo:
  * La partición es CRONOLÓGICA (últimas filas = test). No hay aleatoriedad.
  * Las tablas de calidad (T2) son descriptivas y se calculan sobre el archivo completo.
  * Todo lo que relaciona predictores con `count` (gráficos, correlaciones, hallazgos)
    se calcula SOLO con el tramo de entrenamiento, para no "mirar" el test.

Uso:  python src/fase2_eda.py
"""
from __future__ import annotations

import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from fase1_xy import TARGET, cargar_datos, agregar_calendario, construir_X_y

RANDOM_STATE = 42          # se declara; el corte temporal es determinista y no lo usa
TEST_SIZE = 0.20           # 80/20 orientativo (guía, etapa 6)
N_SPLITS = 5               # TimeSeriesSplit(n_splits=5), obligatorio para este grupo
HORAS_PICO = [7, 8, 17, 18, 19]   # resultado de horas_pico(train) con el train.csv real (ver función)

COLUMNAS_ESPERADAS = ["datetime", "season", "holiday", "workingday", "weather", "temp",
                      "atemp", "humidity", "windspeed", "casual", "registered", "count"]
NUMERICAS = ["temp", "atemp", "humidity", "windspeed"]
NOMBRES_CLIMA = {1: "1 Despejado", 2: "2 Nublado/neblina", 3: "3 Lluvia/nieve ligera",
                 4: "4 Lluvia/nieve fuerte"}


# --------------------------------------------------------------------------- #
# Validación del CSV (mensaje comprensible ante archivo incorrecto: Anexo H.2)
# --------------------------------------------------------------------------- #
def validar_csv(df: pd.DataFrame) -> None:
    """Lanza ValueError con un mensaje claro si el CSV no es el train.csv esperado."""
    faltan = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltan:
        raise ValueError(
            f"El CSV no tiene las columnas esperadas: faltan {faltan}. "
            "Usa el archivo train.csv de la competencia Bike Sharing Demand "
            "(test.csv no trae count)."
        )


# --------------------------------------------------------------------------- #
# Etapa 5 · Calidad de datos (T2)
# --------------------------------------------------------------------------- #
def resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, nulos, % de nulos y valores únicos por columna."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "no_nulos": df.notna().sum(),
        "nulos": df.isna().sum(),
        "%_nulos": (df.isna().mean() * 100).round(2),
        "únicos": df.nunique(),
    })


def huecos_serie(df: pd.DataFrame) -> dict:
    """Huecos de la serie horaria.
    * horas_faltantes_en_dias_presentes: horas que faltan dentro de días que sí tienen registros.
    * dias_ausentes: días calendario sin ningún registro entre el primer y el último dato
      (en el train.csv de Kaggle se espera que falten los días finales de cada mes; verificarlo)."""
    por_dia = df.groupby(df["datetime"].dt.normalize())["datetime"].count()
    dias_rango = pd.date_range(por_dia.index.min(), por_dia.index.max(), freq="D")
    dias_mes = df["datetime"].dt.day
    return {"horas_faltantes_en_dias_presentes": int((24 - por_dia).sum()),
            "dias_presentes": int(len(por_dia)),
            "dias_ausentes": int(len(dias_rango) - len(por_dia)),
            "dia_del_mes_min": int(dias_mes.min()), "dia_del_mes_max": int(dias_mes.max())}


def tabla_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """T2 · Diagnóstico de calidad. Los números salen del CSV; la 'medida adoptada'
    es una PROPUESTA que el grupo debe revisar."""
    n = len(df)
    nulos = int(df.isna().sum().sum())
    dup_filas = int(df.duplicated().sum())
    dup_fecha = int(df["datetime"].duplicated().sum())
    hum0 = int((df["humidity"] == 0).sum())
    vien0 = int((df["windspeed"] == 0).sum())
    cnt0 = int((df[TARGET] == 0).sum())
    hz = huecos_serie(df)
    suma_ok = bool((df["casual"] + df["registered"] == df[TARGET]).all())
    neg = int((df[["humidity", "windspeed", TARGET]] < 0).sum().sum())
    cat_raras = int((~df["weather"].isin([1, 2, 3, 4])).sum() + (~df["season"].isin([1, 2, 3, 4])).sum())

    def pct(k):
        return f"{100 * k / n:.2f} %"

    filas = [
        ("Valores nulos (todas las columnas)", nulos, pct(nulos),
         "Sin nulos que tratar. Se mantiene SimpleImputer dentro del Pipeline como salvaguarda."
         if nulos == 0 else "Imputar dentro del Pipeline (mediana en numéricas, moda en categóricas)."),
        ("Filas duplicadas", dup_filas, pct(dup_filas), "Eliminar si > 0." if dup_filas else "Ninguna acción."),
        ("datetime repetido", dup_fecha, pct(dup_fecha),
         "Revisar antes de cortar; el corte cronológico asume una fila por hora." if dup_fecha
         else "Ninguna acción: una fila por hora."),
        ("Horas ausentes dentro de días presentes", hz["horas_faltantes_en_dias_presentes"],
         pct(hz["horas_faltantes_en_dias_presentes"]),
         "Se documenta. No se rellenan: los modelos no usan rezagos de count."),
        ("Días calendario sin registros (entre primer y último dato)", hz["dias_ausentes"],
         f"{hz['dias_ausentes']} de {hz['dias_ausentes'] + hz['dias_presentes']} días",
         f"Solo existen los días {hz['dia_del_mes_min']}–{hz['dia_del_mes_max']} de cada mes (los demás días "
         "los reserva Kaggle para test.csv, que no trae count). No se rellenan; el corte y los folds se "
         "hacen por orden cronológico."),
        ("humidity = 0", hum0, pct(hum0), "Valor físicamente imposible, concentrado en pocos días: se conserva y se declara como limitación (efecto mínimo)."),
        ("windspeed = 0", vien0, pct(vien0),
         "Posible dato no medido (no hay valores entre 0 y el mínimo positivo): se conserva sin imputar y se declara como limitación."),
        ("count = 0", cnt0, pct(cnt0), "Plausible (madrugada): se conserva."),
        ("Valores negativos (humidity, windspeed, count)", neg, pct(neg), "Ninguna acción." if neg == 0 else "Revisar."),
        ("Categorías fuera de rango (season, weather)", cat_raras, pct(cat_raras),
         "Ninguna acción." if cat_raras == 0 else "Revisar."),
        ("casual + registered ≠ count (filas)", int((df["casual"] + df["registered"] != df[TARGET]).sum()),
         pct(int((df["casual"] + df["registered"] != df[TARGET]).sum())),
         "Confirma que casual y registered son componentes de count: EXCLUIDAS por fuga."
         if suma_ok else "Revisar la relación entre casual, registered y count."),
    ]
    return pd.DataFrame(filas, columns=["Chequeo", "Cantidad", "Porcentaje", "Medida adoptada (propuesta)"])


def outliers_objetivo(y: pd.Series) -> dict:
    """Valores altos de count según la regla 1.5·IQR (descriptivo, no se eliminan)."""
    q1, q3 = y.quantile([0.25, 0.75])
    lim = q3 + 1.5 * (q3 - q1)
    return {"limite_superior_IQR": float(lim), "n_sobre_limite": int((y > lim).sum()),
            "pct_sobre_limite": float(100 * (y > lim).mean())}


def horas_pico(train: pd.DataFrame, factor: float = 1.5) -> list[int]:
    """Horas pico = horas de días laborables cuya demanda media supera `factor` × la media global
    del entrenamiento. Regla definida con el train (nunca con el test)."""
    d = agregar_calendario(train)
    m = d[d["workingday"] == 1].groupby("hour")[TARGET].mean()
    return sorted(int(h) for h in m[m > factor * train[TARGET].mean()].index)


# --------------------------------------------------------------------------- #
# Etapa 6 · Partición temporal (T3)
# --------------------------------------------------------------------------- #
def corte_temporal(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Holdout cronológico: el tramo final (test_size) queda reservado como test.
    No baraja nada: ordena por datetime y corta por posición."""
    df = df.sort_values("datetime").reset_index(drop=True)
    corte = int(len(df) * (1 - test_size))
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


def tabla_particion(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """T3 · Partición de datos."""
    total = len(train) + len(test)
    filas = []
    for nombre, d in (("Entrenamiento", train), ("Prueba (test)", test)):
        filas.append((nombre, len(d), f"{100 * len(d) / total:.1f} %",
                      f"{d['datetime'].min():%Y-%m-%d %H:%M}", f"{d['datetime'].max():%Y-%m-%d %H:%M}"))
    filas.append(("Total", total, "100 %", f"{train['datetime'].min():%Y-%m-%d %H:%M}",
                  f"{test['datetime'].max():%Y-%m-%d %H:%M}"))
    t = pd.DataFrame(filas, columns=["Conjunto", "Registros", "Proporción", "Desde", "Hasta"])
    t["Método"] = ["Primeras filas en orden cronológico", "Últimas filas en orden cronológico", "—"]
    t["Semilla"] = "No aplica (corte determinista); RANDOM_STATE=42 declarado"
    return t


def verificar_particion(train: pd.DataFrame, test: pd.DataFrame, total: int) -> dict:
    """Comprobaciones exigidas en la etapa 6 (ausencia de filas compartidas, orden temporal)."""
    res = {
        "sin_indices_compartidos": not (set(train.index) & set(test.index)),
        "sin_fechas_compartidas": not (set(train["datetime"]) & set(test["datetime"])),
        "train_anterior_a_test": bool(train["datetime"].max() < test["datetime"].min()),
        "suma_registros_igual_total": len(train) + len(test) == total,
        "train_ordenado": bool(train["datetime"].is_monotonic_increasing),
        "test_ordenado": bool(test["datetime"].is_monotonic_increasing),
    }
    assert all(res.values()), f"Falló alguna comprobación de partición: {res}"
    return res


def dividir_X_y(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Atajo para las fases siguientes: (X_train, X_test, y_train, y_test), cronológico.
    Primero se arma X/y (etapa 4) y luego se corta; sin transformar nada todavía."""
    df = df.sort_values("datetime").reset_index(drop=True)
    X, y = construir_X_y(df)
    corte = int(len(df) * (1 - test_size))
    return X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]


def tabla_folds(train: pd.DataFrame, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Muestra los 5 folds de TimeSeriesSplit sobre el entrenamiento (los mismos de la etapa 10)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    filas = []
    for i, (a, b) in enumerate(tscv.split(train), 1):
        assert a.max() < b.min(), "Un fold usa datos futuros en su entrenamiento"
        filas.append((f"Fold {i}", len(a), len(b), round(float(train[TARGET].iloc[b].mean()), 1),
                      f"{train['datetime'].iloc[a[0]]:%Y-%m-%d}", f"{train['datetime'].iloc[a[-1]]:%Y-%m-%d}",
                      f"{train['datetime'].iloc[b[0]]:%Y-%m-%d}", f"{train['datetime'].iloc[b[-1]]:%Y-%m-%d}"))
    return pd.DataFrame(filas, columns=["Fold", "n entrenamiento", "n validación", "count medio validación", "Train desde",
                                         "Train hasta", "Validación desde", "Validación hasta"])


def cobertura_estacional(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Qué meses/años cubre el test y si el entrenamiento ya vio esos meses (riesgo de extrapolación)."""
    meses_test = sorted(test["datetime"].dt.month.unique().tolist())
    vistos_en_train = sorted(set(train["datetime"].dt.month.unique()) & set(meses_test))
    anios_test = sorted(test["datetime"].dt.year.unique().tolist())
    anios_train = sorted(train["datetime"].dt.year.unique().tolist())
    meses_solo_otro_anio = [m for m in meses_test
                            if not ((train["datetime"].dt.month == m) & (train["datetime"].dt.year.isin(anios_test))).any()]
    return {"meses_test": meses_test, "anios_test": anios_test, "anios_train": anios_train,
            "meses_test_vistos_en_train": vistos_en_train,
            "meses_test_sin_datos_del_mismo_anio_en_train": meses_solo_otro_anio}


# --------------------------------------------------------------------------- #
# Etapa 5 · Gráficos (cada función devuelve una Figure; se calculan con TRAIN)
# --------------------------------------------------------------------------- #
def _fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def fig_hist_objetivo(train: pd.DataFrame):
    fig, ax = _fig(8, 4)
    ax.hist(train[TARGET], bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(train[TARGET].mean(), color="crimson", ls="--", label=f"media = {train[TARGET].mean():.0f}")
    ax.axvline(train[TARGET].median(), color="orange", ls="--", label=f"mediana = {train[TARGET].median():.0f}")
    ax.set(xlabel="count (alquileres por hora)", ylabel="Frecuencia", title="Distribución del objetivo (train)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_serie_temporal(train: pd.DataFrame):
    d = train.set_index("datetime")[TARGET].resample("D").mean()
    fig, ax = _fig(10, 4)
    ax.plot(d.index, d.values, color="#9ecae1", lw=0.8, label="media diaria")
    ax.plot(d.index, d.rolling(7, min_periods=1).mean(), color="#08519c", lw=1.8, label="media móvil 7 días")
    ax.set(xlabel="Fecha", ylabel="count medio por hora", title="Serie temporal de la demanda (train)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_perfil_horario(train: pd.DataFrame):
    d = agregar_calendario(train)
    p = d.groupby(["workingday", "hour"])[TARGET].mean().unstack(0)
    fig, ax = _fig(9, 4)
    for col, nom, c in ((1, "Día laborable", "#c44e52"), (0, "No laborable", "#4C72B0")):
        if col in p.columns:
            ax.plot(p.index, p[col], marker="o", label=nom, color=c)
    ax.set(xlabel="Hora del día", ylabel="count medio", title="Perfil horario de la demanda (train)",
           xticks=range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_demanda_mes(train: pd.DataFrame):
    d = agregar_calendario(train)
    p = d.groupby(["month", "year"])[TARGET].mean().unstack()
    fig, ax = _fig(9, 4)
    p.plot(kind="bar", ax=ax, width=0.8)
    ax.set(xlabel="Mes", ylabel="count medio", title="Demanda media por mes y año (train)")
    ax.legend(title="Año")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def fig_temp_vs_count(train: pd.DataFrame):
    fig, ax = _fig(8, 4.5)
    ax.scatter(train["temp"], train[TARGET], s=6, alpha=0.25, color="#4C72B0")
    b = pd.cut(train["temp"], bins=12)
    m = train.groupby(b, observed=True).agg(t=("temp", "mean"), c=(TARGET, "mean"))
    ax.plot(m["t"], m["c"], color="crimson", marker="o", label="media por tramo de temp")
    ax.set(xlabel="temp (°C)", ylabel="count", title="Temperatura vs. demanda (train)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_box_clima(train: pd.DataFrame):
    grupos = [(NOMBRES_CLIMA.get(w, str(w)), g[TARGET].values) for w, g in train.groupby("weather")]
    fig, ax = _fig(9, 4)
    ax.boxplot([g for _, g in grupos])
    ax.set_xticklabels([f"{n}\n(n={len(g)})" for n, g in grupos])
    ax.set(ylabel="count", title="Demanda según condición meteorológica (train)")
    fig.tight_layout()
    return fig


def fig_correlacion(train: pd.DataFrame):
    X, y = construir_X_y(train)
    c = pd.concat([X, y], axis=1).corr(numeric_only=True)
    fig, ax = _fig(7, 6)
    im = ax.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(c)), c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlación de Pearson (train; sin casual/registered)")
    fig.tight_layout()
    return fig


def fig_particion(train: pd.DataFrame, test: pd.DataFrame):
    """Línea de tiempo del corte. Solo muestra fechas y tamaños, nunca valores de count del test."""
    fig, ax = _fig(10, 1.8)
    t0, t1, t2 = train["datetime"].min(), train["datetime"].max(), test["datetime"].max()
    ax.barh([0], [(t1 - t0).days], left=[t0], color="#4C72B0", label=f"Entrenamiento ({len(train):,})")
    ax.barh([0], [(t2 - test["datetime"].min()).days], left=[test["datetime"].min()], color="#dd8452",
            label=f"Prueba ({len(test):,})")
    ax.set_yticks([])
    ax.set_title("Corte cronológico 80/20")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Hallazgos calculados con los datos (borrador de interpretación)
# --------------------------------------------------------------------------- #
def _eta2(d: pd.DataFrame, col: str) -> float:
    """Proporción de la varianza de count explicada por las categorías de `col` (eta²)."""
    g = d.groupby(col)[TARGET]
    entre = (g.count() * (g.mean() - d[TARGET].mean()) ** 2).sum()
    return float(entre / ((d[TARGET] - d[TARGET].mean()) ** 2).sum())


def hallazgos(train: pd.DataFrame) -> list[dict]:
    """Cinco hallazgos con cifras calculadas sobre TRAIN. Texto = borrador para revisar:
    las frases cambian según lo que muestren los datos."""
    d = agregar_calendario(train)
    y = d[TARGET]
    out = []

    out.append({
        "titulo": "Forma de la distribución del objetivo",
        "evidencia": f"media = {y.mean():.1f}, mediana = {y.median():.1f}, máximo = {y.max():.0f}, "
                     f"asimetría = {y.skew():.2f}; {100 * (y <= 10).mean():.1f} % de las horas tiene count ≤ 10.",
        "implicacion": ("La cola derecha eleva la media sobre la mediana: MAE es más robusto que RMSE como "
                        "métrica primaria y conviene analizar el error por franja horaria, no solo el global."
                        if y.skew() > 0.5 else
                        "La distribución es poco asimétrica; igualmente se reporta MAE por franja horaria."),
    })

    pr = d.groupby(["workingday", "hour"])[TARGET].mean()
    top_lab = sorted(pr.loc[1].nlargest(3).index.tolist()) if 1 in pr.index.get_level_values(0) else []
    top_no = sorted(pr.loc[0].nlargest(3).index.tolist()) if 0 in pr.index.get_level_values(0) else []
    e_h, e_m, e_w = _eta2(d, "hour"), _eta2(d, "month"), _eta2(d, "weather")
    difieren = top_lab != top_no
    out.append({
        "titulo": "Patrón horario",
        "evidencia": f"La hora explica {100 * e_h:.1f} % de la varianza de count (mes: {100 * e_m:.1f} %, "
                     f"clima: {100 * e_w:.1f} %). Horas de mayor demanda media: laborables {top_lab}; "
                     f"no laborables {top_no}.",
        "implicacion": ("Las horas pico difieren entre días laborables y no laborables: hay interacción "
                        "hora × workingday que un modelo lineal sin términos de interacción no captura. "
                        if difieren else "Las horas pico coinciden entre tipos de día. ")
                       + f"Horas pico (laborables, demanda media > 1.5 × la global): {horas_pico(train)}.",
    })

    mes_anio = d.groupby(["year", "month"])[TARGET].mean().unstack(0)
    comunes = mes_anio.dropna()                      # meses presentes en todos los años del train
    anios_completos = [a_ for a_ in mes_anio.columns if mes_anio[a_].notna().all()]
    if len(mes_anio.columns) >= 2 and len(comunes) > 0:
        a_, b_ = mes_anio.columns.min(), mes_anio.columns.max()
        mc = d[d["month"].isin(comunes.index)].groupby("year")[TARGET].mean()
        crec = 100 * (mc[b_] / mc[a_] - 1)
        ev = (f"En los meses comunes ({int(comunes.index.min())}–{int(comunes.index.max())}) el count medio "
              f"pasa de {mc[a_]:.1f} en {a_} a {mc[b_]:.1f} en {b_} ({crec:+.1f} %).")
        imp = ("El nivel de demanda cambia entre años y el test es posterior al train: el modelo debe "
               "extrapolar. `year` captura ese nivel pero no sirve para años nuevos "
               "(limitación para el análisis crítico).")
    else:
        ev, imp = "El entrenamiento solo contiene un año.", "No se puede evaluar la tendencia anual."
    ref = mes_anio[anios_completos[0]] if anios_completos else d.groupby("month")[TARGET].mean()
    ref_txt = f"en {anios_completos[0]} (año completo)" if anios_completos else "(promedio de años)"
    out.append({
        "titulo": "Estacionalidad y nivel por año",
        "evidencia": ev + f" {ref_txt.capitalize()}: mes de mayor demanda media = {int(ref.idxmax())} "
                          f"({ref.max():.0f}); de menor = {int(ref.idxmin())} ({ref.min():.0f}).",
        "implicacion": imp,
    })

    ct, cw, ch = (float(d["temp"].corr(y)), float(d["windspeed"].corr(y)), float(d["humidity"].corr(y)))
    cta = float(d["temp"].corr(d["atemp"]))
    out.append({
        "titulo": "Clima continuo y redundancia temp–atemp",
        "evidencia": f"corr(temp, count) = {ct:.2f}; corr(humidity, count) = {ch:.2f}; "
                     f"corr(windspeed, count) = {cw:.2f}; corr(temp, atemp) = {cta:.2f}.",
        "implicacion": ("temp y atemp son casi la misma señal (multicolinealidad): puede inestabilizar "
                        "LinearRegression; decidir si se conserva atemp y justificarlo en T1. "
                        if abs(cta) > 0.9 else "")
                       + "Las correlaciones lineales con count son modestas; la relación real depende de la hora.",
    })

    mw = d.groupby("weather")[TARGET].agg(["mean", "count"])
    txt = "; ".join(f"clima {int(k)}: media {v['mean']:.1f} (n={int(v['count'])})" for k, v in mw.iterrows())
    pocos = [int(k) for k, v in mw.iterrows() if v["count"] < 0.02 * len(d)]
    ok = mw[mw["count"] >= 0.02 * len(d)]            # solo categorías con muestra suficiente
    ref_c, peor = int(ok.index.min()), int(ok.index.max())
    baja = bool(ok.loc[peor, "mean"] < ok.loc[ref_c, "mean"])
    out.append({
        "titulo": "Efecto de la condición meteorológica y tamaño de cada categoría",
        "evidencia": txt + (f". Categorías con < 2 % de los datos: {pocos}." if pocos else "."),
        "implicacion": (f"Entre las categorías con muestra suficiente, la demanda media del clima {peor} es "
                        f"{'menor' if baja else 'mayor'} que la del clima {ref_c}. ")
                       + (f"La(s) categoría(s) {pocos} tiene(n) tan pocos casos que su media no es interpretable "
                          "y el OneHotEncoder debe usar handle_unknown='ignore'; en el análisis de MAE por clima "
                          "hay que indicar n por categoría."
                          if pocos else "Indicar n por categoría en el análisis de MAE por clima."),
    })
    return out


def lecturas_graficos(train: pd.DataFrame) -> dict:
    """Frases cortas con cifras reales para poner debajo de cada gráfico."""
    d = agregar_calendario(train)
    y = d[TARGET]
    mh = d.groupby("hour")[TARGET].mean()
    mm = d.groupby("month")[TARGET].mean()
    bt = d.groupby(pd.cut(d["temp"], 4), observed=True)[TARGET].mean()
    mw = d.groupby("weather")[TARGET].mean()
    return {
        "hist": f"Mediana {y.median():.0f} y media {y.mean():.1f}: la cola derecha eleva la media "
                f"(asimetría {y.skew():.2f}). Máximo {y.max():.0f}.",
        "serie": f"La demanda media diaria va de {d.set_index('datetime')[TARGET].resample('D').mean().min():.0f} "
                 f"a {d.set_index('datetime')[TARGET].resample('D').mean().max():.0f} alquileres/hora "
                 "(media diaria); compara el nivel de 2011 con el de 2012 y busca el patrón estacional.",
        "perfil": f"Hora de mayor demanda media: {int(mh.idxmax())}:00 ({mh.max():.0f}); "
                  f"de menor: {int(mh.idxmin())}:00 ({mh.min():.0f}).",
        "mes": f"Promedio de ambos años: mes de mayor demanda = {int(mm.idxmax())} ({mm.max():.0f}); menor = "
               f"{int(mm.idxmin())} ({mm.min():.0f}). Ojo: los meses que solo tienen un año (en train, los "
               "posteriores a agosto) no son comparables entre años.",
        "temp": f"count medio por cuartos de temp: " + ", ".join(f"{v:.0f}" for v in bt.values) + " (de frío a calor).",
        "clima": "Media de count por clima (n): " + ", ".join(
            f"{int(k)} → {v:.0f} (n={int((d['weather'] == k).sum())})" for k, v in mw.items()) + ".",
        "corr": f"corr(temp, atemp) = {d['temp'].corr(d['atemp']):.2f}; corr(temp, count) = "
                f"{d['temp'].corr(y):.2f}.",
    }


if __name__ == "__main__":
    matplotlib.use("Agg")
    df = cargar_datos()
    validar_csv(df)
    print("=== T2 · Calidad ===")
    print(resumen_columnas(df).to_string())
    print(tabla_calidad(df).to_string(index=False))
    print("Outliers del objetivo:", outliers_objetivo(df[TARGET]))
    train, test = corte_temporal(df)
    print("\n=== T3 · Partición ===")
    print(tabla_particion(train, test).to_string(index=False))
    print("Comprobaciones:", verificar_particion(train, test, len(df)))
    print("Cobertura estacional:", cobertura_estacional(train, test))
    print("\n=== Folds de TimeSeriesSplit sobre train ===")
    print(tabla_folds(train).to_string(index=False))
    print("\n=== Hallazgos (train) ===")
    for h in hallazgos(train):
        print("-", h["titulo"], "|", h["evidencia"])
