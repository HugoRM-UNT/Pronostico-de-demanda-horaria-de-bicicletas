"""App Streamlit · Grupo 04 · Pronóstico de demanda horaria de bicicletas."""
import inspect
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fase1_xy import (TARGET, COLS_FUGA, cargar_datos, construir_X_y,
                      ruta_datos, tabla_exclusiones, verificar)
import fase2_eda as eda
import fase3_modelado as modelado

st.set_page_config(page_title="Grupo 04 · Demanda de bicicletas", layout="wide")

DICCIONARIO = pd.DataFrame(
    [
        ("datetime", "Fecha y hora", "datetime", "hora", "Sí", "Se descompone en hour/month/dayofweek/year"),
        ("season", "Estación (1 a 4)", "categórica", "–", "Sí", "Incluir"),
        ("holiday", "Día feriado", "binaria", "0/1", "Sí", "Incluir"),
        ("workingday", "Día laborable", "binaria", "0/1", "Sí", "Incluir"),
        ("weather", "Clima (1 despejado … 4 severo)", "categórica", "–", "Solo como pronóstico", "Incluir"),
        ("temp", "Temperatura", "numérica", "°C", "Solo como pronóstico", "Incluir"),
        ("atemp", "Sensación térmica", "numérica", "°C", "Solo como pronóstico", "Incluir (revisar redundancia con temp)"),
        ("humidity", "Humedad relativa", "numérica", "%", "Solo como pronóstico", "Incluir"),
        ("windspeed", "Velocidad del viento", "numérica", "–", "Solo como pronóstico", "Incluir"),
        ("casual", "Alquileres de usuarios casuales", "numérica", "bicis", "No", "EXCLUIR (fuga)"),
        ("registered", "Alquileres de usuarios registrados", "numérica", "bicis", "No", "EXCLUIR (fuga)"),
        ("count", "Total de alquileres", "numérica", "bicis", "–", "OBJETIVO (y)"),
    ],
    columns=["Variable", "Definición", "Tipo", "Unidad", "Disponible antes", "Decisión"],
)

ETAPAS = [
    "1. Planteamiento del problema",
    "2. Objetivos y criterio de éxito",
    "3. Dataset y diccionario",
    "4. Separación de X e y",
    "5. Análisis exploratorio (EDA)",
    "6. Partición temporal train/test",
    "7-9. Pipelines y modelos",
    "10-12. Validación y selección",
    "13-14. Evaluación y análisis crítico",
    "15-16. Predicciones y conclusiones",
]

st.sidebar.title("Grupo 04")
st.sidebar.caption("Sistemas Inteligentes y Machine Learning")
etapa = st.sidebar.radio("Etapa", ETAPAS)
st.sidebar.info("La selección usa MAE en validación temporal. El test solo se abre en la etapa 13.")


@st.cache_data(show_spinner=False)
def _cargar(archivo):
    if archivo is not None:
        df = pd.read_csv(archivo, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df = cargar_datos()
    eda.validar_csv(df)
    return df


def obtener_df():
    archivo = st.sidebar.file_uploader("Cargar train.csv (opcional)", type="csv")
    if archivo is None and not ruta_datos().exists():
        return None
    try:
        return _cargar(archivo)
    except Exception as exc:  # CSV incorrecto: mensaje comprensible (Anexo H.2)
        st.error(f"No se pudo usar el CSV: {exc}")
        return None


@st.cache_data(show_spinner="Entrenando CV temporal y GridSearch...")
def ejecutar_modelado(df):
    return modelado.entrenar_experimento(df)


@st.cache_data(show_spinner="Evaluando una vez sobre el test reservado...")
def ejecutar_evaluacion(df):
    experimento = modelado.entrenar_experimento(df)
    return modelado.evaluar(experimento)


def mostrar_grafico(fig, lectura, funcion):
    """Gráfico + lectura con cifras reales + código ejecutable (Anexo E)."""
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(f"**Lectura de los datos:** {lectura}")
    with st.expander("Ver código del gráfico"):
        st.code(inspect.getsource(funcion), language="python")


st.title("Pronóstico de demanda horaria de bicicletas")

if etapa == ETAPAS[0]:
    st.header("1. Planteamiento del problema")
    st.write("Los sistemas de bicicletas compartidas deben tener bicicletas disponibles en cada "
             "estación y hora. Si hay pocas se pierden viajes; si hay demasiadas se desperdicia "
             "logística. Se usa el histórico horario de Capital Bikeshare (Washington D.C.).")
    st.subheader("Pregunta predictiva")
    st.info("¿Cuántas bicicletas se alquilarán (`count`) en una hora futura, dados el momento del "
            "día, el calendario y las condiciones meteorológicas esperadas?")
    c1, c2 = st.columns(2)
    c1.markdown("**Quién:** equipo de operaciones del sistema.\n\n"
                "**Cuándo:** el día anterior o unas horas antes del turno.\n\n"
                "**Decisión:** cuántas bicicletas redistribuir y cuándo hacer mantenimiento.")
    c2.markdown("**Enfoque:** aprendizaje supervisado, regresión.\n\n"
                "**Entradas:** `hour`, `month`, `dayofweek`, `year`, `season`, `holiday`, "
                "`workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed`.\n\n"
                "**Salida:** `count` (entero ≥ 0).")

elif etapa == ETAPAS[1]:
    st.header("2. Objetivos y criterio de éxito")
    st.markdown("**Objetivo general.** Desarrollar y evaluar un sistema predictivo de la demanda "
                "horaria de bicicletas con LinearRegression, KNeighborsRegressor y SVR, usando "
                "validación temporal sin fuga de información y un test final reservado.")
    st.markdown("**Objetivos específicos**\n"
                "1. *Análisis de datos:* identificar predictores disponibles y riesgos de calidad.\n"
                "2. *Entrenamiento:* pipelines reproducibles con `ColumnTransformer`.\n"
                "3. *Validación:* `TimeSeriesSplit(n_splits=5)` y `GridSearchCV` (KNN y SVR).\n"
                "4. *Evaluación:* MAE, RMSE y R² en test y error por hora y clima.")
    st.warning("**Criterio de éxito (propuesta, acordar antes de experimentar):** MAE en test al "
               "menos 30 % menor que el del DummyRegressor y MAE en horas pico ≤ 2 × MAE global.")
    st.markdown("**Costo del error:** subestimar la demanda en hora pico (usuarios sin bicicleta) "
                "es más grave que sobrestimarla (mover bicicletas de más).")
    st.table(pd.DataFrame(
        [("Específico 1", "EDA y diccionario", "T1, T2, ≥5 gráficos"),
         ("Específico 2", "Pipelines con ColumnTransformer", "T4, código"),
         ("Específico 3", "CV temporal y GridSearchCV", "T5, T6"),
         ("Específico 4", "Evaluación única en test", "T8, residuales, MAE por hora")],
        columns=["Objetivo", "Actividad", "Evidencia"]))

elif etapa == ETAPAS[2]:
    st.header("3. Dataset y diccionario de datos")
    st.markdown("**Dataset:** Bike Sharing Demand · **Fuente:** "
                "[Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/data) · "
                "**Archivo:** `train.csv` · **Licencia:** reglas de la competencia de Kaggle "
                "(citar en el informe).")
    df = obtener_df()
    if df is None:
        st.error("No se encontró un `train.csv` válido. Descárgalo de Kaggle (ver `data/LEEME_DATOS.md`) "
                 "o súbelo desde la barra lateral.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{df.shape[0]:,}")
        c2.metric("Columnas", df.shape[1])
        c3.metric("Rango", f"{df['datetime'].min():%Y-%m-%d} → {df['datetime'].max():%Y-%m-%d}")
        st.subheader("Tipos de datos")
        st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame())
        st.subheader("Vista previa")
        st.dataframe(df.head())
    st.subheader("T1 · Diccionario de variables")
    st.dataframe(DICCIONARIO, hide_index=True)
    st.caption("En uso real, las variables de clima serían pronósticos, no valores observados: "
               "el error real sería mayor que el medido aquí.")

elif etapa == ETAPAS[3]:
    st.header("4. Separación de X e y")
    st.code('y = df["count"]\n'
            'X = df.drop(columns=["count", "casual", "registered", "datetime"])\n'
            'assert X.index.equals(y.index)\n'
            'assert not {"casual", "registered", "count"} & set(X.columns)', language="python")
    df = obtener_df()
    if df is None:
        st.error("Falta un `train.csv` válido (o súbelo desde la barra lateral).")
    else:
        X, y = construir_X_y(df)
        c1, c2 = st.columns(2)
        c1.metric("X (filas, columnas)", str(X.shape))
        c2.metric("y (filas,)", str(y.shape))
        st.write("**Columnas de X:**", ", ".join(X.columns))
        st.success("Comprobaciones superadas: " + ", ".join(k for k, v in verificar(X, y).items() if v))
        st.dataframe(X.head())
    st.subheader("Variables excluidas y razones")
    st.table(tabla_exclusiones())
    st.markdown("`casual` y `registered` suman `count`: usarlas sería fuga directa de información.")


# --------------------------------------------------------------------------- #
# Etapa 5 · EDA
# --------------------------------------------------------------------------- #
elif etapa == ETAPAS[4]:
    st.header("5. Análisis exploratorio (EDA)")
    st.markdown("**Propósito.** Entender el dataset, detectar problemas de calidad y reconocer los patrones "
                "que los modelos deberán capturar. **Regla:** la calidad del archivo (T2) se mide en el CSV "
                "completo porque es descriptiva; todo lo que relaciona los predictores con `count` se "
                "calcula **solo con el tramo de entrenamiento** (80 % inicial) para no mirar el test.")
    df = obtener_df()
    if df is None:
        st.error("Falta un `train.csv` válido (o súbelo desde la barra lateral).")
        st.stop()
    train, test = eda.corte_temporal(df)

    st.subheader("5.1 Calidad de datos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Nulos totales", int(df.isna().sum().sum()))
    c4.metric("Duplicados", int(df.duplicated().sum()))
    st.write("**head()**")
    st.dataframe(df.head())
    st.write("**info() resumido y nulos (%)**")
    st.dataframe(eda.resumen_columnas(df))
    st.write("**describe()**")
    st.dataframe(df.describe().T)
    st.write("**T2 · Diagnóstico de calidad** (la columna de medidas es una propuesta del grupo)")
    st.dataframe(eda.tabla_calidad(df), hide_index=True)
    o = eda.outliers_objetivo(train[eda.TARGET])
    st.caption(f"Valores altos de count (> {o['limite_superior_IQR']:.0f}, regla 1.5·IQR, solo train): "
               f"{o['n_sobre_limite']} filas ({o['pct_sobre_limite']:.1f} %). No se eliminan: son horas pico reales.")

    st.subheader("5.2 Gráficos interpretados (solo entrenamiento)")
    L = eda.lecturas_graficos(train)
    st.markdown("**① Distribución del objetivo**")
    mostrar_grafico(eda.fig_hist_objetivo(train), L["hist"], eda.fig_hist_objetivo)
    st.markdown("**② Serie temporal de `count`**")
    mostrar_grafico(eda.fig_serie_temporal(train), L["serie"], eda.fig_serie_temporal)
    st.markdown("**③ Perfil horario (laborable vs. no laborable)**")
    mostrar_grafico(eda.fig_perfil_horario(train), L["perfil"], eda.fig_perfil_horario)
    st.markdown("**④ Demanda por mes y año**")
    mostrar_grafico(eda.fig_demanda_mes(train), L["mes"], eda.fig_demanda_mes)
    st.markdown("**⑤ Temperatura vs. `count`**")
    mostrar_grafico(eda.fig_temp_vs_count(train), L["temp"], eda.fig_temp_vs_count)
    st.markdown("**⑥ `count` según condición meteorológica**")
    mostrar_grafico(eda.fig_box_clima(train), L["clima"], eda.fig_box_clima)
    st.markdown("**⑦ Correlación entre variables**")
    mostrar_grafico(eda.fig_correlacion(train), L["corr"], eda.fig_correlacion)

    st.markdown(f"**Horas pico (regla del grupo):** horas de días laborables con demanda media > 1.5 × la media "
                f"global del train → `{eda.horas_pico(train)}`. Se usarán en el criterio de éxito y en el MAE por franja.")

    st.subheader("5.3 Cinco hallazgos e implicaciones")
    st.caption("Borrador generado con las cifras del entrenamiento: el equipo debe revisarlo y redactarlo "
               "con sus propias palabras antes de entregarlo.")
    for i, h in enumerate(eda.hallazgos(train), 1):
        st.markdown(f"**{i}. {h['titulo']}**  \n*Evidencia:* {h['evidencia']}  \n*Implicación:* {h['implicacion']}")
    st.info("**Descriptivo vs. ajustado al entrenamiento.** Lo anterior es descriptivo: no cambia ningún "
            "dato. Las decisiones que sí se ajustan con datos (medianas de imputación, media/desvío del "
            "escalado, categorías del OneHotEncoder) se calcularán más adelante dentro del Pipeline, "
            "solo con entrenamiento y con cada fold.")

# --------------------------------------------------------------------------- #
# Etapa 6 · Partición temporal
# --------------------------------------------------------------------------- #
elif etapa == ETAPAS[5]:
    st.header("6. Partición temporal train/test")
    st.markdown("**Propósito.** Reservar el test final **antes** de ajustar cualquier transformación. "
                "Los datos son una serie temporal: el test es el **tramo final** (20 %), nunca una mezcla "
                "aleatoria, porque entrenar con horas futuras para predecir pasadas inflaría el resultado. "
                "Dentro del entrenamiento se usará `TimeSeriesSplit(n_splits=5)`.")
    df = obtener_df()
    if df is None:
        st.error("Falta un `train.csv` válido (o súbelo desde la barra lateral).")
        st.stop()
    st.code('df = df.sort_values("datetime").reset_index(drop=True)\n'
            'corte = int(len(df) * (1 - 0.20))\n'
            'train, test = df.iloc[:corte], df.iloc[corte:]      # sin barajar\n'
            'RANDOM_STATE = 42   # declarado; el corte es determinista y no lo usa', language="python")
    train, test = eda.corte_temporal(df)
    st.pyplot(eda.fig_particion(train, test))
    plt.close("all")
    st.write("**T3 · Partición de datos**")
    st.dataframe(eda.tabla_particion(train, test), hide_index=True)
    ver = eda.verificar_particion(train, test, len(df))
    st.success("Comprobaciones superadas: " + ", ".join(k for k, v in ver.items() if v))
    X_tr, X_te, y_tr, y_te = eda.dividir_X_y(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("X_train", str(X_tr.shape))
    c2.metric("X_test", str(X_te.shape))
    c3.metric("y_train", str(y_tr.shape))
    c4.metric("y_test", str(y_te.shape))

    cob = eda.cobertura_estacional(train, test)
    st.subheader("Qué implica este corte")
    st.markdown(f"- El test cubre los meses {cob['meses_test']} de {cob['anios_test']}. "
                f"Del mismo año, el entrenamiento no tiene datos de los meses "
                f"{cob['meses_test_sin_datos_del_mismo_anio_en_train'] or '—'}: esos meses solo se conocen por "
                f"{cob['anios_train'][0]}.\n"
                "- Por tanto el modelo debe **extrapolar** el nivel de demanda a una época posterior: es el "
                "escenario real de uso y por eso el corte temporal es más exigente (y más honesto) que uno aleatorio.\n"
                "- El test no se usa para elegir modelo ni parámetros; se abrirá una sola vez en la etapa 13.")

    st.subheader("Los 5 folds de TimeSeriesSplit (solo sobre entrenamiento)")
    st.dataframe(eda.tabla_folds(train), hide_index=True)
    st.caption("En cada fold el entrenamiento termina antes de que empiece la validación; los tres modelos "
               "usarán exactamente estos mismos folds (etapas 10–11).")

elif etapa == ETAPAS[6]:
    st.header("7–9. Preprocesamiento y modelos")
    st.markdown("Todo el preprocesamiento se ajusta dentro de cada `Pipeline`: imputación, escalado y "
                "codificación no observan el test ni los folds de validación.")
    df = obtener_df()
    if df is None:
        st.error("Falta `train.csv`.")
        st.stop()
    st.write("**Variables numéricas:**", ", ".join(modelado.NUMERICAS))
    st.write("**Variables categóricas:**", ", ".join(modelado.CATEGORICAS))
    st.code("ColumnTransformer([('numeric', SimpleImputer + StandardScaler),\n"
            "                 ('categorical', SimpleImputer + OneHotEncoder(handle_unknown='ignore'))])\n"
            "Pipeline(preprocesador, estimador)", language="python")
    st.dataframe(pd.DataFrame({"Modelo": ["LinearRegression", "KNN", "SVR", "DummyRegressor"],
                               "Uso": ["candidato", "candidato", "candidato", "referencia"]}))
    st.info("`casual` y `registered` no aparecen en X. El DummyRegressor es referencia y no compite.")

elif etapa == ETAPAS[7]:
    st.header("10–12. Validación, GridSearch y selección")
    df = obtener_df()
    if df is None:
        st.error("Falta `train.csv`.")
        st.stop()
    exp = ejecutar_modelado(df)
    st.subheader("T5 · TimeSeriesSplit(n_splits=5)")
    st.dataframe(exp["cv"].round(3), hide_index=True)
    st.subheader("T6 · GridSearchCV")
    st.dataframe(exp["grids"], hide_index=True)
    for nombre, top in exp["grid_top"].items():
        st.caption(f"Top 5 combinaciones de {nombre}")
        st.dataframe(top, hide_index=True)
    st.subheader("T7 · Selección previa al test")
    st.success(f"Ganador elegido sin abrir el test: **{exp['winner']}**, por menor MAE medio de CV.")
    st.write("La selección prioriza MAE porque cada unidad representa una bicicleta de error y porque "
             "la distribución combina horas de baja demanda con horas pico. Se compararon los mismos cinco "
             "folds cronológicos para todos los modelos, de modo que cada candidato enfrentara exactamente "
             "la misma dificultad temporal. La desviación del MAE complementa el promedio: un valor bajo "
             "con variación alta sería menos confiable cuando cambia el periodo observado. KNN y SVR se "
             "ajustaron con rejillas pequeñas y razonables, ya que ampliar muchas combinaciones aumentaría "
             "el costo sin garantía de mejorar la generalización. LinearRegression se mantuvo como referencia "
             "interpretable y DummyRegressor como línea base no competitiva. El ganador se determina con "
             "el MAE de validación, no con el resultado que eventualmente produzca el test. Por eso el "
             "test permanece completamente reservado hasta la etapa 13 y no influye en la elección ni en "
             "los hiperparámetros.")

elif etapa == ETAPAS[8]:
    st.header("13–14. Evaluación final y análisis crítico")
    st.warning("Esta pantalla abre el test una sola vez, después de seleccionar el modelo.")
    df = obtener_df()
    if df is None:
        st.error("Falta `train.csv`.")
        st.stop()
    evaluacion = ejecutar_evaluacion(df)
    st.subheader("T8 · Métricas en test")
    cols = st.columns(3)
    for col, (nombre, valor) in zip(cols, evaluacion["metrics"].items()):
        col.metric(nombre, f"{valor:.3f}")
    detail = evaluacion["detail"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(detail["count"], detail["prediccion"], s=8, alpha=.3)
    lim = [0, max(detail["count"].max(), detail["prediccion"].max())]
    ax.plot(lim, lim, "r--")
    ax.set(xlabel="Real", ylabel="Predicho", title="Real vs. predicho en test")
    st.pyplot(fig)
    plt.close(fig)
    st.write("**MAE por hora**")
    st.dataframe(evaluacion["by_hour"].round(3), hide_index=True)
    st.write("**MAE por condición meteorológica**")
    st.dataframe(evaluacion["by_weather"].round(3), hide_index=True)
    st.write("**Primeras 10 observaciones: valor real, predicción y residual**")
    st.dataframe(
        detail[["datetime", "count", "prediccion", "residual"]].head(10).round(2),
        hide_index=True,
    )
    st.markdown("**Lectura crítica.** El resultado estima desempeño para el tramo final de este CSV, "
                "no garantiza producción. El dataset omite días posteriores al 19 de cada mes, el clima "
                "usado es observado y no un pronóstico, y el cambio interanual obliga a extrapolar. "
                "También pueden existir sesgos por horas o condiciones con pocos casos. Antes de desplegar "
                "se necesitaría validar con datos recientes y pronósticos meteorológicos reales.")

elif etapa == ETAPAS[9]:
    st.header("15–16. Cinco predicciones y conclusiones")
    df = obtener_df()
    if df is None:
        st.error("Falta `train.csv`.")
        st.stop()
    exp = ejecutar_modelado(df)
    exp["model"].fit(exp["x_train"], exp["y_train"])
    evaluacion = {"model": exp["model"]}
    st.subheader("T9 · Escenarios futuros")
    st.dataframe(modelado.cinco_predicciones(exp, evaluacion).round(2), hide_index=True)
    st.caption("Son cinco horas posteriores al último registro de entrenamiento, con las últimas "
               "condiciones meteorológicas disponibles. No tienen `count` real y no se usan como prueba.")
    st.subheader("Predicción interactiva")
    st.caption("El formulario usa el mismo pipeline entrenado y no reentrena ni consulta el test.")
    with st.form("formulario_prediccion"):
        fecha_hora = st.text_input("Fecha y hora (YYYY-MM-DD HH:MM)", "2012-08-06 08:00")
        c1, c2, c3 = st.columns(3)
        season = c1.selectbox("Season", [1, 2, 3, 4], index=2)
        weather = c2.selectbox("Weather", [1, 2, 3, 4], index=0)
        workingday = c3.selectbox("Working day", [0, 1], index=1)
        c4, c5, c6 = st.columns(3)
        temp = c4.number_input("Temperatura", 0.0, 45.0, 20.0)
        atemp = c5.number_input("Sensación térmica", 0.0, 50.0, 23.0)
        humidity = c6.number_input("Humedad", 0.0, 100.0, 60.0)
        c7, c8 = st.columns(2)
        windspeed = c7.number_input("Velocidad del viento", 0.0, 70.0, 10.0)
        holiday = c8.selectbox("Holiday", [0, 1], index=0)
        submitted = st.form_submit_button("Predecir")
    if submitted:
        try:
            fecha = pd.to_datetime(fecha_hora)
            entrada = pd.DataFrame([{
                "datetime": fecha, "season": season, "holiday": holiday,
                "workingday": workingday, "weather": weather, "temp": temp,
                "atemp": atemp, "humidity": humidity, "windspeed": windspeed,
                "count": 0,
            }])
            entrada_x, _ = construir_X_y(entrada)
            valor = float(exp["model"].predict(entrada_x)[0])
            st.metric("Demanda estimada", f"{max(0, valor):.1f} bicicletas")
        except (TypeError, ValueError) as exc:
            st.error(f"Entrada inválida: {exc}")
    st.subheader("Conclusiones y mejoras")
    st.markdown("- La validación temporal refleja mejor el uso futuro que una mezcla aleatoria.\n"
                "- MAE es la métrica principal porque es interpretable en bicicletas.\n"
                "- El error debe revisarse por hora y clima, no solo con un promedio global.\n"
                "- El modelo no debe considerarse productivo sin pronósticos meteorológicos y validación reciente.\n"
                "- Mejoras: incorporar rezagos disponibles en producción, probar modelos con tendencia temporal "
                "regularizada y evaluar periódicamente el drift.")
