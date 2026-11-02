"""Genera TODA la evidencia de la Fase 2 a partir de data/train.csv:
  docs/figuras/*.png       gráficos del EDA y de la partición
  docs/tablas/*.csv        T2, T3, folds, resumen de columnas, describe, X.head()
  docs/informe_fase2_etapas5_6.md   texto listo para llevar al informe (números calculados)

Uso:  python src/generar_evidencias_fase2.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import fase2_eda as eda
from fase1_xy import TARGET, cargar_datos, construir_X_y

RAIZ = Path(__file__).resolve().parent.parent
FIG, TAB = RAIZ / "docs" / "figuras", RAIZ / "docs" / "tablas"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def md_tabla(df: pd.DataFrame) -> str:
    """DataFrame -> tabla markdown (sin depender de tabulate)."""
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    filas = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def main():
    df = cargar_datos()
    eda.validar_csv(df)
    train, test = eda.corte_temporal(df)
    ver = eda.verificar_particion(train, test, len(df))
    X, y = construir_X_y(df)
    Xtr, Xte, ytr, yte = eda.dividir_X_y(df)
    d = eda.agregar_calendario(train)
    pico = eda.horas_pico(train)
    assert pico == eda.HORAS_PICO, f"HORAS_PICO desactualizado: {pico}"

    # ---------------- figuras ----------------
    figs = {
        "01_hist_objetivo": eda.fig_hist_objetivo(train),
        "02_serie_temporal": eda.fig_serie_temporal(train),
        "03_perfil_horario": eda.fig_perfil_horario(train),
        "04_demanda_mes": eda.fig_demanda_mes(train),
        "05_temp_vs_count": eda.fig_temp_vs_count(train),
        "06_count_por_clima": eda.fig_box_clima(train),
        "07_correlacion": eda.fig_correlacion(train),
        "08_particion": eda.fig_particion(train, test),
    }
    for nombre, fig in figs.items():
        fig.savefig(FIG / f"{nombre}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------------- tablas ----------------
    t2 = eda.tabla_calidad(df)
    t3 = eda.tabla_particion(train, test)
    folds = eda.tabla_folds(train)
    resumen = eda.resumen_columnas(df).reset_index(names="columna")
    desc = df.describe().T.round(2).reset_index(names="columna")
    for nombre, t in {"T2_calidad": t2, "T3_particion": t3, "folds_timeseriessplit": folds,
                      "resumen_columnas": resumen, "describe": desc, "X_head": Xtr.head(),
                      "hallazgos": pd.DataFrame(eda.hallazgos(train))}.items():
        t.to_csv(TAB / f"{nombre}.csv", index=False, encoding="utf-8-sig")

    # ---------------- cifras para el texto ----------------
    n, ntr, nte = len(df), len(train), len(test)
    cob = eda.cobertura_estacional(train, test)
    o = eda.outliers_objetivo(train[TARGET])
    yy = train[TARGET]
    hz = eda.huecos_serie(df)
    mh = d.groupby("hour")[TARGET].mean()
    ph = d.groupby(["workingday", "hour"])[TARGET].mean()
    lab, nol = ph.loc[1], ph.loc[0]
    mes_anio = d.groupby(["month", "year"])[TARGET].mean().unstack()
    razon = (mes_anio[2012] / mes_anio[2011]).dropna()
    daily = train.set_index("datetime")[TARGET].resample("D").mean()
    tb = train.groupby(pd.cut(train["temp"], [0, 5, 10, 15, 20, 25, 30, 35, 42]), observed=True)[TARGET].mean()
    rt = train.loc[(train['temp'] > 20) & (train['temp'] <= 25), TARGET]
    mw = d.groupby("weather")[TARGET].agg(["mean", "count"])
    corr = pd.concat([X.loc[train.index], y.loc[train.index]], axis=1).corr(numeric_only=True)[TARGET]
    corr_ta = train["temp"].corr(train["atemp"])
    corr_sm = d["season"].corr(d["month"])
    corr_wd = d["workingday"].corr(d["dayofweek"])
    wtest = test["weather"].value_counts().sort_index()
    h = eda.hallazgos(train)
    min_pos_wind = df.loc[df["windspeed"] > 0, "windspeed"].min()
    dia_hum0 = ", ".join(sorted({f"{x:%Y-%m-%d}" for x in df.loc[df["humidity"] == 0, "datetime"]}))
    n_w0, n_h0 = int((df["windspeed"] == 0).sum()), int((df["humidity"] == 0).sum())

    texto = f"""# Informe · Etapas 5 y 6 (EDA y partición temporal) — Grupo 04

> Texto listo para llevar al informe. **Todas las cifras se calcularon con el `train.csv` real**
> (`python src/generar_evidencias_fase2.py` lo regenera). Los gráficos están en `docs/figuras/` y las tablas
> en `docs/tablas/`. Reescríbanlo con sus palabras donde lo necesiten: la guía pide interpretación del equipo.

---

## 5. Análisis exploratorio de datos (etapa 5)

**Criterio metodológico.** La calidad del archivo (T2) se mide en el CSV completo porque es descriptiva (cuenta
nulos, duplicados y valores raros). Todo gráfico o correlación que relaciona los predictores con `count` se calcula
**solo con el entrenamiento** ({ntr:,} filas, el 80 % inicial), para no mirar el test antes de decidir nada.

### 5.1 Estructura y calidad de los datos

El archivo `train.csv` tiene **{n:,} filas y {df.shape[1]} columnas** (`datetime`, `season`, `holiday`,
`workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed`, `casual`, `registered`, `count`), entre
{df['datetime'].min():%Y-%m-%d %H:%M} y {df['datetime'].max():%Y-%m-%d %H:%M}. No tiene valores nulos ni filas
duplicadas, y cada `datetime` es único. Los tipos son enteros para las variables de calendario y clima
discreto, y decimales para `temp`, `atemp` y `windspeed`. En el 100 % de las filas `casual + registered = count`,
lo que confirma que ambas columnas son componentes del objetivo y deben excluirse (fuga directa).

**T2 · Diagnóstico de calidad**

{md_tabla(t2)}

**Interpretación de T2.**
1. *Sin nulos ni duplicados:* no hay nada que eliminar; se mantiene `SimpleImputer` dentro del Pipeline solo como
   salvaguarda para datos nuevos del formulario.
2. *Días ausentes:* solo existen los días {hz['dia_del_mes_min']}–{hz['dia_del_mes_max']} de cada mes
   ({hz['dias_ausentes']} de {hz['dias_ausentes'] + hz['dias_presentes']} días del rango no tienen registros);
   esto proviene del diseño de la competencia, que reserva el resto de cada mes para `test.csv`. Además faltan
   {hz['horas_faltantes_en_dias_presentes']} horas sueltas dentro de días presentes. Como ningún modelo usa rezagos
   de `count`, no se rellenan; pero implica que la serie no es continua y que los folds y el corte se definen por
   orden cronológico, no por conteo de días.
3. *`humidity = 0`:* {n_h0} filas, todas en {dia_hum0}; es físicamente imposible y parece una falla del sensor ese
   día. Afecta al {100 * n_h0 / n:.2f} % de los datos; se conserva y se declara como limitación.
4. *`windspeed = 0`:* {n_w0:,} filas ({100 * n_w0 / n:.1f} %). Entre 0 y el primer valor positivo
   ({min_pos_wind:.4f}) no hay ningún valor, lo que sugiere que el 0 codifica "no medido" o "por debajo de la
   resolución del sensor". Se conserva sin imputar (una imputación ajustada con todo el CSV sería fuga) y se declara
   como limitación; en la Fase 3 puede compararse contra una versión imputada dentro del Pipeline.
5. *Valores altos de `count`:* con la regla 1.5·IQR (límite {o['limite_superior_IQR']:.0f}) hay
   {o['n_sobre_limite']} filas en el entrenamiento ({o['pct_sobre_limite']:.1f} %). No se eliminan: son horas pico
   reales que el modelo debe aprender a anticipar.

**Resumen por columna (tipo, nulos, % de nulos, únicos)**

{md_tabla(resumen)}

### 5.2 Gráficos interpretados (solo entrenamiento)

**Figura 1 · Distribución del objetivo** (`docs/figuras/01_hist_objetivo.png`)

Media {yy.mean():.1f}, mediana {yy.median():.0f}, desviación {yy.std():.1f}, máximo {yy.max():.0f} y asimetría
{yy.skew():.2f}. El {100 * (yy <= 10).mean():.1f} % de las horas tiene `count ≤ 10` (madrugadas) y el 10 % más alto
supera {yy.quantile(.9):.0f}. *Interpretación:* la distribución es asimétrica a la derecha, con muchas horas de
baja demanda y una cola de horas pico. Por eso la media ({yy.mean():.0f}) sobrestima la hora típica ({yy.median():.0f}), el MAE es una
métrica más robusta que el RMSE como criterio principal, y un error global único ocultaría el comportamiento
distinto de las horas pico y valle.

**Figura 2 · Serie temporal de `count`** (`02_serie_temporal.png`)

La demanda media diaria oscila entre {daily.min():.0f} y {daily.max():.0f} alquileres por hora. *Interpretación:*
hay un ciclo estacional (ver la figura 4: mínimos en invierno, máximos en verano) y un nivel general más alto en
2012 que en 2011. La serie tiene saltos por los días ausentes (días 20–31 de cada mes). Esto justifica incluir
`month` y `year`, y obliga a validar de forma cronológica: mezclar días al azar dejaría que el modelo "viera" días
vecinos del futuro.

**Figura 3 · Perfil horario, laborable vs. no laborable** (`03_perfil_horario.png`)

En días laborables hay dos picos: 8:00 ({lab[8]:.0f}) y 17:00–18:00 ({lab[17]:.0f} y {lab[18]:.0f}); entre
medio la demanda cae (10:00: {lab[10]:.0f}). En días no laborables hay una sola
meseta de 11:00 a 16:00 (máximo a las {nol.idxmax()}:00 con {nol.max():.0f}). De madrugada (3–5 h) la demanda es
casi nula. *Interpretación:* el patrón diario cambia según el tipo de día (conmuta al trabajo vs. paseo). Es una
interacción **hora × workingday**: un modelo lineal sin términos de interacción no la captura, mientras que KNN y
SVR con kernel no lineal sí pueden. Con la regla "hora laborable con demanda media > 1.5 × la media global" las
**horas pico son {pico}**, definición que se usará en el criterio de éxito y en el análisis de MAE por franja.

**Figura 4 · Demanda media por mes y año** (`04_demanda_mes.png`)

En 2011 (año completo) la demanda crece de {mes_anio.loc[1, 2011]:.0f} en enero a {mes_anio.loc[7, 2011]:.0f} en
julio y baja a {mes_anio.loc[12, 2011]:.0f} en diciembre. En 2012, para los meses comunes (enero–agosto), la
demanda es entre {razon.min():.2f} y {razon.max():.2f} veces la de 2011. *Interpretación:* coexisten estacionalidad
y crecimiento interanual del servicio. El test (agosto–diciembre de 2012) caerá en un nivel más alto que gran
parte del entrenamiento, así que el modelo deberá **extrapolar**; es la situación real de uso, pero anticipa
errores mayores que en una división aleatoria.

**Figura 5 · Temperatura vs. `count`** (`05_temp_vs_count.png`)

La media de `count` por tramos de temperatura sube de {tb.iloc[0]:.0f} (0–5 °C) a {tb.iloc[-1]:.0f}
(35–42 °C); la correlación lineal es {corr['temp']:.2f}. El crecimiento es sostenido hasta los 30–35 °C y se aplana en el último tramo. *Interpretación:* hay relación positiva pero con mucha dispersión vertical: a una misma temperatura (tramo 20–25 °C)
la demanda va de {rt.min():.0f} a {rt.max():.0f} según la hora. La temperatura sola explica poco; actúa combinada con la hora.

**Figura 6 · `count` según condición meteorológica** (`06_count_por_clima.png`)

Medias: clima 1 = {mw.loc[1, 'mean']:.0f} (n={int(mw.loc[1, 'count']):,}), clima 2 = {mw.loc[2, 'mean']:.0f}
(n={int(mw.loc[2, 'count']):,}), clima 3 = {mw.loc[3, 'mean']:.0f} (n={int(mw.loc[3, 'count'])}); el clima 4
tiene **{int(mw.loc[4, 'count'])} sola observación** en todo el entrenamiento. *Interpretación:* a peor clima,
menor demanda (el clima 3 rinde {100 * (1 - mw.loc[3, 'mean'] / mw.loc[1, 'mean']):.0f} % menos que el 1). La
categoría 4 no es analizable: su media no significa nada y el test no contiene ningún caso (solo climas
{', '.join(map(str, wtest.index))}). Por eso el `OneHotEncoder` debe usar `handle_unknown="ignore"` y el análisis de
MAE por clima solo podrá hacerse con 3 categorías, indicando el n de cada una.

**Figura 7 · Matriz de correlación** (`07_correlacion.png`)

Correlaciones con `count`: temp {corr['temp']:.2f}, atemp {corr['atemp']:.2f}, hora {corr['hour']:.2f}, humedad
{corr['humidity']:.2f}, año {corr['year']:.2f}, clima {corr['weather']:.2f}, viento {corr['windspeed']:.2f};
`workingday` {corr['workingday']:.2f}, `dayofweek` {corr['dayofweek']:.2f} y `holiday` {corr['holiday']:.2f}.
Además, corr(temp, atemp) = {corr_ta:.2f}, corr(season, month) = {corr_sm:.2f} y corr(workingday, dayofweek) =
{corr_wd:.2f}. *Interpretación:* las variables de calendario casi no se correlacionan
linealmente con `count` aunque la figura 3 muestra que sí importan: su efecto es de interacción, que Pearson no
detecta. `temp` y `atemp` son prácticamente la misma señal (multicolinealidad): no afecta la capacidad predictiva de
la regresión lineal pero inestabiliza sus coeficientes; se conservan ambas por ahora (decisión de la Fase 1) y se
puede probar quitar `atemp` en la Fase 3. `season` y `month` también son casi redundantes (mismo calendario
anual); se conservan porque `month` da más resolución, y la redundancia no perjudica a KNN/SVR de forma relevante.

### 5.3 Cinco hallazgos e implicaciones

| # | Hallazgo | Evidencia | Implicación para el modelado |
|---|---|---|---|
""" + "\n".join(f"| {i} | {x['titulo']} | {x['evidencia']} | {x['implicacion']} |" for i, x in enumerate(h, 1)) + f"""

### 5.4 Descriptivo vs. decisiones ajustadas al entrenamiento

Todo lo anterior es **descriptivo**: no modifica ningún dato. Las decisiones que sí se ajustan con datos (mediana
de imputación, media y desvío del escalado, categorías del `OneHotEncoder`) se calcularán más adelante **dentro del
Pipeline, solo con el entrenamiento y con cada fold**, sin tocar el test. Además, los gráficos y correlaciones de
esta sección usan únicamente el entrenamiento.

---

## 6. Partición y protocolo experimental (etapa 6)

### 6.1 Método

El conjunto se ordena por `datetime` y se corta por posición: el **80 % inicial** es entrenamiento y el
**20 % final** es prueba (test). No se baraja nada. `RANDOM_STATE = 42` queda declarado, pero el corte es
determinista y no lo utiliza. Las variables de calendario (`hour`, `month`, `dayofweek`, `year`) se derivan de
`datetime` fila a fila (no consultan otras filas, por lo que no filtran información del futuro).

```python
df = df.sort_values("datetime").reset_index(drop=True)
corte = int(len(df) * (1 - 0.20))
train, test = df.iloc[:corte], df.iloc[corte:]   # sin barajar
```

**T3 · Partición de datos**

{md_tabla(t3[['Conjunto', 'Registros', 'Proporción', 'Desde', 'Hasta', 'Método']])}

Semilla: no aplica (corte determinista); `RANDOM_STATE = 42` declarado.

**Comprobaciones superadas:** {', '.join(k for k, v in ver.items() if v)}.
Dimensiones tras separar X e y: `X_train` {Xtr.shape}, `X_test` {Xte.shape}, `y_train` {ytr.shape},
`y_test` {yte.shape}.

### 6.2 Justificación del esquema

Los datos son una serie temporal horaria: el consumo de una hora se parece al de las horas y días vecinos. Con una
división aleatoria, el modelo se entrenaría con horas posteriores a las que luego debe predecir y el resultado en
test quedaría inflado. Con el corte cronológico, el test representa el uso real: predecir un período que viene
**después** de todo lo que el modelo conoce. El test se reserva **antes** de ajustar cualquier transformación y se
abrirá una sola vez, en la etapa 13, con el modelo ya elegido.

### 6.3 Qué implica este corte

- El test cubre {test['datetime'].min():%Y-%m-%d} a {test['datetime'].max():%Y-%m-%d}
  (meses {cob['meses_test']} de {cob['anios_test'][0]}). Los meses
  {cob['meses_test_sin_datos_del_mismo_anio_en_train']} solo aparecen en el entrenamiento de {cob['anios_train'][0]}:
  el modelo debe trasladar a 2012 lo aprendido de esa época de 2011, y por eso `year` pesa en el resultado.
- El test no incluye clima 4 (solo {', '.join(f'clima {k}: {v}' for k, v in wtest.items())} filas) y el
  entrenamiento tiene una sola fila de clima 4.
- Por la tendencia creciente, se espera que los errores absolutos (MAE) del test sean mayores que los de
  validación en los primeros folds.

### 6.4 Los cinco folds de `TimeSeriesSplit(n_splits=5)` sobre el entrenamiento

{md_tabla(folds)}

En cada fold el entrenamiento termina antes de que empiece la validación (verificado con un `assert`), y los
tres modelos usarán exactamente estos mismos folds. *Comentario de estabilidad esperada:* el fold 1 entrena con solo
{folds.iloc[0, 1]:,} filas (enero–abril de 2011) y debe predecir abril–julio, una época que no vio; además el
`count` medio de las ventanas de validación varía entre {folds['count medio validación'].min():.0f} y
{folds['count medio validación'].max():.0f}. Por tanto no debe sorprender que el MAE cambie mucho entre folds:
refleja diferencias de nivel y estación, no necesariamente un modelo inestable.

---

## Decisiones adoptadas en la Fase 2 (revisarlas en grupo)

| Tema | Decisión | Justificación |
|---|---|---|
| EDA | Gráficos y correlaciones solo con train | No mirar el test |
| Horas pico | {pico} | Regla: demanda media laborable > 1.5 × media global (train) |
| `windspeed = 0` | Conservar y declarar limitación | No hay valores entre 0 y {min_pos_wind:.2f}; imputar con todo el CSV sería fuga |
| `humidity = 0` | Conservar y declarar limitación | {n_h0} filas de un solo día ({dia_hum0}) |
| `temp` y `atemp` | Conservar ambas | Redundancia declarada (corr {corr_ta:.2f}); probar quitar `atemp` en Fase 3 |
| `season` y `month` | Conservar ambas | Redundancia declarada (corr {corr_sm:.2f}); `month` aporta más resolución |
| Clima 4 | Conservar; `handle_unknown="ignore"` | 1 fila en train, 0 en test |
| Partición | 80/20 cronológico + TimeSeriesSplit(5) | Obligatorio para el grupo |

## Pendiente del grupo antes de la Fase 3
- [ ] **Acordar el criterio de éxito** de la etapa 2 (propuesta: MAE de test ≥ 30 % menor que el DummyRegressor y
      MAE en horas pico ≤ 2 × MAE global). Debe quedar fijado antes de entrenar.
- [ ] Completar la ficha del dataset (fecha de consulta, licencia/reglas de Kaggle).
- [ ] Revisar y reescribir con sus palabras las interpretaciones de este documento.
- [ ] Guardar las capturas de la app (pantallas 5 y 6) y del notebook `02_fase2.ipynb`.
"""
    (RAIZ / "docs" / "informe_fase2_etapas5_6.md").write_text(texto, encoding="utf-8")
    print("Evidencias generadas en docs/figuras, docs/tablas y docs/informe_fase2_etapas5_6.md")


if __name__ == "__main__":
    main()
