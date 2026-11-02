# Informe · Etapas 5 y 6 (EDA y partición temporal) — Grupo 04

> Texto listo para llevar al informe. **Todas las cifras se calcularon con el `train.csv` real**
> (`python src/generar_evidencias_fase2.py` lo regenera). Los gráficos están en `docs/figuras/` y las tablas
> en `docs/tablas/`. Reescríbanlo con sus palabras donde lo necesiten: la guía pide interpretación del equipo.

---

## 5. Análisis exploratorio de datos (etapa 5)

**Criterio metodológico.** La calidad del archivo (T2) se mide en el CSV completo porque es descriptiva (cuenta
nulos, duplicados y valores raros). Todo gráfico o correlación que relaciona los predictores con `count` se calcula
**solo con el entrenamiento** (8,708 filas, el 80 % inicial), para no mirar el test antes de decidir nada.

### 5.1 Estructura y calidad de los datos

El archivo `train.csv` tiene **10,886 filas y 12 columnas** (`datetime`, `season`, `holiday`,
`workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed`, `casual`, `registered`, `count`), entre
2011-01-01 00:00 y 2012-12-19 23:00. No tiene valores nulos ni filas
duplicadas, y cada `datetime` es único. Los tipos son enteros para las variables de calendario y clima
discreto, y decimales para `temp`, `atemp` y `windspeed`. En el 100 % de las filas `casual + registered = count`,
lo que confirma que ambas columnas son componentes del objetivo y deben excluirse (fuga directa).

**T2 · Diagnóstico de calidad**

| Chequeo | Cantidad | Porcentaje | Medida adoptada (propuesta) |
|---|---|---|---|
| Valores nulos (todas las columnas) | 0 | 0.00 % | Sin nulos que tratar. Se mantiene SimpleImputer dentro del Pipeline como salvaguarda. |
| Filas duplicadas | 0 | 0.00 % | Ninguna acción. |
| datetime repetido | 0 | 0.00 % | Ninguna acción: una fila por hora. |
| Horas ausentes dentro de días presentes | 58 | 0.53 % | Se documenta. No se rellenan: los modelos no usan rezagos de count. |
| Días calendario sin registros (entre primer y último dato) | 263 | 263 de 719 días | Solo existen los días 1–19 de cada mes (los demás días los reserva Kaggle para test.csv, que no trae count). No se rellenan; el corte y los folds se hacen por orden cronológico. |
| humidity = 0 | 22 | 0.20 % | Valor físicamente imposible, concentrado en pocos días: se conserva y se declara como limitación (efecto mínimo). |
| windspeed = 0 | 1313 | 12.06 % | Posible dato no medido (no hay valores entre 0 y el mínimo positivo): se conserva sin imputar y se declara como limitación. |
| count = 0 | 0 | 0.00 % | Plausible (madrugada): se conserva. |
| Valores negativos (humidity, windspeed, count) | 0 | 0.00 % | Ninguna acción. |
| Categorías fuera de rango (season, weather) | 0 | 0.00 % | Ninguna acción. |
| casual + registered ≠ count (filas) | 0 | 0.00 % | Confirma que casual y registered son componentes de count: EXCLUIDAS por fuga. |

**Interpretación de T2.**
1. *Sin nulos ni duplicados:* no hay nada que eliminar; se mantiene `SimpleImputer` dentro del Pipeline solo como
   salvaguarda para datos nuevos del formulario.
2. *Días ausentes:* solo existen los días 1–19 de cada mes
   (263 de 719 días del rango no tienen registros);
   esto proviene del diseño de la competencia, que reserva el resto de cada mes para `test.csv`. Además faltan
   58 horas sueltas dentro de días presentes. Como ningún modelo usa rezagos
   de `count`, no se rellenan; pero implica que la serie no es continua y que los folds y el corte se definen por
   orden cronológico, no por conteo de días.
3. *`humidity = 0`:* 22 filas, todas en 2011-03-10; es físicamente imposible y parece una falla del sensor ese
   día. Afecta al 0.20 % de los datos; se conserva y se declara como limitación.
4. *`windspeed = 0`:* 1,313 filas (12.1 %). Entre 0 y el primer valor positivo
   (6.0032) no hay ningún valor, lo que sugiere que el 0 codifica "no medido" o "por debajo de la
   resolución del sensor". Se conserva sin imputar (una imputación ajustada con todo el CSV sería fuga) y se declara
   como limitación; en la Fase 3 puede compararse contra una versión imputada dentro del Pipeline.
5. *Valores altos de `count`:* con la regla 1.5·IQR (límite 590) hay
   239 filas en el entrenamiento (2.7 %). No se eliminan: son horas pico
   reales que el modelo debe aprender a anticipar.

**Resumen por columna (tipo, nulos, % de nulos, únicos)**

| columna | dtype | no_nulos | nulos | %_nulos | únicos |
|---|---|---|---|---|---|
| datetime | datetime64[ns] | 10886 | 0 | 0.0 | 10886 |
| season | int64 | 10886 | 0 | 0.0 | 4 |
| holiday | int64 | 10886 | 0 | 0.0 | 2 |
| workingday | int64 | 10886 | 0 | 0.0 | 2 |
| weather | int64 | 10886 | 0 | 0.0 | 4 |
| temp | float64 | 10886 | 0 | 0.0 | 49 |
| atemp | float64 | 10886 | 0 | 0.0 | 60 |
| humidity | int64 | 10886 | 0 | 0.0 | 89 |
| windspeed | float64 | 10886 | 0 | 0.0 | 28 |
| casual | int64 | 10886 | 0 | 0.0 | 309 |
| registered | int64 | 10886 | 0 | 0.0 | 731 |
| count | int64 | 10886 | 0 | 0.0 | 822 |

### 5.2 Gráficos interpretados (solo entrenamiento)

**Figura 1 · Distribución del objetivo** (`docs/figuras/01_hist_objetivo.png`)

Media 174.4, mediana 130, desviación 166.4, máximo 891 y asimetría
1.28. El 12.1 % de las horas tiene `count ≤ 10` (madrugadas) y el 10 % más alto
supera 411. *Interpretación:* la distribución es asimétrica a la derecha, con muchas horas de
baja demanda y una cola de horas pico. Por eso la media (174) sobrestima la hora típica (130), el MAE es una
métrica más robusta que el RMSE como criterio principal, y un error global único ocultaría el comportamiento
distinto de las horas pico y valle.

**Figura 2 · Serie temporal de `count`** (`02_serie_temporal.png`)

La demanda media diaria oscila entre 26 y 346 alquileres por hora. *Interpretación:*
hay un ciclo estacional (ver la figura 4: mínimos en invierno, máximos en verano) y un nivel general más alto en
2012 que en 2011. La serie tiene saltos por los días ausentes (días 20–31 de cada mes). Esto justifica incluir
`month` y `year`, y obliga a validar de forma cronológica: mezclar días al azar dejaría que el modelo "viera" días
vecinos del futuro.

**Figura 3 · Perfil horario, laborable vs. no laborable** (`03_perfil_horario.png`)

En días laborables hay dos picos: 8:00 (433) y 17:00–18:00 (479 y 450); entre
medio la demanda cae (10:00: 121). En días no laborables hay una sola
meseta de 11:00 a 16:00 (máximo a las 13:00 con 352). De madrugada (3–5 h) la demanda es
casi nula. *Interpretación:* el patrón diario cambia según el tipo de día (conmuta al trabajo vs. paseo). Es una
interacción **hora × workingday**: un modelo lineal sin términos de interacción no la captura, mientras que KNN y
SVR con kernel no lineal sí pueden. Con la regla "hora laborable con demanda media > 1.5 × la media global" las
**horas pico son [7, 8, 17, 18, 19]**, definición que se usará en el criterio de éxito y en el análisis de MAE por franja.

**Figura 4 · Demanda media por mes y año** (`04_demanda_mes.png`)

En 2011 (año completo) la demanda crece de 55 en enero a 204 en
julio y baja a 134 en diciembre. En 2012, para los meses comunes (enero–agosto), la
demanda es entre 1.31 y 2.40 veces la de 2011. *Interpretación:* coexisten estacionalidad
y crecimiento interanual del servicio. El test (agosto–diciembre de 2012) caerá en un nivel más alto que gran
parte del entrenamiento, así que el modelo deberá **extrapolar**; es la situación real de uso, pero anticipa
errores mayores que en una división aleatoria.

**Figura 5 · Temperatura vs. `count`** (`05_temp_vs_count.png`)

La media de `count` por tramos de temperatura sube de 53 (0–5 °C) a 325
(35–42 °C); la correlación lineal es 0.43. El crecimiento es sostenido hasta los 30–35 °C y se aplana en el último tramo. *Interpretación:* hay relación positiva pero con mucha dispersión vertical: a una misma temperatura (tramo 20–25 °C)
la demanda va de 1 a 850 según la hora. La temperatura sola explica poco; actúa combinada con la hora.

**Figura 6 · `count` según condición meteorológica** (`06_count_por_clima.png`)

Medias: clima 1 = 188 (n=5,858), clima 2 = 160
(n=2,142), clima 3 = 103 (n=707); el clima 4
tiene **1 sola observación** en todo el entrenamiento. *Interpretación:* a peor clima,
menor demanda (el clima 3 rinde 45 % menos que el 1). La
categoría 4 no es analizable: su media no significa nada y el test no contiene ningún caso (solo climas
1, 2, 3). Por eso el `OneHotEncoder` debe usar `handle_unknown="ignore"` y el análisis de
MAE por clima solo podrá hacerse con 3 categorías, indicando el n de cada una.

**Figura 7 · Matriz de correlación** (`07_correlacion.png`)

Correlaciones con `count`: temp 0.43, atemp 0.43, hora 0.40, humedad
-0.33, año 0.23, clima -0.14, viento 0.10;
`workingday` 0.00, `dayofweek` 0.01 y `holiday` -0.01.
Además, corr(temp, atemp) = 0.99, corr(season, month) = 0.97 y corr(workingday, dayofweek) =
-0.71. *Interpretación:* las variables de calendario casi no se correlacionan
linealmente con `count` aunque la figura 3 muestra que sí importan: su efecto es de interacción, que Pearson no
detecta. `temp` y `atemp` son prácticamente la misma señal (multicolinealidad): no afecta la capacidad predictiva de
la regresión lineal pero inestabiliza sus coeficientes; se conservan ambas por ahora (decisión de la Fase 1) y se
puede probar quitar `atemp` en la Fase 3. `season` y `month` también son casi redundantes (mismo calendario
anual); se conservan porque `month` da más resolución, y la redundancia no perjudica a KNN/SVR de forma relevante.

### 5.3 Cinco hallazgos e implicaciones

| # | Hallazgo | Evidencia | Implicación para el modelado |
|---|---|---|---|
| 1 | Forma de la distribución del objetivo | media = 174.4, mediana = 130.0, máximo = 891, asimetría = 1.28; 12.1 % de las horas tiene count ≤ 10. | La cola derecha eleva la media sobre la mediana: MAE es más robusto que RMSE como métrica primaria y conviene analizar el error por franja horaria, no solo el global. |
| 2 | Patrón horario | La hora explica 50.5 % de la varianza de count (mes: 8.9 %, clima: 2.2 %). Horas de mayor demanda media: laborables [8, 17, 18]; no laborables [13, 14, 15]. | Las horas pico difieren entre días laborables y no laborables: hay interacción hora × workingday que un modelo lineal sin términos de interacción no captura. Horas pico (laborables, demanda media > 1.5 × la global): [7, 8, 17, 18, 19]. |
| 3 | Estacionalidad y nivel por año | En los meses comunes (1–8) el count medio pasa de 136.4 en 2011 a 224.2 en 2012 (+64.4 %). En 2011 (año completo): mes de mayor demanda media = 7 (204); de menor = 1 (55). | El nivel de demanda cambia entre años y el test es posterior al train: el modelo debe extrapolar. `year` captura ese nivel pero no sirve para años nuevos (limitación para el análisis crítico). |
| 4 | Clima continuo y redundancia temp–atemp | corr(temp, count) = 0.43; corr(humidity, count) = -0.33; corr(windspeed, count) = 0.10; corr(temp, atemp) = 0.99. | temp y atemp son casi la misma señal (multicolinealidad): puede inestabilizar LinearRegression; decidir si se conserva atemp y justificarlo en T1. Las correlaciones lineales con count son modestas; la relación real depende de la hora. |
| 5 | Efecto de la condición meteorológica y tamaño de cada categoría | clima 1: media 188.2 (n=5858); clima 2: media 160.3 (n=2142); clima 3: media 102.6 (n=707); clima 4: media 164.0 (n=1). Categorías con < 2 % de los datos: [4]. | Entre las categorías con muestra suficiente, la demanda media del clima 3 es menor que la del clima 1. La(s) categoría(s) [4] tiene(n) tan pocos casos que su media no es interpretable y el OneHotEncoder debe usar handle_unknown='ignore'; en el análisis de MAE por clima hay que indicar n por categoría. |

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

| Conjunto | Registros | Proporción | Desde | Hasta | Método |
|---|---|---|---|---|---|
| Entrenamiento | 8708 | 80.0 % | 2011-01-01 00:00 | 2012-08-05 04:00 | Primeras filas en orden cronológico |
| Prueba (test) | 2178 | 20.0 % | 2012-08-05 05:00 | 2012-12-19 23:00 | Últimas filas en orden cronológico |
| Total | 10886 | 100 % | 2011-01-01 00:00 | 2012-12-19 23:00 | — |

Semilla: no aplica (corte determinista); `RANDOM_STATE = 42` declarado.

**Comprobaciones superadas:** sin_indices_compartidos, sin_fechas_compartidas, train_anterior_a_test, suma_registros_igual_total, train_ordenado, test_ordenado.
Dimensiones tras separar X e y: `X_train` (8708, 12), `X_test` (2178, 12), `y_train` (8708,),
`y_test` (2178,).

### 6.2 Justificación del esquema

Los datos son una serie temporal horaria: el consumo de una hora se parece al de las horas y días vecinos. Con una
división aleatoria, el modelo se entrenaría con horas posteriores a las que luego debe predecir y el resultado en
test quedaría inflado. Con el corte cronológico, el test representa el uso real: predecir un período que viene
**después** de todo lo que el modelo conoce. El test se reserva **antes** de ajustar cualquier transformación y se
abrirá una sola vez, en la etapa 13, con el modelo ya elegido.

### 6.3 Qué implica este corte

- El test cubre 2012-08-05 a 2012-12-19
  (meses [8, 9, 10, 11, 12] de 2012). Los meses
  [9, 10, 11, 12] solo aparecen en el entrenamiento de 2011:
  el modelo debe trasladar a 2012 lo aprendido de esa época de 2011, y por eso `year` pesa en el resultado.
- El test no incluye clima 4 (solo clima 1: 1334, clima 2: 692, clima 3: 152 filas) y el
  entrenamiento tiene una sola fila de clima 4.
- Por la tendencia creciente, se espera que los errores absolutos (MAE) del test sean mayores que los de
  validación en los primeros folds.

### 6.4 Los cinco folds de `TimeSeriesSplit(n_splits=5)` sobre el entrenamiento

| Fold | n entrenamiento | n validación | count medio validación | Train desde | Train hasta | Validación desde | Validación hasta |
|---|---|---|---|---|---|---|---|
| Fold 1 | 1453 | 1451 | 172.8 | 2011-01-01 | 2011-04-06 | 2011-04-06 | 2011-07-09 |
| Fold 2 | 2904 | 1451 | 181.1 | 2011-01-01 | 2011-07-09 | 2011-07-09 | 2011-10-13 |
| Fold 3 | 4355 | 1451 | 142.5 | 2011-01-01 | 2011-10-13 | 2011-10-13 | 2012-01-17 |
| Fold 4 | 5806 | 1451 | 200.6 | 2011-01-01 | 2012-01-17 | 2012-01-17 | 2012-05-01 |
| Fold 5 | 7257 | 1451 | 274.8 | 2011-01-01 | 2012-05-01 | 2012-05-01 | 2012-08-05 |

En cada fold el entrenamiento termina antes de que empiece la validación (verificado con un `assert`), y los
tres modelos usarán exactamente estos mismos folds. *Comentario de estabilidad esperada:* el fold 1 entrena con solo
1,453 filas (enero–abril de 2011) y debe predecir abril–julio, una época que no vio; además el
`count` medio de las ventanas de validación varía entre 142 y
275. Por tanto no debe sorprender que el MAE cambie mucho entre folds:
refleja diferencias de nivel y estación, no necesariamente un modelo inestable.

---

## Decisiones adoptadas en la Fase 2 (revisarlas en grupo)

| Tema | Decisión | Justificación |
|---|---|---|
| EDA | Gráficos y correlaciones solo con train | No mirar el test |
| Horas pico | [7, 8, 17, 18, 19] | Regla: demanda media laborable > 1.5 × media global (train) |
| `windspeed = 0` | Conservar y declarar limitación | No hay valores entre 0 y 6.00; imputar con todo el CSV sería fuga |
| `humidity = 0` | Conservar y declarar limitación | 22 filas de un solo día (2011-03-10) |
| `temp` y `atemp` | Conservar ambas | Redundancia declarada (corr 0.99); probar quitar `atemp` en Fase 3 |
| `season` y `month` | Conservar ambas | Redundancia declarada (corr 0.97); `month` aporta más resolución |
| Clima 4 | Conservar; `handle_unknown="ignore"` | 1 fila en train, 0 en test |
| Partición | 80/20 cronológico + TimeSeriesSplit(5) | Obligatorio para el grupo |

## Pendiente del grupo antes de la Fase 3
- [ ] **Acordar el criterio de éxito** de la etapa 2 (propuesta: MAE de test ≥ 30 % menor que el DummyRegressor y
      MAE en horas pico ≤ 2 × MAE global). Debe quedar fijado antes de entrenar.
- [ ] Completar la ficha del dataset (fecha de consulta, licencia/reglas de Kaggle).
- [ ] Revisar y reescribir con sus palabras las interpretaciones de este documento.
- [ ] Guardar las capturas de la app (pantallas 5 y 6) y del notebook `02_fase2.ipynb`.
