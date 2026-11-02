# Etapa 3 · Ficha del dataset y diccionario de variables

## Ficha del dataset
| Campo | Contenido |
|---|---|
| Nombre | Bike Sharing Demand |
| Fuente | Kaggle: https://www.kaggle.com/competitions/bike-sharing-demand/data |
| Archivo usado | `train.csv` (contiene `count`; `test.csv` no la incluye) |
| Origen de los datos | Capital Bikeshare (Washington D.C.); basado en Fanaee-T y Gama (2013), repositorio UCI |
| Licencia / condiciones | Reglas de la competencia de Kaggle. **LEERLAS Y CITARLAS AQUÍ:** ____________ |
| Filas × columnas | **10 886 × 12** (verificado con `df.shape` sobre el `train.csv` descargado) |
| Rango temporal | **2011-01-01 00:00 a 2012-12-19 23:00** (verificado; solo días 1–19 de cada mes) |
| Fecha de consulta | ____________ |
| Versión / nombre exacto del CSV | ____________ |

## T1 · Diccionario de variables
Ver también `T1_diccionario_variables.csv`.

| Variable | Definición | Tipo | Unidad | Disponible antes | Decisión |
|---|---|---|---|---|---|
| `datetime` | Fecha y hora | datetime | hora | Sí | Se descompone en `hour`, `month`, `dayofweek`, `year`; no se usa cruda |
| `season` | Estación (1 a 4) | categórica | – | Sí | Incluir |
| `holiday` | Día feriado | binaria | 0/1 | Sí | Incluir |
| `workingday` | Día laborable | binaria | 0/1 | Sí | Incluir |
| `weather` | Clima (1 despejado … 4 severo) | categórica | – | Solo como pronóstico | Incluir |
| `temp` | Temperatura | numérica | °C | Solo como pronóstico | Incluir |
| `atemp` | Sensación térmica | numérica | °C | Solo como pronóstico | Incluir (revisar redundancia con `temp`) |
| `humidity` | Humedad relativa | numérica | % | Solo como pronóstico | Incluir |
| `windspeed` | Velocidad del viento | numérica | – | Solo como pronóstico | Incluir |
| `casual` | Alquileres de usuarios casuales | numérica | bicis | No | **Excluir (fuga)** |
| `registered` | Alquileres de usuarios registrados | numérica | bicis | No | **Excluir (fuga)** |
| `count` | Total de alquileres | numérica | bicis | – | **Objetivo (y)** |

**Riesgo a declarar:** en uso real, las variables de clima serían *pronósticos* y no valores observados; el error real sería mayor que el medido con clima observado. Incluirlo en las limitaciones (etapa 14).
