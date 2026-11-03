# Informe de modelado y evaluación · Grupo 04

## Metodología

Se reservó el 20 % final del archivo como test temporal. La selección se hizo únicamente
con el 80 % inicial mediante `TimeSeriesSplit(n_splits=5)`. Las imputaciones, el escalado
y la codificación se ajustaron dentro de cada `Pipeline`. `casual` y `registered` fueron
excluidas por fuga directa. Se añadieron variables cíclicas, `days_since_start` y la
interacción categórica `workingday_hour`.

## Selección previa al test

Se compararon LinearRegression, KNN y SVR. KNN y SVR usaron `GridSearchCV`, objetivo
transformado con `log1p` y rejillas ampliadas. El ganador fue **SVR**, con MAE CV óptimo
45.01. El desempate declarado es: si al menos dos candidatos quedan a ≤1 % del mejor MAE,
se elige la menor desviación y después la prioridad LinearRegression, SVR, KNN.
El DummyRegressor se conserva como referencia y no compite. HistGradientBoosting queda
como mejora futura y no participa en la selección exigida.

## Evaluación final

En el test reservado, SVR obtuvo MAE = **35.49**, RMSE = **55.46** y R² = **0.935**.
El MAE en horas pico (7–8 y 17–19) fue **59.82**, equivalente a **1.69×** el MAE global;
por tanto, cumple el criterio de no superar 2×.

El resultado corresponde al tramo final de este CSV y no garantiza producción: el conjunto
tiene días ausentes, el clima usado es observado y el tramo final exige extrapolar el nivel
interanual. Las tablas `T8_mae_por_hora.csv` y `T8_mae_por_clima.csv` detallan los errores.
