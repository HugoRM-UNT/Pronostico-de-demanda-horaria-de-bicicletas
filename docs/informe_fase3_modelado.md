# Informe de modelado y evaluación · Grupo 04

## Metodología

Se reservó el 20 % final del archivo como test temporal. La selección se hizo únicamente
con el 80 % inicial mediante `TimeSeriesSplit(n_splits=5)`. Las imputaciones, el escalado
y la codificación se ajustaron dentro de cada `Pipeline`. `casual` y `registered` fueron
excluidas por fuga directa.

## Selección previa al test

El modelo elegido fue **LinearRegression**, por obtener el menor MAE de validación entre
LinearRegression, KNN y SVR ajustado. La selección prioriza MAE porque cada unidad representa
una bicicleta de error y porque la distribución combina horas de baja demanda con horas pico.
Se compararon los mismos cinco folds cronológicos para todos los modelos, de modo que cada
candidato enfrentara exactamente la misma dificultad temporal. La desviación del MAE complementa
el promedio: un valor bajo con variación alta sería menos confiable cuando cambia el periodo
observado. KNN y SVR se ajustaron con rejillas pequeñas y razonables, ya que ampliar muchas
combinaciones aumentaría el costo sin garantía de mejorar la generalización. LinearRegression se
conservó como referencia interpretable y DummyRegressor como línea base no competitiva. El ganador
se determina con el MAE de validación, no con el resultado que eventualmente produzca el test.
Por eso el test permanece completamente reservado hasta la evaluación final y no influye en la
elección ni en los hiperparámetros.

## Evaluación final

En el test reservado, el modelo obtuvo MAE = **98.38**, RMSE = **131.27**
y R² = **0.636**. El MAE es la métrica principal porque expresa directamente
el error promedio en bicicletas.

El error cambia por hora y condición meteorológica; por ello se entregan las tablas
`T8_mae_por_hora.csv` y `T8_mae_por_clima.csv`. El resultado no garantiza producción:
el conjunto tiene días ausentes, el clima usado es observado y el tramo final exige
extrapolar el nivel interanual.
