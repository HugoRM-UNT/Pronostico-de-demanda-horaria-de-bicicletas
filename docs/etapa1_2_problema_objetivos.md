# Etapas 1 y 2 · Planteamiento, objetivos y criterio de éxito

## Etapa 1. Planteamiento del problema

**Contexto.** Los sistemas de bicicletas compartidas deben tener bicicletas disponibles en cada estación y hora. Si hay pocas, se pierden viajes; si hay demasiadas, se desperdicia logística. Este trabajo usa el histórico horario de Capital Bikeshare (Washington D.C.) para pronosticar cuántas bicicletas se alquilarán en horas futuras.

**Pregunta predictiva.**
> ¿Cuántas bicicletas se alquilarán (`count`) en una hora futura, dados el momento del día, el calendario y las condiciones meteorológicas esperadas?

**Quién, cuándo y para qué.**
- Usuario: el equipo de operaciones del sistema.
- Momento: el día anterior o unas horas antes del turno.
- Decisión: cuántas bicicletas redistribuir y cuándo programar mantenimiento.

**Enfoque supervisado (regresión).** Existe un histórico con la respuesta real (`count`) para cada hora, y la salida es numérica continua.

**Esquema de entradas y salida.**
- Entradas: `hour`, `month`, `dayofweek`, `year`, `season`, `holiday`, `workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed`.
- Salida: `count` (entero no negativo).

**Beneficios esperados.** Mejor redistribución de bicicletas, menos usuarios sin servicio en horas pico y mantenimiento planificado en horas de baja demanda.

## Etapa 2. Objetivos y criterio de éxito

**Objetivo general.** Desarrollar y evaluar un sistema predictivo de la demanda horaria de bicicletas con tres modelos de regresión (LinearRegression, KNeighborsRegressor y SVR), usando validación temporal sin fuga de información y un test final reservado.

**Objetivos específicos.**
1. *Análisis de datos:* identificar los predictores disponibles al momento de predecir y los riesgos de calidad del dataset.
2. *Entrenamiento:* construir pipelines reproducibles (`ColumnTransformer` + estimador) para los tres algoritmos.
3. *Validación:* comparar los modelos con `TimeSeriesSplit(n_splits=5)` y optimizar KNN y SVR con `GridSearchCV`.
4. *Evaluación:* medir el modelo elegido una sola vez en el tramo final con MAE, RMSE y R², y analizar el error por hora del día y por condición meteorológica.

**Criterio de éxito (PROPUESTA: acordar en grupo antes de experimentar).**
> El modelo se considera útil si su MAE en test es al menos 30 % menor que el del DummyRegressor (media del entrenamiento) y el MAE en horas pico no supera el doble del MAE global.

**Costo del error.** Subestimar la demanda en hora pico deja usuarios sin bicicleta (error más grave). Sobrestimar implica mover bicicletas de más (error menos grave).

**Tabla objetivo–actividad–evidencia.**

| Objetivo | Actividad | Evidencia |
|---|---|---|
| Específico 1 | EDA y diccionario de variables | T1, T2, mínimo 5 gráficos |
| Específico 2 | Pipelines con ColumnTransformer | T4, código |
| Específico 3 | CV temporal y GridSearchCV | T5, T6 |
| Específico 4 | Evaluación única en test | T8, residuales, MAE por hora |
