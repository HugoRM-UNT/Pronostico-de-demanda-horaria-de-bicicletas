# Grupo 04 · Pronóstico de demanda horaria de bicicletas
Sistemas Inteligentes y Machine Learning · Trabajo aplicativo (regresión)

**Contenido de este paquete: FASE 1 (etapas 1–4), FASE 2 (etapas 5–6) y FASE 3–5
(modelado, evaluación y predicciones).**

## Estructura
```
Grupo04_Streamlit/
├── app.py                      # App Streamlit (etapas 1–16)
├── requirements.txt
├── data/                       # Aquí va train.csv (ver data/LEEME_DATOS.md)
├── docs/
│   ├── etapa1_2_problema_objetivos.md
│   ├── etapa3_dataset_diccionario.md
│   ├── etapa4_X_y.md
│   ├── informe_fase2_etapas5_6.md   # texto del informe (etapas 5-6) con cifras reales
│   ├── figuras/                 # 8 gráficos en PNG
│   ├── tablas/                  # T2, T3, folds, describe, resumen, hallazgos (CSV)
│   └── T1_diccionario_variables.csv
├── src/fase1_xy.py             # Carga, variables de calendario, X e y, verificaciones
├── src/fase2_eda.py            # Calidad (T2), corte temporal (T3), folds, gráficos y hallazgos
├── src/fase3_modelado.py       # Pipelines, CV temporal, GridSearch y evaluación
├── src/generar_evidencias_fase3.py # Regenera T4-T9, figuras e informe de modelado
├── src/generar_evidencias_fase2.py  # Regenera figuras, tablas e informe de la fase 2
├── notebooks/01_fase1.ipynb    # Evidencias de la fase 1 con salidas
└── notebooks/02_fase2.ipynb    # Evidencias de la fase 2 con salidas
```

## Cómo ejecutar
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
# 1) Descargar train.csv de Kaggle y colocarlo en data/   (ver data/LEEME_DATOS.md)
python src/fase1_xy.py            # imprime dimensiones y comprobaciones
python src/fase2_eda.py           # imprime T2, T3, folds y hallazgos
python src/generar_evidencias_fase2.py   # regenera docs/figuras, docs/tablas e informe
python src/generar_evidencias_fase3.py   # genera T4-T9 y el informe de modelado
streamlit run app.py
```

## Estado de la Fase 2 (completa)
Hecho con el `train.csv` real (10 886 × 12): T2, T3, 7 gráficos interpretados, 5 hallazgos, 5 folds de
`TimeSeriesSplit`, pantallas 5 y 6 de la app y notebook `02_fase2.ipynb` ejecutado (con salidas).
El texto para el informe está en `docs/informe_fase2_etapas5_6.md`. La app también incluye las
etapas 7–16; estas deben ejecutarse después de colocar el CSV correcto.

## Pendiente del grupo antes de la Fase 3
- [ ] Acordar el criterio de éxito (etapa 2) ANTES de entrenar y escribirlo en el informe.
- [ ] Completar nombres de integrantes.
- [ ] Completar la ficha del dataset: fecha de consulta y licencia/reglas de Kaggle.
- [ ] Revisar y reescribir con sus palabras las interpretaciones; confirmar las decisiones del informe.
- [ ] Tomar las capturas de la app (pantallas 5 y 6) con `streamlit run app.py`.
- [ ] Ejecutar las etapas 7–16 y exportar T4–T9 con el CSV Bike Sharing correcto.

## Reglas metodológicas (no romper)
- Prohibido usar `casual` y `registered` (suman `count`).
- Holdout temporal del tramo final + `TimeSeriesSplit(n_splits=5)`; nada de mezcla aleatoria.
- Preprocesamiento solo dentro de `Pipeline`; el test se usa una sola vez.
- Los resultados deben salir del CSV descargado; no inventar cifras.
- El archivo debe ser el `train.csv` de Bike Sharing Demand y contener `datetime`, `casual`,
  `registered` y `count`. El archivo actualmente ubicado en la raíz debe verificarse antes de usarlo.
