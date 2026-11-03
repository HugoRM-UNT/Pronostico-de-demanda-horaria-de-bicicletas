"""Genera T4-T9 y el informe de modelado a partir del train.csv real."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fase1_xy import cargar_datos
from fase3_modelado import cinco_predicciones, entrenar_experimento, evaluar

RAIZ = Path(__file__).resolve().parent.parent
TAB = RAIZ / "docs" / "tablas"
FIG = RAIZ / "docs" / "figuras"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def main():
    df = cargar_datos()
    exp = entrenar_experimento(df)
    ev = evaluar(exp)
    predicciones = cinco_predicciones(exp, ev)

    pd.DataFrame({
        "Paso": ["Preprocesamiento", "Validación", "Selección"],
        "Descripción": [
            "Imputación, escalado y OneHotEncoder dentro de cada Pipeline",
            "TimeSeriesSplit(n_splits=5), sin barajar",
            f"Menor MAE CV: {exp['winner']}",
        ],
    }).to_csv(TAB / "T4_pipelines.csv", index=False, encoding="utf-8-sig")
    exp["cv"].to_csv(TAB / "T5_validacion.csv", index=False, encoding="utf-8-sig")
    exp["grids"].to_csv(TAB / "T6_gridsearch.csv", index=False, encoding="utf-8-sig")
    for nombre, top in exp["grid_top"].items():
        top.to_csv(TAB / f"T6_top5_{nombre}.csv", index=False, encoding="utf-8-sig")
    exp["selection"].to_csv(TAB / "T7_seleccion.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([ev["metrics"]]).assign(
        Modelo=exp["winner"],
        MAE_horas_pico=ev["peak_mae"],
        ratio_pico_global=ev["peak_mae"] / ev["metrics"]["MAE"],
    ).to_csv(
        TAB / "T8_metricas_test.csv", index=False, encoding="utf-8-sig"
    )
    ev["by_hour"].to_csv(TAB / "T8_mae_por_hora.csv", index=False, encoding="utf-8-sig")
    ev["by_weather"].to_csv(TAB / "T8_mae_por_clima.csv", index=False, encoding="utf-8-sig")
    ev["detail"][["datetime", "count", "prediccion", "residual"]].head(10).to_csv(
        TAB / "T8_real_vs_predicho_10.csv", index=False, encoding="utf-8-sig"
    )
    predicciones.to_csv(TAB / "T9_predicciones.csv", index=False, encoding="utf-8-sig")

    detail = ev["detail"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(detail["count"], detail["prediccion"], s=8, alpha=.3)
    lim = [0, max(detail["count"].max(), detail["prediccion"].max())]
    axes[0].plot(lim, lim, "r--")
    axes[0].set(xlabel="Real", ylabel="Predicho", title=f"Real vs. predicho ({exp['winner']})")
    axes[1].hist(detail["residual"], bins=35, color="#4c72b0", edgecolor="white")
    axes[1].axvline(0, color="crimson", ls="--")
    axes[1].set(xlabel="Residual (real - predicho)", title="Distribución de residuales")
    fig.tight_layout()
    fig.savefig(FIG / "09_evaluacion_residuales.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = ev["metrics"]
    report = f"""# Informe de modelado y evaluación · Grupo 04

## Metodología

Se reservó el 20 % final del archivo como test temporal. La selección se hizo únicamente
con el 80 % inicial mediante `TimeSeriesSplit(n_splits=5)`. Las imputaciones, el escalado
y la codificación se ajustaron dentro de cada `Pipeline`. `casual` y `registered` fueron
excluidas por fuga directa. Se añadieron variables cíclicas, `days_since_start` y la
interacción categórica `workingday_hour`.

## Selección previa al test

El modelo elegido fue **{exp['winner']}**, por obtener el menor MAE de validación entre
LinearRegression, KNN y SVR ajustado. La selección prioriza MAE porque cada unidad representa
una bicicleta de error y porque la distribución combina horas de baja demanda con horas pico.
Se compararon los mismos cinco folds cronológicos para todos los modelos, de modo que cada
candidato enfrentara exactamente la misma dificultad temporal. La desviación del MAE complementa
el promedio: un valor bajo con variación alta sería menos confiable cuando cambia el periodo
observado. KNN y SVR se ajustaron con `GridSearchCV`; sus rejillas incluyen valores interiores
y de borde (`n_neighbors` = 3, 5, 10, 15, 30; `C` = 10, 50, 100, 300 y varios `gamma`).
KNN y SVR transforman el objetivo con `log1p` y vuelven a la escala original para reducir el
efecto de la asimetría de `count`. LinearRegression se conservó como referencia interpretable y
DummyRegressor como línea base no competitiva. El ganador se determina por menor MAE CV.
Solo si al menos dos candidatos quedan a ≤1 % del mejor MAE se declara empate práctico y se
desempata por menor desviación; después se prioriza LinearRegression, SVR y KNN. El test
permanece completamente reservado hasta la evaluación final.

## Evaluación final

En el test reservado, el modelo obtuvo MAE = **{metrics['MAE']:.2f}**, RMSE = **{metrics['RMSE']:.2f}**
y R² = **{metrics['R²']:.3f}**. El MAE es la métrica principal porque expresa directamente
el error promedio en bicicletas. El MAE en horas pico (7–8 y 17–19) fue **{ev['peak_mae']:.2f}**,
es decir, **{ev['peak_mae'] / metrics['MAE']:.2f}×** el MAE global; el criterio de éxito
debe considerarse {'cumplido' if ev['peak_mae'] <= 2 * metrics['MAE'] else 'no cumplido'}.

El criterio acordado exige que el MAE en horas pico no supere dos veces el MAE global. El error
cambia por hora y condición meteorológica; por ello se entregan las tablas
`T8_mae_por_hora.csv` y `T8_mae_por_clima.csv`. El resultado no garantiza producción:
el conjunto tiene días ausentes, el clima usado es observado y el tramo final exige
extrapolar el nivel interanual. HistGradientBoosting queda como mejora futura y no participa
en la selección exigida por la guía.
"""
    (RAIZ / "docs" / "informe_fase3_modelado.md").write_text(report, encoding="utf-8")
    print(f"Ganador: {exp['winner']}")
    print(metrics)
    print("Evidencias generadas en docs/tablas, docs/figuras y docs/informe_fase3_modelado.md")


if __name__ == "__main__":
    main()
