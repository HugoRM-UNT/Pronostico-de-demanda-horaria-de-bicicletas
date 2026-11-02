# Etapa 4 · Separación de X e y

Código completo y ejecutable en `src/fase1_xy.py` y `notebooks/01_fase1.ipynb`.

```python
y = df["count"]
X = df.drop(columns=["count", "casual", "registered", "datetime"])
assert X.index.equals(y.index)
assert not {"casual", "registered", "count"} & set(X.columns)
```

## Tabla de exclusiones
| Columna | Razón |
|---|---|
| `count` | Es el objetivo |
| `casual`, `registered` | Suman `count`: fuga directa |
| `datetime` | Se reemplaza por `hour`, `month`, `dayofweek`, `year`; el modelo no debe memorizar timestamps |

## Evidencia que debe ir en el informe
1. Captura de la celda con X e y.
2. Captura de las dimensiones (`X.shape`, `y.shape`).
3. Captura del resultado de los `assert` (sin fuga, índices coinciden).
4. Esta tabla de exclusiones con su interpretación.
