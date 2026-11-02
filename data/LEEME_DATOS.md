# Datos
1. Entrar a https://www.kaggle.com/competitions/bike-sharing-demand/data (requiere cuenta de Kaggle y aceptar las reglas de la competencia).
2. Descargar `train.csv` (es el que contiene `count`; `test.csv` NO tiene la variable objetivo).
3. Colocarlo aquí como `data/train.csv`.

Por la licencia/reglas de la competencia, no se incluye el CSV en este paquete.
Alternativa con la CLI de Kaggle:
```bash
pip install kaggle
kaggle competitions download -c bike-sharing-demand -f train.csv -p data/
```
Anotar en la ficha: URL, nombre exacto del archivo, fecha de consulta, filas × columnas reales.
