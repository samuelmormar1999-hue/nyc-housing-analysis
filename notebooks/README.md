# EDA NYC Housing – Notebook process (`eda_nyc_housing.ipynb`)

## ES
**Qué hace:** análisis exploratorio del dataset de viviendas en NYC y validación rápida con SQL (SQLite).

**Pasos:**
1) Carga del CSV y revisión inicial (shape, columnas, tipos, describe).  
2) Limpieza básica (renombrado de columnas, eliminación de columnas sobrantes, nulos/duplicados).  
3) EDA de precios (distribución y revisión de outliers con gráficas).  
4) Comparación por borough (conteos y métricas como mediana de precio y precio/m²).  
5) Export a SQLite y consultas SQL simples para validar resultados.

**Outputs:** figuras en `../reports/figures/` y base de datos en `../sql/`.

---

## EN
**What it does:** exploratory analysis of NYC housing data + quick SQL (SQLite) validation.

**Steps:**
1) Load CSV and run initial checks (shape, columns, dtypes, describe).  
2) Basic cleaning (rename/drop columns, missing values/duplicates).  
3) Price EDA (distribution + outlier checks with charts).  
4) Borough comparison (counts + median price and price per sqft).  
5) Export to SQLite and run simple SQL queries for validation.

**Outputs:** charts in `../reports/figures/` and a DB in `../sql/`.
