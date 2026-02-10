## ES — Resumen del análisis (Tableau + SQL)

Este proyecto combina visualización en **Tableau** y consultas en **SQL (SQLite)** para analizar el mercado inmobiliario de Nueva York desde tres perspectivas clave: **distribución del precio**, **actividad por borough** y **segmentación entre mercados de volumen y de valor**.

### Visualización (Tableau)
- **Building Age vs Sale Price (scatter):** analiza la relación entre la antigüedad del edificio y el precio de venta, así como la dispersión y posibles outliers.
- **Distribution Borough (map/heatmap):** muestra la concentración geográfica de las transacciones y precios por zona.
- Incluye filtro por **Borough** para comparar patrones entre ubicaciones.

### Preguntas de negocio respondidas con SQL

1. **¿Dónde se concentra la liquidez del mercado inmobiliario?**  
   Se analiza el número de transacciones por borough para identificar las zonas con mayor actividad.

2. **¿Qué boroughs presentan los precios medios más altos?**  
   Se comparan los precios medios de venta para detectar mercados de alto valor.

3. **¿Existen mercados de alto precio y bajo volumen?**  
   Se evalúa conjuntamente el volumen de transacciones y el precio medio para identificar mercados exclusivos.

## EN — Analysis Summary (Tableau + SQL)

This project combines **Tableau visualizations** and **SQL (SQLite) queries** to analyze the New York City housing market from three key perspectives: **price distribution**, **activity by borough**, and **segmentation between volume-driven and value-driven markets**.

### Visualization (Tableau)
- **Building Age vs Sale Price (scatter):** analyzes the relationship between building age and sale price, including dispersion and potential outliers.
- **Distribution Borough (map/heatmap):** shows the geographic concentration of transactions and prices by area.
- Includes a **Borough** filter to compare patterns across locations.

### Business questions answered with SQL

1. **Where is housing market liquidity concentrated?**  
   The number of transactions by borough is analyzed to identify the most active areas.

2. **Which boroughs show the highest average prices?**  
   Average sale prices are compared to identify high-value markets.

3. **Are there high-price, low-volume markets?**  
   Transaction volume and average price are analyzed together to detect exclusive market segments.
