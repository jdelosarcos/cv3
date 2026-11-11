# Crime Occurrence Intelligence — Streamlit

Converted and expanded from `Crime_Occurrence_Analysis_Dashboard (1).ipynb`.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload the original Excel crime datasets using the sidebar. The supplied notebook did **not** contain the four underlying Excel files, so actual model scores, maps, and predictions cannot be verified until data is uploaded.

## Modules

- `modules/data.py`: Excel sheets, column normalization, quality flags, derived fields; cached
- `modules/charts.py`: trends, time-of-day and weekday patterns, geographic offense comparisons
- `modules/maps.py`: geocoded marker and density maps with public basemap tiles
- `modules/mining.py`: offense-area frequency patterns, K-means, silhouette and Davies–Bouldin evaluation
- `modules/forecast.py`: monthly lag panel, seven prediction methods, holdout and forward-chaining CV
- `app.py`: themed navigation, filters, KPIs and export

## Model interpretation

The final 20% of calendar months are held out for chronological evaluation. Cross-validation folds use expanding training periods drawn only from earlier (pre-holdout) months. The historical last-month count is the naive baseline. A PASS requires at least 10% improvement in holdout MAE and nonnegative holdout R². No passing result is guaranteed. Monthly zero filling assumes missing reports mean zero incidents; confirm this assumption against data collection procedures.

## Mapping

Map plots use actual valid coordinates from uploaded workbooks and Plotly `scatter_map`/`density_map`, with public OpenStreetMap basemap tiles. Internet access is required for tiles. No invented or geocoded coordinates. If no valid coordinates exist, maps show a clear warning rather than failing.

## Deployment and ethics

Keep personal data out of public apps. Remove additional free-text identifying columns as needed, use authentication and restricted access for operational deployments, and validate reporting completeness and exposure-adjusted rates before using area rankings in resource allocation.
