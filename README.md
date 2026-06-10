# Canada Trade KPI Lab · FCBB

Interactive dashboard for Canadian International Merchandise Trade Statistics, sourced directly from **Statistics Canada**.

## Data Source

| Table | Description |
|-------|-------------|
| 12-10-0011-01 | Merchandise **imports** by product section/chapter and trade partner |
| 12-10-0012-01 | Merchandise **exports** by product section/chapter and trade partner |

Values are in **Canadian dollars**. The SCALAR_FACTOR column is read automatically (typically "Thousands") and applied. Since the data is downloaded directly from Statistics Canada, totals match official figures exactly.

## Pipeline

```
Statistics Canada bulk download (ZIP → CSV)
         │
         ▼
src/extract_statcan.py   →  data/canada_trade.parquet
         │
         ▼
src/upload_hf.py         →  HuggingFace: WilgnerCH/canada-trade-data
         │
         ▼
src/build_dashboard_data.py  →  data/*.json
         │
         ▼
index.html  (static dashboard, loads JSON files)
```

Runs automatically on the **10th of every month** via GitHub Actions.

## Scripts

| Script | Purpose |
|--------|---------|
| `src/extract_statcan.py` | Download trade tables from Statistics Canada, tag each row with `row_type`, save parquet |
| `src/upload_hf.py` | Upload parquet to Hugging Face Hub (requires `HF_TOKEN`) |
| `src/build_dashboard_data.py` | Load parquet, build aggregated JSON files for the dashboard |

## row_type — preventing double-counting

| row_type | partner | commodity | Used for |
|----------|---------|-----------|----------|
| `grand_total` | Total, all … | Total, all … | Monthly totals |
| `country_total` | Specific country | Total, all … | Country rankings |
| `commodity_total` | Total, all … | Specific chapter | Commodity rankings |
| `detail` | Specific country | Specific chapter | Cross-tab |

Each aggregation in `build_dashboard_data.py` uses only the appropriate `row_type`, so values are never summed twice.

## Dashboard Features

- **Overview**: KPI cards (imports, exports, balance, YoY growth) + monthly trend chart
- **Countries**: World map choropleth + top 15 bar chart + full sortable table
- **Commodities**: Top 15 bar chart + full table with chapter names from Statistics Canada
- **Year filter**: Filter all views to a specific calendar year

## Local Setup

```bash
pip install -r requirements.txt

# Full pipeline
python src/extract_statcan.py           # downloads ~200 MB from StatCan
HF_TOKEN=your_token python src/upload_hf.py
python src/build_dashboard_data.py

# Build dashboard from existing HuggingFace parquet (no download)
python src/build_dashboard_data.py --hf
```

## GitHub Secrets

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Hugging Face token with write access to `WilgnerCH/canada-trade-data` |
