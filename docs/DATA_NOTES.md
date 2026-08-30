# SentinelFin — Data Notes & Exploratory Analysis (Phase 1)

## Data Pipeline Summary

- **Source Dataset**: CFPB Consumer Complaint Database (`complaints.csv`, ~9.18 GB).
- **Total Raw Complaint Records**: 8,660,851 complaints.
- **Narrative Presence**: 1,924,614 complaints (22.22% overall narrative-availability rate).
- **Quality Filtering**: Complaints with narratives under 10 words dropped (66,535 records dropped).
- **Final Processed Panel**: 1,858,079 clean narrative complaints spanning **137 monthly windows** (`window_id` from `2015-03` to `2026-08`).
- **Median Narrative Length**: ~112 words per complaint.
- **Persisted Format**: `/data/processed/panel.parquet`.

## Key Insights & Quality Considerations

1. **Narrative Availability over Time**:
   - The CFPB began publishing optional consumer complaint narratives in March 2015 (`2015-03`).
   - Prior to 2015-03, zero narratives exist in the database.
   - Post-2015, the narrative presence rate stabilized between 20%–25% of total complaints received.

2. **Product Distribution**:
   - Top complaint products: Credit Reporting / Repair Services, Debt Collection, Credit Cards, Checking / Savings Accounts, Mortgages.
   - Emerging sub-issues and products (e.g. Buy Now Pay Later, Earned Wage Access) begin appearing in narratives prior to official product category re-classification.

3. **Windowing Strategy**:
   - Monthly time-windows (`window_id` format `YYYY-MM`) provide sufficient temporal resolution to track cluster birth, growth, and centroid trajectory drift while keeping computation tractable.

## Generated Exploratory Data Analysis (EDA) Artifacts

- `outputs/eda_volume_over_time.png`: Monthly complaint volume over time.
- `outputs/eda_narrative_length.png`: Narrative word length distribution.
- `outputs/eda_top_products.png`: Top 15 product breakdown.
- `outputs/volume_by_window.csv`: Detailed monthly count breakdown.
- `outputs/top_products.csv`: Product frequency counts.
