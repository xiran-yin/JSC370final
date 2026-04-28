# JSC370 Final Project — Weather Impact on TTC Subway Delays

This repository contains the full final project deliverables:

- Quarto website
- 6-7 page final report (HTML + PDF)
- 3 interactive visualizations (HW5 requirement)
- 5-minute presentation slides
- Reproducible data and modeling pipeline

## Website

Replace the link below after GitHub Pages deployment:

- https://your-username.github.io/JSC370-finalproject

## Data Sources

- Open-Meteo API: https://archive-api.open-meteo.com/v1/archive
- Toronto Open Data CKAN API package:
  https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show?id=ttc-subway-delay-data

## Reproducible Build

```bash
python3 scripts/final_pipeline.py
quarto render
```

Or run the all-in-one presentation/report pipeline:

```bash
./scripts/presentation_pipeline.sh
```

## Repository Structure

- `scripts/final_pipeline.py` — data acquisition, cleaning, feature engineering, modeling, and artifact export
- `report.qmd` — final written report
- `viz.qmd` — interactive figures page
- `slides.qmd` — presentation deck
- `data/` — processed analysis datasets
- `outputs/` — tables, plots, and interactive HTML visualizations
- `presentation/` — speaker timing guide

## Notes

No local absolute paths are required. All files are built from project-relative paths for reproducibility.
