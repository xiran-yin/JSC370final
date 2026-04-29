# JSC370 Final Project — Weather Impact on TTC Subway Delays

This project investigates how weather conditions are associated with day-level TTC subway delays in Toronto (2014-2026), using reproducible data acquisition, modeling, and reporting.

Scope note: the final version excludes the NLP/reddit sentiment component and focuses on weather-driven TTC delay modeling, aligned with midterm rubric feedback.

## Project Links

- GitHub repository: https://github.com/xiran-yin/JSC370final
- GitHub Pages website: https://xiran-yin.github.io/JSC370final/
- Final report (PDF): https://xiran-yin.github.io/JSC370final/report.pdf
- Interactive visualizations: https://xiran-yin.github.io/JSC370final/viz.html
- 5-minute presentation video (YouTube): https://youtu.be/sfOBl8uQWwI

## Data Sources

- Open-Meteo archive API: https://archive-api.open-meteo.com/v1/archive
- Toronto Open Data CKAN API package (`ttc-subway-delay-data`):  
  https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show?id=ttc-subway-delay-data

## Reproducibility

Run the pipeline and render the site/report from project root:

```bash
python3 scripts/final_pipeline.py
quarto render
```

Optional all-in-one script:

```bash
./scripts/presentation_pipeline.sh
```

## Repository Structure

- `scripts/final_pipeline.py` — API pulls, cleaning, feature engineering, modeling, export
- `index.qmd` — website summary page
- `report.qmd` — final written report (HTML/PDF)
- `viz.qmd` — interactive visualizations page
- `slides.qmd` — presentation deck
- `data/` — saved analysis-ready datasets (e.g., `processed_daily.csv`)
- `outputs/` — model tables, static figures, interactive artifacts
- `_site/` — rendered website files

All build steps use project-relative paths. No local absolute paths are required.
