#!/usr/bin/env python3
"""Build reproducible data, models, tables, and figures for JSC370 final project."""

from __future__ import annotations

import io
import json
import os
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("xgboost is required for this project pipeline.") from exc


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
STYLE_CACHE_DIR = PROJECT_DIR / ".cache"
MPL_CACHE_DIR = PROJECT_DIR / ".mplconfig"

for d in [DATA_DIR, OUTPUT_DIR, STYLE_CACHE_DIR, MPL_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(STYLE_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.serif": ["DejaVu Serif"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)

COLORS = {
    "navy": "#1f4e79",
    "blue": "#3b82b5",
    "teal": "#5aa89e",
    "orange": "#d9903d",
    "coral": "#cf7657",
    "purple": "#7a6fa9",
    "gray": "#98a3b3",
    "light_grid": "#d9e3ef",
}


WEATHER_START = "2014-01-01"
WEATHER_END = "2026-03-31"
WEATHER_LAT = 43.7001
WEATHER_LON = -79.4163
RANDOM_STATE = 370


def norm_col(col: str) -> str:
    return "".join(ch for ch in str(col).lower() if ch.isalnum())


def find_date_col(df: pd.DataFrame) -> str | None:
    candidates = []
    for c in df.columns:
        n = norm_col(c)
        score = 0
        if n in {"date", "occdate", "occurrencedate", "reportdate"}:
            score += 5
        if "date" in n:
            score += 2
        sample = pd.to_datetime(df[c].astype(str).head(600), errors="coerce")
        parse_ratio = sample.notna().mean()
        candidates.append((score, parse_ratio, c))
    if not candidates:
        return None
    best = sorted(candidates, reverse=True)[0]
    if best[0] < 2 or best[1] < 0.25:
        return None
    return best[2]


def find_delay_col(df: pd.DataFrame) -> str | None:
    candidates = []
    for c in df.columns:
        n = norm_col(c)
        if "delay" not in n:
            continue

        score = 0
        if n in {"mindelay", "minimumdelay", "delay", "delayminutes"}:
            score += 6
        if "mindelay" in n or "minimumdelay" in n:
            score += 4
        if any(x in n for x in ["code", "reason", "category", "type"]):
            score -= 4

        numeric_ratio = pd.to_numeric(df[c].head(1200), errors="coerce").notna().mean()
        candidates.append((score, numeric_ratio, c))

    if not candidates:
        return None
    best = sorted(candidates, reverse=True)[0]
    if best[0] < 0 or best[1] < 0.25:
        return None
    return best[2]


def find_optional_col(df: pd.DataFrame, keys: tuple[str, ...]) -> str | None:
    for c in df.columns:
        n = norm_col(c)
        if any(k in n for k in keys):
            return c
    return None


def standardize_ttc_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    date_col = find_date_col(df)
    delay_col = find_delay_col(df)
    line_col = find_optional_col(df, ("line",))
    station_col = find_optional_col(df, ("station", "location", "stop"))

    if not date_col or not delay_col:
        return None

    keep = [date_col, delay_col]
    rename_map = {date_col: "date", delay_col: "min_delay"}
    if line_col and line_col not in keep:
        keep.append(line_col)
        rename_map[line_col] = "line"
    if station_col and station_col not in keep:
        keep.append(station_col)
        rename_map[station_col] = "station"

    sub = df[keep].rename(columns=rename_map).copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce").dt.normalize()
    sub["min_delay"] = pd.to_numeric(sub["min_delay"], errors="coerce")
    sub = sub.dropna(subset=["date", "min_delay"])
    sub = sub[(sub["min_delay"] >= 0) & (sub["min_delay"] <= 600)]
    if sub.empty:
        return None

    if "line" in sub.columns:
        sub["line"] = sub["line"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    if "station" in sub.columns:
        sub["station"] = sub["station"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

    return sub


def classify_weather(precip_mm: float, snowfall_cm: float) -> str:
    if pd.isna(precip_mm):
        return "Unknown"
    if precip_mm >= 10 or (pd.notna(snowfall_cm) and snowfall_cm >= 5):
        return "Heavy Rain/Snow"
    if precip_mm > 0 or (pd.notna(snowfall_cm) and snowfall_cm > 0):
        return "Light Rain/Snow"
    return "Clear"


def fetch_weather() -> pd.DataFrame:
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "start_date": WEATHER_START,
        "end_date": WEATHER_END,
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "precipitation_sum",
                "precipitation_hours",
                "snowfall_sum",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
            ]
        ),
        "timezone": "America/Toronto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive"
    res = requests.get(url, params=params, timeout=90)
    res.raise_for_status()
    raw = res.json()["daily"]
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["time"]),
            "temp_max": raw["temperature_2m_max"],
            "temp_min": raw["temperature_2m_min"],
            "feels_like_max": raw["apparent_temperature_max"],
            "precip_mm": raw["precipitation_sum"],
            "precip_hours": raw["precipitation_hours"],
            "snowfall_cm": raw["snowfall_sum"],
            "wind_speed_max": raw["wind_speed_10m_max"],
            "gust_speed": raw["wind_gusts_10m_max"],
        }
    )
    weather["mean_temp_c"] = (weather["temp_max"] + weather["temp_min"]) / 2
    weather["precip_intensity"] = np.where(
        weather["precip_hours"] > 0,
        weather["precip_mm"] / weather["precip_hours"],
        0.0,
    )
    weather["is_freezing"] = (weather["temp_min"] < 0).astype(int)
    weather["is_heatwave"] = (weather["temp_max"] > 32).astype(int)
    return weather


def fetch_ttc_incidents() -> tuple[pd.DataFrame, pd.DataFrame]:
    package_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show"
    res = requests.get(package_url, params={"id": "ttc-subway-delay-data"}, timeout=90)
    res.raise_for_status()
    resources = res.json()["result"]["resources"]

    records = []
    pull_log = []

    for r in resources:
        fmt = str(r.get("format", "")).upper()
        if fmt not in {"CSV", "XLS", "XLSX"}:
            continue

        name = r.get("name", "unknown_resource")
        url = r.get("url")
        if not url:
            continue

        try:
            f = requests.get(url, timeout=90)
            f.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            pull_log.append({"resource": name, "status": f"failed: {exc}"})
            continue

        try:
            if fmt == "CSV":
                df = pd.read_csv(io.BytesIO(f.content), low_memory=False)
                sub = standardize_ttc_frame(df)
                if sub is None:
                    pull_log.append(
                        {"resource": name, "status": "skipped: missing usable date/delay columns"}
                    )
                    continue
                records.append(sub)
                pull_log.append({"resource": name, "status": f"ok ({len(sub):,} rows)"})
            else:
                sheet_map = pd.read_excel(io.BytesIO(f.content), sheet_name=None)
                parsed_sheets = []
                for _, sheet_df in sheet_map.items():
                    sub = standardize_ttc_frame(sheet_df)
                    if sub is not None:
                        parsed_sheets.append(sub)

                if not parsed_sheets:
                    pull_log.append(
                        {"resource": name, "status": "skipped: no parseable sheet with date/delay"}
                    )
                    continue

                sub_all = pd.concat(parsed_sheets, ignore_index=True)
                records.append(sub_all)
                pull_log.append(
                    {
                        "resource": name,
                        "status": (
                            f"ok ({len(sub_all):,} rows from {len(parsed_sheets)} sheet(s))"
                        ),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            pull_log.append({"resource": name, "status": f"failed: {exc}"})
            continue

    if not records:
        raise RuntimeError("No TTC delay records could be parsed from CKAN resources.")

    incidents = pd.concat(records, ignore_index=True)
    incidents = incidents.sort_values("date").reset_index(drop=True)

    daily = (
        incidents.groupby("date", as_index=False)
        .agg(
            total_delay_mins=("min_delay", "sum"),
            total_incidents=("min_delay", "size"),
            median_incident_delay=("min_delay", "median"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    pd.DataFrame(pull_log).to_csv(OUTPUT_DIR / "ttc_resource_pull_log.csv", index=False)
    return incidents, daily


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict[str, float | str]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "model": model_name,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def classification_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def build_outputs(df: pd.DataFrame) -> None:
    features = [
        "precip_mm",
        "snowfall_cm",
        "gust_speed",
        "mean_temp_c",
        "precip_intensity",
        "is_weekend_int",
        "month_num",
    ]
    model_source = df.loc[df["total_incidents"] > 0].copy()
    model_df = model_source[features + ["total_delay_mins"]].dropna().copy()
    severe_threshold = float(model_df["total_delay_mins"].quantile(0.75))
    model_df["severe_day"] = (model_df["total_delay_mins"] >= severe_threshold).astype(int)

    # Regression models
    X_reg = model_df[features]
    y_reg = model_df["total_delay_mins"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=RANDOM_STATE
    )

    lr = LinearRegression()
    dt_reg = DecisionTreeRegressor(
        max_depth=7,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
    )
    rf_reg = RandomForestRegressor(
        n_estimators=700,
        max_depth=12,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_reg = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=4,
    )

    lr.fit(Xr_train, yr_train)
    dt_reg.fit(Xr_train, yr_train)
    rf_reg.fit(Xr_train, yr_train)
    xgb_reg.fit(Xr_train, yr_train)

    pred_lr = lr.predict(Xr_test)
    pred_dt_reg = dt_reg.predict(Xr_test)
    pred_rf = rf_reg.predict(Xr_test)
    pred_xgb_reg = xgb_reg.predict(Xr_test)

    reg_metrics = pd.DataFrame(
        [
            regression_metrics(yr_test, pred_lr, "Linear Regression"),
            regression_metrics(yr_test, pred_dt_reg, "Decision Tree Regressor"),
            regression_metrics(yr_test, pred_rf, "Random Forest Regressor"),
            regression_metrics(yr_test, pred_xgb_reg, "XGBoost Regressor"),
        ]
    ).sort_values("rmse")
    reg_metrics.to_csv(OUTPUT_DIR / "table_regression_metrics.csv", index=False)

    reg_importance_rf = pd.DataFrame(
        {"feature": features, "importance": rf_reg.feature_importances_}
    ).sort_values("importance", ascending=False)
    reg_importance_xgb = pd.DataFrame(
        {"feature": features, "importance": xgb_reg.feature_importances_}
    ).sort_values("importance", ascending=False)
    reg_importance_rf.to_csv(OUTPUT_DIR / "table_feature_importance_regression_rf.csv", index=False)
    reg_importance_xgb.to_csv(OUTPUT_DIR / "table_feature_importance_regression_xgb.csv", index=False)

    best_reg_model_name = str(reg_metrics.iloc[0]["model"])
    if best_reg_model_name == "Random Forest Regressor":
        reg_importance = reg_importance_rf.copy()
    elif best_reg_model_name == "XGBoost Regressor":
        reg_importance = reg_importance_xgb.copy()
    else:
        # Linear model has coefficient-based interpretation rather than tree importance.
        # Use XGBoost importance table for a richer non-linear feature view.
        reg_importance = reg_importance_xgb.copy()
    reg_importance["importance_model"] = (
        best_reg_model_name if best_reg_model_name != "Linear Regression" else "XGBoost Regressor"
    )
    reg_importance.to_csv(OUTPUT_DIR / "table_feature_importance_regression.csv", index=False)

    # Classification models
    X_cls = model_df[features]
    y_cls = model_df["severe_day"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls,
        y_cls,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )

    logit = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    dt_cls = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf_cls = RandomForestClassifier(
        n_estimators=700,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_cls = XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=3.0,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=4,
    )

    logit.fit(Xc_train, yc_train)
    dt_cls.fit(Xc_train, yc_train)
    rf_cls.fit(Xc_train, yc_train)
    xgb_cls.fit(Xc_train, yc_train)

    pred_logit = logit.predict(Xc_test)
    prob_logit = logit.predict_proba(Xc_test)[:, 1]
    pred_dt_cls = dt_cls.predict(Xc_test)
    prob_dt_cls = dt_cls.predict_proba(Xc_test)[:, 1]
    pred_rf_cls = rf_cls.predict(Xc_test)
    prob_rf_cls = rf_cls.predict_proba(Xc_test)[:, 1]
    pred_xgb_cls = xgb_cls.predict(Xc_test)
    prob_xgb_cls = xgb_cls.predict_proba(Xc_test)[:, 1]

    cls_metrics = pd.DataFrame(
        [
            classification_metrics(yc_test, pred_logit, prob_logit, "Logistic Regression"),
            classification_metrics(yc_test, pred_dt_cls, prob_dt_cls, "Decision Tree Classifier"),
            classification_metrics(yc_test, pred_rf_cls, prob_rf_cls, "Random Forest Classifier"),
            classification_metrics(yc_test, pred_xgb_cls, prob_xgb_cls, "XGBoost Classifier"),
        ]
    ).sort_values("roc_auc", ascending=False)
    cls_metrics.to_csv(OUTPUT_DIR / "table_classification_metrics.csv", index=False)

    cls_importance_rf = pd.DataFrame(
        {"feature": features, "importance": rf_cls.feature_importances_}
    ).sort_values("importance", ascending=False)
    cls_importance_xgb = pd.DataFrame(
        {"feature": features, "importance": xgb_cls.feature_importances_}
    ).sort_values("importance", ascending=False)
    cls_importance_rf.to_csv(OUTPUT_DIR / "table_feature_importance_classification_rf.csv", index=False)
    cls_importance_xgb.to_csv(OUTPUT_DIR / "table_feature_importance_classification_xgb.csv", index=False)
    best_cls_model_name = str(cls_metrics.iloc[0]["model"])
    if best_cls_model_name == "Random Forest Classifier":
        cls_importance = cls_importance_rf.copy()
    elif best_cls_model_name == "XGBoost Classifier":
        cls_importance = cls_importance_xgb.copy()
    else:
        cls_importance = cls_importance_xgb.copy()
    cls_importance["importance_model"] = (
        best_cls_model_name if best_cls_model_name != "Logistic Regression" else "XGBoost Classifier"
    )
    cls_importance.to_csv(OUTPUT_DIR / "table_feature_importance_classification.csv", index=False)

    # Summary table for weather categories
    summary_weather = (
        model_source.groupby("weather_type", as_index=False)
        .agg(
            n_days=("date", "size"),
            mean_delay_mins=("total_delay_mins", "mean"),
            median_delay_mins=("total_delay_mins", "median"),
            mean_incidents=("total_incidents", "mean"),
            mean_precip_mm=("precip_mm", "mean"),
        )
        .round(2)
    )
    summary_weather.to_csv(OUTPUT_DIR / "table_weather_summary.csv", index=False)

    # Static figures for report
    monthly = (
        df.set_index("date")
        .resample("ME")
        .agg(
            total_delay_mins=("total_delay_mins", "sum"),
            precip_mm=("precip_mm", "sum"),
            incident_days=("total_incidents", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    monthly["delay_for_plot"] = monthly["total_delay_mins"]

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(
        monthly["date"],
        monthly["delay_for_plot"],
        color=COLORS["navy"],
        linewidth=2.4,
        marker="o",
        markersize=2.8,
    )
    ax1.set_ylabel("Monthly total delay minutes", color=COLORS["navy"])
    ax1.tick_params(axis="y", labelcolor=COLORS["navy"])
    ax1.set_xlabel("Month")
    ax1.grid(color=COLORS["light_grid"], linewidth=0.8, alpha=0.9)

    ax2 = ax1.twinx()
    ax2.bar(
        monthly["date"],
        monthly["precip_mm"],
        alpha=0.35,
        color=COLORS["teal"],
        width=18,
    )
    ax2.set_ylabel("Monthly precipitation (mm)", color=COLORS["teal"])
    ax2.tick_params(axis="y", labelcolor=COLORS["teal"])

    ax1.set_title("Monthly TTC Delay Minutes and Precipitation (Toronto, 2014-2026)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_monthly_delay_precip.png", dpi=240)
    plt.close(fig)

    box_df = df.loc[df["total_incidents"] > 0].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=box_df,
        x="weather_type",
        y="total_delay_mins",
        order=["Clear", "Light Rain/Snow", "Heavy Rain/Snow"],
        palette=[COLORS["blue"], COLORS["teal"], COLORS["coral"]],
        ax=ax,
    )
    ax.set_title("Daily Delay Distribution by Weather Category")
    ax.set_xlabel("Weather category")
    ax.set_ylabel("Daily total delay (minutes)")
    ax.grid(axis="y", color=COLORS["light_grid"], linewidth=0.8, alpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_weather_boxplot.png", dpi=240)
    plt.close(fig)

    reg_pred_map = {
        "Linear Regression": pred_lr,
        "Decision Tree Regressor": pred_dt_reg,
        "Random Forest Regressor": pred_rf,
        "XGBoost Regressor": pred_xgb_reg,
    }
    best_reg_pred = reg_pred_map[best_reg_model_name]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        yr_test,
        best_reg_pred,
        s=20,
        alpha=0.58,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.2,
    )
    lim_min = float(min(yr_test.min(), best_reg_pred.min()))
    lim_max = float(max(yr_test.max(), best_reg_pred.max()))
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.8,
    )
    ax.set_xlabel("Actual delay minutes")
    ax.set_ylabel("Predicted delay minutes")
    ax.set_title(f"{best_reg_model_name}: Actual vs Predicted")
    ax.grid(color=COLORS["light_grid"], linewidth=0.8, alpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_rf_actual_vs_pred.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    imp_top = reg_importance.head(7).iloc[::-1]
    bar_colors = [
        COLORS["purple"],
        COLORS["coral"],
        COLORS["orange"],
        COLORS["teal"],
        "#6f93bf",
        COLORS["blue"],
        COLORS["navy"],
    ]
    ax.barh(imp_top["feature"], imp_top["importance"], color=bar_colors[: len(imp_top)])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    imp_model_label = reg_importance["importance_model"].iloc[0]
    ax.set_title(f"Top Predictors of Daily Delay ({imp_model_label})")
    ax.grid(axis="x", color=COLORS["light_grid"], linewidth=0.8, alpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_feature_importance.png", dpi=240)
    plt.close(fig)

    fpr_logit, tpr_logit, _ = roc_curve(yc_test, prob_logit)
    fpr_dt, tpr_dt, _ = roc_curve(yc_test, prob_dt_cls)
    fpr_rf, tpr_rf, _ = roc_curve(yc_test, prob_rf_cls)
    fpr_xgb, tpr_xgb, _ = roc_curve(yc_test, prob_xgb_cls)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr_logit,
        tpr_logit,
        label="Logistic Regression",
        color=COLORS["orange"],
        linewidth=2.2,
    )
    ax.plot(
        fpr_dt,
        tpr_dt,
        label="Decision Tree",
        color=COLORS["teal"],
        linewidth=2.0,
    )
    ax.plot(
        fpr_rf,
        tpr_rf,
        label="Random Forest",
        color=COLORS["navy"],
        linewidth=2.4,
    )
    ax.plot(
        fpr_xgb,
        tpr_xgb,
        label="XGBoost",
        color=COLORS["purple"],
        linewidth=2.2,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Severe Delay Day Classification")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#c9d7e6")
    ax.grid(color=COLORS["light_grid"], linewidth=0.8, alpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_roc_curve.png", dpi=240)
    plt.close(fig)

    cls_pred_map = {
        "Logistic Regression": pred_logit,
        "Decision Tree Classifier": pred_dt_cls,
        "Random Forest Classifier": pred_rf_cls,
        "XGBoost Classifier": pred_xgb_cls,
    }
    best_cls_pred = cls_pred_map[best_cls_model_name]
    cm = confusion_matrix(yc_test, best_cls_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["Not Severe", "Severe"],
        yticklabels=["Not Severe", "Severe"],
        ax=ax,
    )
    ax.set_title(f"{best_cls_model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_confusion_matrix.png", dpi=240)
    plt.close(fig)

    # Interactive visualizations (for HW5 requirement)
    roll = df[["date", "total_delay_mins", "precip_mm"]].copy()
    roll["delay_30d_ma"] = roll["total_delay_mins"].rolling(30, min_periods=1).mean()

    fig_int1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_int1.add_trace(
        go.Scatter(
            x=roll["date"],
            y=roll["total_delay_mins"],
            name="Daily total delay",
            mode="lines",
            line={"color": COLORS["blue"], "width": 1.1},
            opacity=0.65,
        ),
        secondary_y=False,
    )
    fig_int1.add_trace(
        go.Scatter(
            x=roll["date"],
            y=roll["delay_30d_ma"],
            name="30-day moving average",
            mode="lines",
            line={"color": COLORS["orange"], "width": 2.4},
        ),
        secondary_y=False,
    )
    fig_int1.add_trace(
        go.Bar(
            x=roll["date"],
            y=roll["precip_mm"],
            name="Precipitation (mm)",
            marker_color=COLORS["teal"],
            opacity=0.34,
        ),
        secondary_y=True,
    )
    fig_int1.update_layout(
        title="Figure 1: Delay Trend with Precipitation",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 50, "r": 50, "t": 70, "b": 40},
    )
    fig_int1.update_yaxes(title_text="Daily delay (minutes)", secondary_y=False)
    fig_int1.update_yaxes(title_text="Precipitation (mm)", secondary_y=True)
    fig_int1.write_html(OUTPUT_DIR / "viz1_timeseries.html", include_plotlyjs="cdn")

    monthly_heat = (
        df.assign(year=df["date"].dt.year, month=df["date"].dt.month)
        .groupby(["year", "month"], as_index=False)
        .agg(mean_delay_mins=("total_delay_mins", "mean"))
    )
    heat = (
        monthly_heat.pivot(index="year", columns="month", values="mean_delay_mins")
        .reindex(columns=list(range(1, 13)))
        .sort_index()
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_int2 = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=month_labels,
            y=heat.index.astype(str),
            colorscale="YlGnBu",
            colorbar={"title": "Mean delay (min)"},
            hovertemplate="Year %{y}<br>Month %{x}<br>Mean delay %{z:.1f} min<extra></extra>",
        )
    )
    fig_int2.update_layout(
        title="Figure 2: Monthly Mean Daily Delay (Year × Month)",
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Year",
        margin={"l": 60, "r": 25, "t": 70, "b": 50},
    )
    fig_int2.write_html(OUTPUT_DIR / "viz2_heatmap.html", include_plotlyjs="cdn")

    # For boxplot comparability, exclude days with zero incidents.
    box_df = df.loc[df["total_incidents"] > 0].copy()
    fig_int3 = px.box(
        box_df,
        x="weather_type",
        y="total_delay_mins",
        color="day_type",
        category_orders={
            "weather_type": ["Clear", "Light Rain/Snow", "Heavy Rain/Snow"],
            "day_type": ["Weekday", "Weekend"],
        },
        color_discrete_map={"Weekday": COLORS["navy"], "Weekend": COLORS["orange"]},
        points=False,
        title="Figure 3: Weekday vs Weekend Delay by Weather Type (Incident Days)",
        labels={
            "weather_type": "Weather category",
            "total_delay_mins": "Daily total delay (minutes)",
            "day_type": "Day type",
        },
    )
    fig_int3.update_layout(template="plotly_white", margin={"l": 40, "r": 20, "t": 70, "b": 40})
    fig_int3.write_html(OUTPUT_DIR / "viz3_boxplot.html", include_plotlyjs="cdn")

    # Save compact key numbers for inline report text
    key_numbers = {
        "n_days": int(len(df)),
        "n_days_with_incidents": int((df["total_incidents"] > 0).sum()),
        "n_zero_incident_days": int((df["total_incidents"] == 0).sum()),
        "n_model_days": int(len(model_df)),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "severe_threshold": severe_threshold,
        "best_reg_model": best_reg_model_name,
        "best_reg_rmse": float(reg_metrics.iloc[0]["rmse"]),
        "best_reg_r2": float(reg_metrics.iloc[0]["r2"]),
        "best_cls_model": best_cls_model_name,
        "best_cls_auc": float(cls_metrics.iloc[0]["roc_auc"]),
        "best_cls_f1": float(cls_metrics.iloc[0]["f1"]),
        "xgb_reg_rmse": float(
            reg_metrics.loc[reg_metrics["model"] == "XGBoost Regressor", "rmse"].iloc[0]
        ),
        "xgb_reg_r2": float(
            reg_metrics.loc[reg_metrics["model"] == "XGBoost Regressor", "r2"].iloc[0]
        ),
        "xgb_cls_auc": float(
            cls_metrics.loc[cls_metrics["model"] == "XGBoost Classifier", "roc_auc"].iloc[0]
        ),
        "xgb_cls_f1": float(
            cls_metrics.loc[cls_metrics["model"] == "XGBoost Classifier", "f1"].iloc[0]
        ),
        "precip_delay_corr": float(
            model_source[["precip_mm", "total_delay_mins"]].corr().iloc[0, 1]
        ),
        "severe_rate": float(model_df["severe_day"].mean()),
        "weekday_mean_delay": float(
            model_source.loc[model_source["day_type"] == "Weekday", "total_delay_mins"].mean()
        ),
        "weekend_mean_delay": float(
            model_source.loc[model_source["day_type"] == "Weekend", "total_delay_mins"].mean()
        ),
        "heavy_mean_delay": float(
            model_source.loc[
                model_source["weather_type"] == "Heavy Rain/Snow", "total_delay_mins"
            ].mean()
        ),
        "clear_mean_delay": float(
            model_source.loc[model_source["weather_type"] == "Clear", "total_delay_mins"].mean()
        ),
    }
    key_numbers["heavy_vs_clear_ratio"] = (
        key_numbers["heavy_mean_delay"] / key_numbers["clear_mean_delay"]
        if key_numbers["clear_mean_delay"] > 0
        else float("nan")
    )
    (OUTPUT_DIR / "key_numbers.json").write_text(json.dumps(key_numbers, indent=2), encoding="utf-8")

    # Persist prediction tables
    pred_table = pd.DataFrame(
        {
            "actual_delay_mins": yr_test.values,
            "pred_rf_delay_mins": pred_rf,
            "pred_lr_delay_mins": pred_lr,
            "pred_dt_delay_mins": pred_dt_reg,
            "pred_xgb_delay_mins": pred_xgb_reg,
        }
    )
    pred_table.to_csv(OUTPUT_DIR / "table_regression_predictions_test.csv", index=False)

    cls_pred = pd.DataFrame(
        {
            "actual_severe": yc_test.values,
            "pred_logit": pred_logit,
            "prob_logit": prob_logit,
            "pred_dt": pred_dt_cls,
            "prob_dt": prob_dt_cls,
            "pred_rf": pred_rf_cls,
            "prob_rf": prob_rf_cls,
            "pred_xgb": pred_xgb_cls,
            "prob_xgb": prob_xgb_cls,
        }
    )
    cls_pred.to_csv(OUTPUT_DIR / "table_classification_predictions_test.csv", index=False)


def main() -> None:
    print("1) Fetching weather data...")
    weather = fetch_weather()
    print(f"   Weather rows: {len(weather):,}")

    print("2) Fetching TTC delay data...")
    incidents, ttc_daily = fetch_ttc_incidents()
    print(f"   Incident rows: {len(incidents):,}")
    print(f"   Daily rows: {len(ttc_daily):,}")

    print("3) Building merged analysis dataset...")
    analysis_start = max(ttc_daily["date"].min(), weather["date"].min())
    analysis_end = min(ttc_daily["date"].max(), weather["date"].max())
    calendar = pd.DataFrame({"date": pd.date_range(analysis_start, analysis_end, freq="D")})

    merged = (
        calendar.merge(weather, on="date", how="left")
        .merge(ttc_daily, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # No incident record on a calendar day is treated as zero daily delay by construction.
    merged["total_delay_mins"] = merged["total_delay_mins"].fillna(0.0)
    merged["total_incidents"] = merged["total_incidents"].fillna(0).astype(int)
    merged["median_incident_delay"] = merged["median_incident_delay"].fillna(0.0)

    merged["month_num"] = merged["date"].dt.month
    merged["month"] = merged["date"].dt.month_name()
    merged["month_abbr"] = merged["date"].dt.strftime("%b")
    merged["day_of_week"] = merged["date"].dt.day_name()
    merged["is_weekend"] = merged["date"].dt.dayofweek >= 5
    merged["is_weekend_int"] = merged["is_weekend"].astype(int)
    merged["day_type"] = np.where(merged["is_weekend"], "Weekend", "Weekday")
    merged["weather_type"] = merged.apply(
        lambda r: classify_weather(r["precip_mm"], r["snowfall_cm"]), axis=1
    )
    merged["avg_delay_per_incident"] = merged["total_delay_mins"] / merged["total_incidents"].replace(0, np.nan)
    incident_delay_threshold = merged.loc[merged["total_incidents"] > 0, "total_delay_mins"].quantile(0.75)
    merged["severe_day"] = np.where(
        (merged["total_incidents"] > 0) & (merged["total_delay_mins"] >= incident_delay_threshold),
        1,
        0,
    )

    coverage_by_year = (
        merged.assign(year=merged["date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            calendar_days=("date", "size"),
            incident_days=("total_incidents", lambda s: int((s > 0).sum())),
        )
    )
    coverage_by_year["coverage_pct"] = (
        coverage_by_year["incident_days"] / coverage_by_year["calendar_days"] * 100
    )
    coverage_by_year.to_csv(OUTPUT_DIR / "table_data_coverage_by_year.csv", index=False)

    merged.to_csv(DATA_DIR / "processed_daily.csv", index=False)
    # Keep a representative sample of incident-level data for transparency.
    # Random sampling avoids over-representing early years when data is date-sorted.
    sample_n = min(15000, len(incidents))
    incidents_sample = incidents.sample(n=sample_n, random_state=RANDOM_STATE).sort_values("date")
    incidents_sample.to_csv(DATA_DIR / "ttc_incidents_sample.csv", index=False)

    print("4) Training models and exporting tables/figures...")
    build_outputs(merged)

    print("Done.")
    print(f"- Processed data: {DATA_DIR / 'processed_daily.csv'}")
    print(f"- Outputs folder: {OUTPUT_DIR}")
    print(f"- Run Quarto next to render report/site/slides.")


if __name__ == "__main__":
    main()
