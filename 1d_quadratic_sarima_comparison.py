# 1d_quadratic_sarima_comparison.py
# Compare the annual quadratic CO2 model with SARIMA(1,1,1)(2,1,2,12).
#
# This script reads existing outputs from:
#   annual_fit_outputs/
#   SARIMA图/
# It does not refit any model.

from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("question_1d_outputs")

ANNUAL_OUTPUT_DIR = Path("annual_fit_outputs")
QUADRATIC_DIR = ANNUAL_OUTPUT_DIR / "models" / "quadratic"

SARIMA_OUTPUT_DIR = Path("SARIMA图")
SARIMA_MODEL_KEY = "SARIMA_1_1_1_2_1_2_12"
SARIMA_MODEL_NAME = "SARIMA(1,1,1)(2,1,2,12)"
SARIMA_DIR = SARIMA_OUTPUT_DIR / "models" / SARIMA_MODEL_KEY

TEST_START_YEAR = 2012


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def evaluate_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residuals = observed - predicted

    sse = float(np.sum(residuals ** 2))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / observed)) * 100)

    ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = float(1 - sse / ss_total) if ss_total != 0 else float("nan")

    return {
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    }


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def annualize_fitted_values(df):
    annual_df = df.copy()
    annual_df["Date"] = pd.to_datetime(annual_df["Date"] + "-01")
    annual_df["Year"] = annual_df["Date"].dt.year

    annual_df = (
        annual_df.groupby("Year", as_index=False)
        .agg(
            Observed_CO2=("Observed_CO2", "mean"),
            Fitted_CO2=("Fitted_CO2", "mean"),
        )
    )
    annual_df["Residual"] = annual_df["Observed_CO2"] - annual_df["Fitted_CO2"]
    annual_df["Absolute_Residual"] = annual_df["Residual"].abs()
    return annual_df


def annualize_test_forecasts(df):
    annual_df = df.copy()
    annual_df["Date"] = pd.to_datetime(annual_df["Date"] + "-01")
    annual_df["Year"] = annual_df["Date"].dt.year

    annual_df = (
        annual_df.groupby("Year", as_index=False)
        .agg(
            Observed_CO2=("Observed_CO2", "mean"),
            Forecast_CO2=("Forecast_CO2", "mean"),
        )
    )
    annual_df["Residual"] = annual_df["Observed_CO2"] - annual_df["Forecast_CO2"]
    annual_df["Absolute_Residual"] = annual_df["Residual"].abs()
    return annual_df


def load_inputs():
    paths = {
        "annual_summary": ANNUAL_OUTPUT_DIR / "annual_model_summary.csv",
        "quadratic_fitted": QUADRATIC_DIR / "quadratic_fitted_values.csv",
        "quadratic_test": QUADRATIC_DIR / "quadratic_test_predictions.csv",
        "sarima_summary": SARIMA_OUTPUT_DIR / "SARIMA_model_summary.csv",
        "sarima_fitted": SARIMA_DIR / f"{SARIMA_MODEL_KEY}_fitted_values.csv",
        "sarima_test": SARIMA_DIR / f"{SARIMA_MODEL_KEY}_test_forecasts.csv",
    }
    for path in paths.values():
        require_file(path)

    return {
        key: pd.read_csv(path)
        for key, path in paths.items()
    }


def build_metric_tables(data):
    annual_summary = data["annual_summary"]
    sarima_summary = data["sarima_summary"]
    quadratic_fitted = data["quadratic_fitted"]
    quadratic_test = data["quadratic_test"]
    sarima_fitted = data["sarima_fitted"]
    sarima_test = data["sarima_test"]

    quadratic_train = quadratic_fitted[quadratic_fitted["Fit_Stage"] == "training_data"].copy()
    sarima_train = annualize_fitted_values(
        sarima_fitted[sarima_fitted["Fit_Stage"] == "training_data"].copy()
    )
    sarima_test_annual = annualize_test_forecasts(sarima_test)

    metric_rows = []
    metric_sources = [
        (
            "Quadratic annual",
            "Train",
            quadratic_train["Observed_CO2"],
            quadratic_train["Fitted_CO2"],
            "annual training fitted values",
        ),
        (
            "Quadratic annual",
            "Test",
            quadratic_test["Observed_CO2"],
            quadratic_test["Forecast_CO2"],
            "annual test forecasts",
        ),
        (
            "SARIMA annualized",
            "Train",
            sarima_train["Observed_CO2"],
            sarima_train["Fitted_CO2"],
            "monthly training fitted values averaged by year",
        ),
        (
            "SARIMA annualized",
            "Test",
            sarima_test_annual["Observed_CO2"],
            sarima_test_annual["Forecast_CO2"],
            "monthly test forecasts averaged by year",
        ),
    ]

    for model, stage, observed, predicted, source_note in metric_sources:
        metrics = evaluate_metrics(observed, predicted)
        for metric_name, value in metrics.items():
            metric_rows.append({
                "Model": model,
                "Stage": stage,
                "Metric": metric_name,
                "Value": value,
                "Source_Note": source_note,
            })

    annualized_metrics = pd.DataFrame(metric_rows)

    quadratic_summary_row = annual_summary[annual_summary["Model_Key"] == "quadratic"].iloc[0]
    sarima_summary_row = sarima_summary[
        (sarima_summary["Model"] == SARIMA_MODEL_NAME)
        & (sarima_summary["Trend"] == "c")
    ].iloc[0]

    source_summary_rows = []
    for model, row in [
        ("Quadratic annual source summary", quadratic_summary_row),
        ("SARIMA monthly source summary", sarima_summary_row),
    ]:
        for stage in ["Train", "Test"]:
            for metric in ["RMSE", "MAE", "MAPE", "R2"]:
                column = f"{stage}_{metric}"
                source_summary_rows.append({
                    "Model": model,
                    "Stage": stage,
                    "Metric": metric,
                    "Value": float(row[column]),
                    "Source_Note": "directly read from original summary output",
                })

    source_metrics = pd.DataFrame(source_summary_rows)
    return annualized_metrics, source_metrics, sarima_train, sarima_test_annual


def build_fit_comparison_series(data, sarima_train, sarima_test_annual):
    quadratic_fitted = data["quadratic_fitted"]
    quadratic_test = data["quadratic_test"]
    sarima_fitted = data["sarima_fitted"]

    quadratic_full = (
        quadratic_fitted[quadratic_fitted["Fit_Stage"] == "full_data"]
        [["Year", "Observed_CO2", "Fitted_CO2"]]
        .rename(columns={"Fitted_CO2": "Quadratic_Full_Fitted_CO2"})
    )
    quadratic_train = (
        quadratic_fitted[quadratic_fitted["Fit_Stage"] == "training_data"]
        [["Year", "Fitted_CO2"]]
        .rename(columns={"Fitted_CO2": "Quadratic_Train_Fitted_Or_Test_Forecast_CO2"})
    )
    quadratic_test_line = (
        quadratic_test[["Year", "Forecast_CO2"]]
        .rename(columns={"Forecast_CO2": "Quadratic_Train_Fitted_Or_Test_Forecast_CO2"})
    )
    quadratic_train_test_line = pd.concat([quadratic_train, quadratic_test_line], ignore_index=True)

    sarima_full = annualize_fitted_values(
        sarima_fitted[sarima_fitted["Fit_Stage"] == "full_data"].copy()
    )[["Year", "Fitted_CO2"]].rename(columns={"Fitted_CO2": "SARIMA_Full_Fitted_CO2"})

    sarima_train_line = sarima_train[["Year", "Fitted_CO2"]].rename(
        columns={"Fitted_CO2": "SARIMA_Train_Fitted_Or_Test_Forecast_CO2"}
    )
    sarima_test_line = sarima_test_annual[["Year", "Forecast_CO2"]].rename(
        columns={"Forecast_CO2": "SARIMA_Train_Fitted_Or_Test_Forecast_CO2"}
    )
    sarima_train_test_line = pd.concat([sarima_train_line, sarima_test_line], ignore_index=True)

    fit_df = quadratic_full.merge(quadratic_train_test_line, on="Year", how="left")
    fit_df = fit_df.merge(sarima_full, on="Year", how="left")
    fit_df = fit_df.merge(sarima_train_test_line, on="Year", how="left")
    fit_df = fit_df.sort_values("Year").reset_index(drop=True)

    return fit_df


def build_test_residual_series(data, sarima_test_annual):
    quadratic_test = data["quadratic_test"]

    residual_df = pd.DataFrame({
        "Year": quadratic_test["Year"].astype(int),
        "Quadratic_Residual": quadratic_test["Error"].astype(float),
    })

    sarima_residual = sarima_test_annual[["Year", "Residual"]].rename(
        columns={"Residual": "SARIMA_Annualized_Residual"}
    )
    residual_df = residual_df.merge(sarima_residual, on="Year", how="left")

    return residual_df


def build_full_residual_series(data):
    quadratic_fitted = data["quadratic_fitted"]
    sarima_fitted = data["sarima_fitted"]

    quadratic_full = (
        quadratic_fitted[quadratic_fitted["Fit_Stage"] == "full_data"]
        [["Year", "Residual"]]
        .rename(columns={"Residual": "Quadratic_Full_Data_Residual"})
    )

    sarima_full = annualize_fitted_values(
        sarima_fitted[sarima_fitted["Fit_Stage"] == "full_data"].copy()
    )[["Year", "Residual"]].rename(
        columns={"Residual": "SARIMA_Annualized_Full_Data_Residual"}
    )

    residual_df = quadratic_full.merge(sarima_full, on="Year", how="outer")
    residual_df = residual_df.sort_values("Year").reset_index(drop=True)
    return residual_df


def plot_error_metrics(metrics_df):
    plot_metrics = ["RMSE", "MAE", "MAPE"]
    stage_order = ["Train", "Test"]
    model_order = ["Quadratic annual", "SARIMA annualized"]
    colors = {
        "Quadratic annual": "#1f77b4",
        "SARIMA annualized": "#ff7f0e",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, plot_metrics):
        metric_df = metrics_df[metrics_df["Metric"] == metric]
        x_values = np.arange(len(stage_order))
        width = 0.34

        for offset, model in [(-width / 2, model_order[0]), (width / 2, model_order[1])]:
            values = []
            for stage in stage_order:
                value = metric_df[
                    (metric_df["Model"] == model)
                    & (metric_df["Stage"] == stage)
                ]["Value"].iloc[0]
                values.append(value)

            bars = ax.bar(
                x_values + offset,
                values,
                width,
                label=model,
                color=colors[model],
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)

        ax.set_xticks(x_values)
        ax.set_xticklabels(stage_order)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.35)
        if metric == "MAPE":
            ax.set_ylabel("Percent / %")
        else:
            ax.set_ylabel("CO2 error / ppm")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Quadratic Annual vs SARIMA(1,1,1)(2,1,2,12): Train/Test Error Metrics", y=1.04)
    save_plot(OUTPUT_DIR / "question_1d_quadratic_sarima_error_metrics_bar.svg")


def plot_fit_comparison(fit_df):
    plt.figure(figsize=(13, 7))
    plt.plot(
        fit_df["Year"],
        fit_df["Observed_CO2"],
        color="black",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Observed annual CO2",
    )
    plt.plot(
        fit_df["Year"],
        fit_df["Quadratic_Full_Fitted_CO2"],
        color="#1f77b4",
        linewidth=2.2,
        label="Quadratic full-data fit",
    )
    plt.plot(
        fit_df["Year"],
        fit_df["Quadratic_Train_Fitted_Or_Test_Forecast_CO2"],
        color="#1f77b4",
        linewidth=2,
        linestyle="--",
        label="Quadratic training fit + test forecast",
    )
    plt.plot(
        fit_df["Year"],
        fit_df["SARIMA_Full_Fitted_CO2"],
        color="#ff7f0e",
        linewidth=2.2,
        label="SARIMA full-data fit annualized",
    )
    plt.plot(
        fit_df["Year"],
        fit_df["SARIMA_Train_Fitted_Or_Test_Forecast_CO2"],
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
        label="SARIMA training fit + test forecast annualized",
    )
    plt.axvline(TEST_START_YEAR, color="gray", linestyle=":", linewidth=1.5, label="Test period starts")
    plt.xlabel("Year")
    plt.ylabel("CO2 concentration / ppm")
    plt.title("Quadratic Annual and SARIMA Fit/Forecast Comparison")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    save_plot(OUTPUT_DIR / "question_1d_quadratic_sarima_fit_comparison_line.svg")


def plot_test_residual_comparison(residual_df):
    plt.figure(figsize=(11, 6))
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.4, label="Zero residual")
    plt.plot(
        residual_df["Year"],
        residual_df["Quadratic_Residual"],
        marker="o",
        linewidth=2,
        label="Quadratic annual residual",
    )
    plt.plot(
        residual_df["Year"],
        residual_df["SARIMA_Annualized_Residual"],
        marker="o",
        linewidth=2,
        label="SARIMA annualized residual",
    )
    plt.xlabel("Year")
    plt.ylabel("Residual / ppm")
    plt.title("Test Residual Comparison: Quadratic Annual vs SARIMA(1,1,1)(2,1,2,12)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    save_plot(OUTPUT_DIR / "question_1d_quadratic_sarima_test_residual_comparison_line.svg")


def plot_full_residual_comparison(residual_df):
    plt.figure(figsize=(13, 6))
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.4, label="Zero residual")
    plt.plot(
        residual_df["Year"],
        residual_df["Quadratic_Full_Data_Residual"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Quadratic full-data residual",
    )
    plt.plot(
        residual_df["Year"],
        residual_df["SARIMA_Annualized_Full_Data_Residual"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="SARIMA annualized full-data residual",
    )
    plt.xlabel("Year")
    plt.ylabel("Residual / ppm")
    plt.title("Full-Data Residual Comparison: Quadratic Annual vs SARIMA(1,1,1)(2,1,2,12)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    save_plot(OUTPUT_DIR / "question_1d_quadratic_sarima_full_residual_comparison_line.svg")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    metrics_df, source_metrics_df, sarima_train, sarima_test_annual = build_metric_tables(data)
    fit_df = build_fit_comparison_series(data, sarima_train, sarima_test_annual)
    residual_df = build_test_residual_series(data, sarima_test_annual)
    full_residual_df = build_full_residual_series(data)

    metrics_csv = OUTPUT_DIR / "question_1d_quadratic_sarima_error_metrics_annualized.csv"
    source_metrics_csv = OUTPUT_DIR / "question_1d_quadratic_sarima_error_metrics_source_summary.csv"
    fit_csv = OUTPUT_DIR / "question_1d_quadratic_sarima_fit_comparison_series.csv"
    residual_csv = OUTPUT_DIR / "question_1d_quadratic_sarima_test_residuals.csv"
    full_residual_csv = OUTPUT_DIR / "question_1d_quadratic_sarima_full_residuals.csv"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    source_metrics_df.to_csv(source_metrics_csv, index=False, encoding="utf-8-sig")
    fit_df.to_csv(fit_csv, index=False, encoding="utf-8-sig")
    residual_df.to_csv(residual_csv, index=False, encoding="utf-8-sig")
    full_residual_df.to_csv(full_residual_csv, index=False, encoding="utf-8-sig")

    plot_error_metrics(metrics_df)
    plot_fit_comparison(fit_df)
    plot_test_residual_comparison(residual_df)
    plot_full_residual_comparison(full_residual_df)

    manifest = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "comparison_models": [
            "Quadratic annual",
            SARIMA_MODEL_NAME,
        ],
        "note": (
            "SARIMA fitted values and forecasts are averaged by calendar year for "
            "direct comparison with the annual quadratic model."
        ),
        "input_files": {
            "annual_summary": str(ANNUAL_OUTPUT_DIR / "annual_model_summary.csv"),
            "quadratic_fitted": str(QUADRATIC_DIR / "quadratic_fitted_values.csv"),
            "quadratic_test_predictions": str(QUADRATIC_DIR / "quadratic_test_predictions.csv"),
            "sarima_summary": str(SARIMA_OUTPUT_DIR / "SARIMA_model_summary.csv"),
            "sarima_fitted": str(SARIMA_DIR / f"{SARIMA_MODEL_KEY}_fitted_values.csv"),
            "sarima_test_forecasts": str(SARIMA_DIR / f"{SARIMA_MODEL_KEY}_test_forecasts.csv"),
        },
        "output_files": [
            str(metrics_csv),
            str(source_metrics_csv),
            str(fit_csv),
            str(residual_csv),
            str(full_residual_csv),
            str(OUTPUT_DIR / "question_1d_quadratic_sarima_error_metrics_bar.svg"),
            str(OUTPUT_DIR / "question_1d_quadratic_sarima_fit_comparison_line.svg"),
            str(OUTPUT_DIR / "question_1d_quadratic_sarima_test_residual_comparison_line.svg"),
            str(OUTPUT_DIR / "question_1d_quadratic_sarima_full_residual_comparison_line.svg"),
        ],
    }
    (OUTPUT_DIR / "question_1d_quadratic_sarima_manifest.json").write_text(
        json.dumps(to_builtin(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("QUESTION 1D QUADRATIC VS SARIMA COMPARISON PLOTS COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nAnnualized train/test metrics:")
    print(
        metrics_df[metrics_df["Metric"].isin(["RMSE", "MAE", "MAPE", "R2"])]
        .pivot_table(index=["Model", "Stage"], columns="Metric", values="Value")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
