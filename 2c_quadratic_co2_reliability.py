# 2c_quadratic_co2_reliability.py
# Extend the 2b quadratic centered CO2-temperature model into the future and
# run rolling-origin validation for reliability diagnostics.

from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from himcm_data import get_aligned_co2_temperature_data


OUTPUT_DIR = Path("question_2_outputs") / "2c_quadratic_co2_reliability"
FUTURE_CO2_CSV = Path("question_1c_outputs") / "question_1c_future_forecasts.csv"

FORECAST_START_YEAR = 2022
FORECAST_END_YEAR = 2100
KEY_YEARS = [2050, 2100]
ROLLING_TRAIN_END_YEARS = [1981, 1986, 1991, 1996, 2001]
MAX_FORECAST_HORIZON = 20
TEMP_CHANGE_TARGETS = [1.25, 1.50, 2.00]
BASE_PERIOD_LABEL = "1951-1980"


def reset_output_dir(output_dir):
    script_dir = Path(__file__).resolve().parent
    resolved_output_dir = (script_dir / output_dir).resolve()

    if script_dir not in resolved_output_dir.parents:
        raise RuntimeError(f"Refusing to clean outside script directory: {resolved_output_dir}")

    if resolved_output_dir.exists():
        shutil.rmtree(resolved_output_dir)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. Run 1c_co2_prediction_from_outputs.py first."
        )


def to_builtin(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if math.isnan(converted) or math.isinf(converted):
            return None
        return converted
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    return value


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def fit_quadratic_centered_co2(co2, temp, years):
    co2 = np.asarray(co2, dtype=float)
    temp = np.asarray(temp, dtype=float)
    years = np.asarray(years, dtype=int)

    co2_mean = float(np.mean(co2))
    x = co2 - co2_mean
    a, b, c = np.polyfit(x, temp, 2)

    def predict(new_co2):
        new_x = np.asarray(new_co2, dtype=float) - co2_mean
        return a * new_x ** 2 + b * new_x + c

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "C_mean": co2_mean,
        "Fit_Start_Year": int(years[0]),
        "Fit_End_Year": int(years[-1]),
        "predict": predict,
    }


def calculate_group_metrics(group):
    residuals = group["Residual"].to_numpy(dtype=float)
    abs_residuals = np.abs(residuals)
    return pd.Series({
        "RMSE": float(np.sqrt(np.mean(residuals ** 2))),
        "MAE": float(np.mean(abs_residuals)),
        "Mean_Residual": float(np.mean(residuals)),
        "Max_Absolute_Residual": float(np.max(abs_residuals)),
        "Sample_Count": int(len(group)),
    })


def load_future_co2_predictions():
    require_file(FUTURE_CO2_CSV)
    future_df = pd.read_csv(FUTURE_CO2_CSV)

    required_columns = {"Model", "Year", "Forecast_CO2_ppm"}
    missing_columns = required_columns - set(future_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {FUTURE_CO2_CSV}: {sorted(missing_columns)}")

    if "Model_Source" in future_df.columns:
        future_df["CO2_Model"] = future_df["Model"] + " | " + future_df["Model_Source"]
    else:
        future_df["CO2_Model"] = future_df["Model"]

    future_df = future_df[
        (future_df["Year"] >= FORECAST_START_YEAR)
        & (future_df["Year"] <= FORECAST_END_YEAR)
    ].copy()

    future_df = future_df.rename(columns={"Forecast_CO2_ppm": "Predicted_CO2_ppm"})
    return future_df[["Year", "CO2_Model", "Predicted_CO2_ppm"]].sort_values(["CO2_Model", "Year"])


def build_future_temperature_predictions(future_co2_df, full_fit):
    future_df = future_co2_df.copy()
    future_df["Predicted_Temperature_Change_C"] = full_fit["predict"](
        future_df["Predicted_CO2_ppm"].to_numpy(dtype=float)
    )
    return future_df


def run_rolling_validation(years, co2, temp):
    rows = []

    year_to_index = {int(year): i for i, year in enumerate(years)}

    for train_end_year in ROLLING_TRAIN_END_YEARS:
        if train_end_year not in year_to_index:
            raise ValueError(f"Training end year {train_end_year} is not in the aligned data.")

        train_mask = years <= train_end_year
        train_years = years[train_mask]
        train_co2 = co2[train_mask]
        train_temp = temp[train_mask]

        fit = fit_quadratic_centered_co2(train_co2, train_temp, train_years)
        max_year = min(train_end_year + MAX_FORECAST_HORIZON, int(years[-1]))

        for forecast_year in range(train_end_year + 1, max_year + 1):
            idx = year_to_index[forecast_year]
            observed_co2 = float(co2[idx])
            observed_temp = float(temp[idx])
            predicted_temp = float(fit["predict"]([observed_co2])[0])
            residual = observed_temp - predicted_temp

            rows.append({
                "Training_End_Year": int(train_end_year),
                "Forecast_Year": int(forecast_year),
                "Forecast_Horizon": int(forecast_year - train_end_year),
                "Observed_CO2": observed_co2,
                "Observed_Temperature": observed_temp,
                "Predicted_Temperature": predicted_temp,
                "Residual": float(residual),
                "Absolute_Residual": float(abs(residual)),
            })

    residuals_df = pd.DataFrame(rows)
    by_horizon_df = (
        residuals_df
        .groupby("Forecast_Horizon", as_index=False)
        .apply(calculate_group_metrics, include_groups=False)
        .reset_index(drop=True)
    )
    by_horizon_df["Forecast_Horizon"] = by_horizon_df["Forecast_Horizon"].astype(int)
    by_horizon_df["Sample_Count"] = by_horizon_df["Sample_Count"].astype(int)

    return residuals_df, by_horizon_df


def plot_future_temperature(years, temp, future_df):
    plt.figure(figsize=(12, 7))
    plt.plot(
        years,
        temp,
        color="black",
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="Observed temperature change",
    )

    for model_name, model_df in future_df.groupby("CO2_Model", sort=False):
        plt.plot(
            model_df["Year"],
            model_df["Predicted_Temperature_Change_C"],
            linewidth=2,
            label=model_name,
        )

        key_df = model_df[model_df["Year"].isin(KEY_YEARS)]
        plt.scatter(
            key_df["Year"],
            key_df["Predicted_Temperature_Change_C"],
            s=45,
            zorder=5,
        )
        for _, row in key_df.iterrows():
            plt.annotate(
                f"{int(row['Year'])}: {row['Predicted_Temperature_Change_C']:.2f} C",
                (row["Year"], row["Predicted_Temperature_Change_C"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )

    for target in TEMP_CHANGE_TARGETS:
        plt.axhline(
            y=target,
            linestyle="--",
            linewidth=1.2,
            label=f"{target:.2f} C above {BASE_PERIOD_LABEL}",
        )

    plt.axvline(x=years[-1], color="gray", linestyle=":", linewidth=1.2, label="Forecast starts")
    plt.xlabel("Year")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title("Future Temperature Prediction Using Quadratic CO2-Temperature Model")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    save_plot(OUTPUT_DIR / "future_temperature_prediction_quadratic_co2.svg")


def plot_rmse_mae_by_horizon(by_horizon_df):
    plt.figure(figsize=(10, 6))
    plt.plot(
        by_horizon_df["Forecast_Horizon"],
        by_horizon_df["RMSE"],
        marker="o",
        linewidth=2,
        label="RMSE",
    )
    plt.plot(
        by_horizon_df["Forecast_Horizon"],
        by_horizon_df["MAE"],
        marker="o",
        linewidth=2,
        label="MAE",
    )
    plt.xlabel("Forecast horizon / years")
    plt.ylabel("Error / C")
    plt.title("Rolling Validation Error by Forecast Horizon")
    plt.xticks(range(1, MAX_FORECAST_HORIZON + 1))
    plt.grid(True, alpha=0.35)
    plt.legend()
    save_plot(OUTPUT_DIR / "rolling_validation_rmse_mae_by_horizon_quadratic_co2.svg")


def plot_residual_boxplot_by_horizon(residuals_df):
    boxplot_data = [
        residuals_df[residuals_df["Forecast_Horizon"] == horizon]["Residual"].to_numpy(dtype=float)
        for horizon in range(1, MAX_FORECAST_HORIZON + 1)
    ]

    plt.figure(figsize=(12, 6))
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1.2)
    plt.boxplot(
        boxplot_data,
        positions=list(range(1, MAX_FORECAST_HORIZON + 1)),
        showfliers=True,
        patch_artist=True,
        boxprops={"facecolor": "#d9e8f5"},
        medianprops={"color": "black"},
    )
    plt.xlabel("Forecast horizon / years")
    plt.ylabel("Residual / C")
    plt.title("Rolling Validation Residual Distribution by Forecast Horizon")
    plt.xticks(range(1, MAX_FORECAST_HORIZON + 1))
    plt.grid(True, axis="y", alpha=0.35)
    save_plot(OUTPUT_DIR / "rolling_validation_residuals_by_horizon_boxplot_quadratic_co2.svg")


def plot_residuals_over_time(residuals_df):
    plt.figure(figsize=(12, 6))
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1.2, label="Zero residual")

    for train_end_year, group in residuals_df.groupby("Training_End_Year", sort=True):
        plt.plot(
            group["Forecast_Year"],
            group["Residual"],
            marker="o",
            linewidth=1.7,
            label=f"Train <= {train_end_year}",
        )

    plt.xlabel("Forecast year")
    plt.ylabel("Residual / C")
    plt.title("Rolling Validation Residuals Over Time")
    plt.grid(True, alpha=0.35)
    plt.legend()
    save_plot(OUTPUT_DIR / "rolling_validation_residuals_over_time_quadratic_co2.svg")


def plot_mean_residual_by_horizon(by_horizon_df):
    plt.figure(figsize=(10, 6))
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1.2, label="Zero residual")
    plt.plot(
        by_horizon_df["Forecast_Horizon"],
        by_horizon_df["Mean_Residual"],
        marker="o",
        linewidth=2,
        label="Mean residual",
    )
    plt.xlabel("Forecast horizon / years")
    plt.ylabel("Mean residual / C")
    plt.title("Rolling Validation Mean Residual by Forecast Horizon")
    plt.xticks(range(1, MAX_FORECAST_HORIZON + 1))
    plt.grid(True, alpha=0.35)
    plt.legend()
    save_plot(OUTPUT_DIR / "rolling_validation_mean_residual_by_horizon_quadratic_co2.svg")


def print_summary(full_fit, key_years_df, by_horizon_df):
    print("=" * 80)
    print("QUESTION 2C: QUADRATIC CENTERED CO2 RELIABILITY AND FUTURE TEMPERATURE")
    print("=" * 80)
    print("\nFull-data quadratic centered CO2 model:")
    print("T = a * (C - C_mean)^2 + b * (C - C_mean) + c")
    print(f"a      = {full_fit['a']:.12f}")
    print(f"b      = {full_fit['b']:.12f}")
    print(f"c      = {full_fit['c']:.12f}")
    print(f"C_mean = {full_fit['C_mean']:.6f} ppm")
    print(f"Fit years: {full_fit['Fit_Start_Year']} to {full_fit['Fit_End_Year']}")

    print("\nKey future years:")
    print(key_years_df.to_string(index=False))

    print("\nRolling validation by forecast horizon:")
    printable_cols = ["Forecast_Horizon", "RMSE", "MAE", "Mean_Residual", "Sample_Count"]
    print(by_horizon_df[printable_cols].to_string(index=False))

    early_df = by_horizon_df[by_horizon_df["Forecast_Horizon"] <= 5]
    late_df = by_horizon_df[by_horizon_df["Forecast_Horizon"] >= 15]
    print("\nDiagnostic notes, without choosing a reliability cutoff:")
    print(
        f"- Horizons 1-5: mean RMSE = {early_df['RMSE'].mean():.4f} C, "
        f"mean MAE = {early_df['MAE'].mean():.4f} C."
    )
    print(
        f"- Horizons 15-20: mean RMSE = {late_df['RMSE'].mean():.4f} C, "
        f"mean MAE = {late_df['MAE'].mean():.4f} C."
    )
    largest_rmse_row = by_horizon_df.loc[by_horizon_df["RMSE"].idxmax()]
    largest_bias_row = by_horizon_df.loc[by_horizon_df["Mean_Residual"].abs().idxmax()]
    print(
        f"- Largest horizon-level RMSE appears at h={int(largest_rmse_row['Forecast_Horizon'])}: "
        f"{largest_rmse_row['RMSE']:.4f} C."
    )
    print(
        f"- Largest absolute mean residual appears at h={int(largest_bias_row['Forecast_Horizon'])}: "
        f"{largest_bias_row['Mean_Residual']:.4f} C."
    )


def main():
    reset_output_dir(OUTPUT_DIR)

    years, co2, temp = get_aligned_co2_temperature_data()
    full_fit = fit_quadratic_centered_co2(co2, temp, years)

    future_co2_df = load_future_co2_predictions()
    future_temp_df = build_future_temperature_predictions(future_co2_df, full_fit)
    key_years_df = future_temp_df[future_temp_df["Year"].isin(KEY_YEARS)].copy()

    residuals_df, by_horizon_df = run_rolling_validation(years, co2, temp)

    parameters_df = pd.DataFrame([{
        "a": full_fit["a"],
        "b": full_fit["b"],
        "c": full_fit["c"],
        "C_mean": full_fit["C_mean"],
        "Fit_Start_Year": full_fit["Fit_Start_Year"],
        "Fit_End_Year": full_fit["Fit_End_Year"],
    }])

    future_temp_df.to_csv(
        OUTPUT_DIR / "future_temperature_predictions_quadratic_co2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    key_years_df.to_csv(
        OUTPUT_DIR / "future_temperature_key_years_quadratic_co2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_horizon_df.to_csv(
        OUTPUT_DIR / "rolling_validation_by_horizon_quadratic_co2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residuals_df.to_csv(
        OUTPUT_DIR / "rolling_validation_all_residuals_quadratic_co2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    parameters_df.to_csv(
        OUTPUT_DIR / "quadratic_co2_model_parameters_full_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "model": "Quadratic centered CO2-temperature relationship",
        "formula": "T = a * (C - C_mean)^2 + b * (C - C_mean) + c",
        "base_period": BASE_PERIOD_LABEL,
        "future_co2_source": str(FUTURE_CO2_CSV),
        "future_year_range": [FORECAST_START_YEAR, FORECAST_END_YEAR],
        "rolling_train_end_years": ROLLING_TRAIN_END_YEARS,
        "max_forecast_horizon": MAX_FORECAST_HORIZON,
        "outputs": [
            "future_temperature_predictions_quadratic_co2.csv",
            "future_temperature_key_years_quadratic_co2.csv",
            "rolling_validation_by_horizon_quadratic_co2.csv",
            "rolling_validation_all_residuals_quadratic_co2.csv",
            "quadratic_co2_model_parameters_full_data.csv",
            "future_temperature_prediction_quadratic_co2.svg",
            "rolling_validation_rmse_mae_by_horizon_quadratic_co2.svg",
            "rolling_validation_residuals_by_horizon_boxplot_quadratic_co2.svg",
            "rolling_validation_residuals_over_time_quadratic_co2.svg",
            "rolling_validation_mean_residual_by_horizon_quadratic_co2.svg",
        ],
    }
    (OUTPUT_DIR / "question_2c_manifest.json").write_text(
        json.dumps(to_builtin(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    plot_future_temperature(years, temp, future_temp_df)
    plot_rmse_mae_by_horizon(by_horizon_df)
    plot_residual_boxplot_by_horizon(residuals_df)
    plot_residuals_over_time(residuals_df)
    plot_mean_residual_by_horizon(by_horizon_df)

    print_summary(full_fit, key_years_df, by_horizon_df)
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
