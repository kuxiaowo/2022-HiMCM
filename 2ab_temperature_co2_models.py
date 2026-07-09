# question_2ab_temperature_co2_models.py
# 2a: Model future global land-ocean temperature change from the 1951-1980 base period.
# 2b: Model the relationship between atmospheric CO2 and land-ocean temperature change.
#
# Data are loaded from himcm_data.py.

from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from himcm_data import (
    get_aligned_co2_temperature_data,
    get_co2_data,
    get_temperature_data,
)


# ============================================================
# 0. Settings
# ============================================================

OUTPUT_DIR = Path("question_2_outputs")
TEMP_OUTPUT_DIR = OUTPUT_DIR / "2a_temperature_forecast"
REL_OUTPUT_DIR = OUTPUT_DIR / "2b_co2_temperature_relationship"

TEST_YEARS_COUNT = 10
FUTURE_END_YEAR = 2200
PLOT_END_YEAR = 2100
BASE_PERIOD_LABEL = "1951-1980"
TEMP_CHANGE_TARGETS = [1.25, 1.50, 2.00]


TEMP_TIME_MODELS = [
    {"name": "Quadratic time trend", "key": "quadratic_time", "type": "polynomial_time", "degree": 2},
    {"name": "Quartic time trend", "key": "quartic_time", "type": "polynomial_time", "degree": 4},
    {"name": "Shifted exponential time trend", "key": "shifted_exp_time", "type": "shifted_exp_time"},
]


RELATIONSHIP_MODELS = [
    {"name": "Linear CO2", "key": "linear_co2", "type": "linear_co2"},
    {"name": "Quadratic centered CO2", "key": "quadratic_centered_co2", "type": "polynomial_centered_co2", "degree": 2},
    {"name": "Radiative forcing proxy log2(CO2/280)", "key": "log2_co2_280", "type": "log2_co2_280"},
]


# ============================================================
# 1. General helpers
# ============================================================

def reset_output_dir(output_dir):
    script_dir = Path(__file__).resolve().parent
    resolved_output_dir = (script_dir / output_dir).resolve()

    if script_dir not in resolved_output_dir.parents:
        raise RuntimeError(f"Refusing to clean outside script directory: {resolved_output_dir}")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    for child in resolved_output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def safe_filename(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
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


def evaluate_model(y_true, y_pred, parameter_count):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    n = len(y_true)
    sse = float(np.sum(residuals ** 2))
    mse = float(sse / n)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    mean_error = float(np.mean(residuals))
    max_abs_error = float(np.max(np.abs(residuals)))

    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - sse / sst) if sst != 0 else np.nan

    if n > parameter_count and not np.isnan(r2):
        adjusted_r2 = float(1 - (1 - r2) * (n - 1) / (n - parameter_count))
    else:
        adjusted_r2 = np.nan

    if sse > 0:
        aic = float(n * np.log(sse / n) + 2 * parameter_count)
        bic = float(n * np.log(sse / n) + parameter_count * np.log(n))
    else:
        aic = -np.inf
        bic = -np.inf

    return {
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Mean_Error": mean_error,
        "Max_Absolute_Error": max_abs_error,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
        "AIC": aic,
        "BIC": bic,
    }


def prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def coefficient_labels(count):
    return [chr(ord("a") + index) for index in range(count)]


# ============================================================
# 2a. Temperature time models
# ============================================================

def fit_temperature_time_model(spec, years, temp):
    base_year = int(years[0])
    years = np.asarray(years, dtype=float)
    temp = np.asarray(temp, dtype=float)

    fit_mask = np.ones(len(years), dtype=bool)

    if spec["type"] == "recent_linear":
        fit_mask = years >= spec["fit_start_year"]

    fit_years = years[fit_mask]
    fit_temp = temp[fit_mask]
    fit_t = fit_years - base_year

    if spec["type"] == "polynomial_time":
        degree = spec["degree"]
        coefficients = np.polyfit(fit_t, fit_temp, degree)
        parameters = {
            label: float(value)
            for label, value in zip(coefficient_labels(len(coefficients)), coefficients)
        }

        def predict(new_years):
            new_t = np.asarray(new_years, dtype=float) - base_year
            return np.polyval(coefficients, new_t)

        formula = (
            f"Temperature change from {BASE_PERIOD_LABEL} base period = "
            f"polynomial degree {degree} in t, t = year - {base_year}"
        )
        parameter_count = degree + 1

    elif spec["type"] == "log_time":
        x = np.log1p(fit_t)
        coefficients = np.polyfit(x, fit_temp, 1)
        parameters = {"a": float(coefficients[0]), "b": float(coefficients[1])}

        def predict(new_years):
            new_t = np.asarray(new_years, dtype=float) - base_year
            return coefficients[0] * np.log1p(new_t) + coefficients[1]

        formula = f"Temperature change from {BASE_PERIOD_LABEL} base period = a * ln(1 + t) + b, t = year - {base_year}"
        parameter_count = 2

    elif spec["type"] == "sqrt_time":
        x = np.sqrt(fit_t)
        coefficients = np.polyfit(x, fit_temp, 1)
        parameters = {"a": float(coefficients[0]), "b": float(coefficients[1])}

        def predict(new_years):
            new_t = np.asarray(new_years, dtype=float) - base_year
            return coefficients[0] * np.sqrt(new_t) + coefficients[1]

        formula = f"Temperature change from {BASE_PERIOD_LABEL} base period = a * sqrt(t) + b, t = year - {base_year}"
        parameter_count = 2

    elif spec["type"] == "shifted_exp_time":
        shift = max(0.1, -float(np.min(fit_temp)) + 0.1)
        shifted_temp = fit_temp + shift
        coefficients = np.polyfit(fit_t, np.log(shifted_temp), 1)
        b_value = float(coefficients[0])
        a_value = float(np.exp(coefficients[1]))
        parameters = {"A": a_value, "B": b_value, "shift": float(shift)}

        def predict(new_years):
            new_t = np.asarray(new_years, dtype=float) - base_year
            return a_value * np.exp(b_value * new_t) - shift

        formula = f"Temperature change from {BASE_PERIOD_LABEL} base period = A * exp(B * t) - shift, t = year - {base_year}"
        parameter_count = 3

    elif spec["type"] == "recent_linear":
        coefficients = np.polyfit(fit_t, fit_temp, 1)
        parameters = {
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "fit_start_year": int(spec["fit_start_year"]),
        }

        def predict(new_years):
            new_t = np.asarray(new_years, dtype=float) - base_year
            return coefficients[0] * new_t + coefficients[1]

        formula = (
            f"Temperature change from {BASE_PERIOD_LABEL} base period = a * t + b, t = year - {base_year}; "
            f"fit uses years >= {spec['fit_start_year']}"
        )
        parameter_count = 2

    else:
        raise ValueError(f"Unsupported temperature model type: {spec['type']}")

    return {
        "predict": predict,
        "parameters": parameters,
        "formula": formula,
        "parameter_count": parameter_count,
        "fit_start_year": int(fit_years[0]),
        "fit_end_year": int(fit_years[-1]),
    }


def estimate_change_target_from_forecast(forecast_df, target_change):
    target_rows = forecast_df[
        forecast_df["Forecast_Temperature_Change_From_1951_1980_C"] >= target_change
    ]

    if len(target_rows) == 0:
        return None, None, None

    first_row = target_rows.iloc[0]
    return int(first_row["Year"]), float(first_row["Forecast_Temperature_Change_From_1951_1980_C"])


def plot_temperature_raw(years, temp):
    plt.figure(figsize=(10, 6))
    plt.scatter(years, temp, s=28, color="black", label=f"Observed temperature change from {BASE_PERIOD_LABEL}")
    plt.plot(years, temp, color="black", linewidth=1)
    for target_change in TEMP_CHANGE_TARGETS:
        plt.axhline(
            y=target_change,
            linestyle="--",
            linewidth=1.2,
            label=f"{target_change:.2f} C above {BASE_PERIOD_LABEL}",
        )
    plt.xlabel("Year")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title(f"Observed Temperature Change Relative to {BASE_PERIOD_LABEL}")
    plt.grid(True)
    plt.legend()
    save_plot(TEMP_OUTPUT_DIR / "plots" / "temperature_change_observed.svg")


def plot_temperature_model_fit(
    spec,
    years,
    temp,
    years_train,
    years_test,
    full_fit,
    train_fit,
):
    model_dir = TEMP_OUTPUT_DIR / "models" / spec["key"]
    forecast_years = np.arange(int(years[0]), PLOT_END_YEAR + 1)

    plt.figure(figsize=(12, 6))
    plt.scatter(years, temp, s=26, color="black", label=f"Observed temperature change from {BASE_PERIOD_LABEL}")
    plt.plot(forecast_years, full_fit["predict"](forecast_years), linewidth=2, label="Full-data fit and forecast")
    plt.plot(forecast_years, train_fit["predict"](forecast_years), linewidth=2, linestyle="--", label="Training-only fit and forecast")
    plt.axvline(x=years_test[0], color="gray", linestyle="--", label="Start of test period")
    for target_change in TEMP_CHANGE_TARGETS:
        plt.axhline(y=target_change, linestyle=":", linewidth=1.2, label=f"{target_change:.2f} C above {BASE_PERIOD_LABEL}")
    plt.xlabel("Year")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title(f"{spec['name']}: Temperature Fit and Forecast")
    plt.grid(True)
    plt.legend()
    save_plot(model_dir / f"{spec['key']}_fit_forecast.svg")


def plot_temperature_residuals(spec, years_values, residuals, label):
    model_dir = TEMP_OUTPUT_DIR / "models" / spec["key"]
    plt.figure(figsize=(10, 5))
    plt.axhline(y=0, color="gray", linestyle="--", label="Zero residual")
    plt.plot(years_values, residuals, marker="o", linewidth=1.5, markersize=4, label="Residuals")
    plt.xlabel("Year")
    plt.ylabel("Residual / C")
    plt.title(f"{spec['name']}: {label} Residuals")
    plt.grid(True)
    plt.legend()
    save_plot(model_dir / f"{spec['key']}_{label}_residuals.svg")


def plot_temperature_all_models(years, temp, forecast_df):
    plt.figure(figsize=(13, 7))
    plt.scatter(years, temp, s=26, color="black", label=f"Observed temperature change from {BASE_PERIOD_LABEL}")
    plt.plot(years, temp, color="black", linewidth=1)

    plot_forecasts = forecast_df[
        (forecast_df["Fit_Stage"] == "full_data")
        & (forecast_df["Year"] <= PLOT_END_YEAR)
    ]

    for model_name, model_df in plot_forecasts.groupby("Model", sort=False):
        plt.plot(model_df["Year"], model_df["Forecast_Temperature_Change_From_1951_1980_C"], linewidth=1.8, label=model_name)

    for target_change in TEMP_CHANGE_TARGETS:
        plt.axhline(
            y=target_change,
            linestyle="--",
            linewidth=1.2,
            label=f"{target_change:.2f} C above {BASE_PERIOD_LABEL}",
        )

    plt.xlabel("Year")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title(f"Forecasts of Temperature Change Relative to {BASE_PERIOD_LABEL}")
    plt.grid(True)
    plt.legend(fontsize=8)
    save_plot(TEMP_OUTPUT_DIR / "plots" / "temperature_all_model_forecasts.svg")


def plot_temperature_all_models_zoomed(years, temp, forecast_df):
    plt.figure(figsize=(13, 7))
    plt.scatter(years, temp, s=26, color="black", label=f"Observed temperature change from {BASE_PERIOD_LABEL}")
    plt.plot(years, temp, color="black", linewidth=1)

    plot_forecasts = forecast_df[
        (forecast_df["Fit_Stage"] == "full_data")
        & (forecast_df["Year"] <= PLOT_END_YEAR)
    ]

    for model_name, model_df in plot_forecasts.groupby("Model", sort=False):
        plt.plot(model_df["Year"], model_df["Forecast_Temperature_Change_From_1951_1980_C"], linewidth=1.8, label=model_name)

    for target_change in TEMP_CHANGE_TARGETS:
        plt.axhline(
            y=target_change,
            linestyle="--",
            linewidth=1.2,
            label=f"{target_change:.2f} C above {BASE_PERIOD_LABEL}",
        )

    plt.ylim(-0.25, 4.0)
    plt.xlabel("Year")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title(f"Forecasts of Temperature Change Relative to {BASE_PERIOD_LABEL} (Zoomed)")
    plt.grid(True)
    plt.legend(fontsize=8)
    save_plot(TEMP_OUTPUT_DIR / "plots" / "temperature_all_model_forecasts_zoomed.svg")


def plot_temperature_error_comparison(summary_df, metric_prefix, filename, title):
    sorted_df = summary_df.sort_values(f"{metric_prefix}_RMSE")
    x = np.arange(len(sorted_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, sorted_df[f"{metric_prefix}_RMSE"], width, label=f"{metric_prefix} RMSE")
    plt.bar(x + width / 2, sorted_df[f"{metric_prefix}_MAE"], width, label=f"{metric_prefix} MAE")
    plt.xticks(x, sorted_df["Model"], rotation=25, ha="right")
    plt.ylabel("Error / C")
    plt.title(title)
    plt.grid(axis="y")
    plt.legend()
    save_plot(TEMP_OUTPUT_DIR / "plots" / filename)


def plot_temperature_change_targets(target_df):
    plot_df = target_df[
        (target_df["Fit_Stage"] == "full_data")
        & target_df["First_Year_Predicted_Change_At_Least_Target"].notna()
    ].copy()

    if plot_df.empty:
        return

    pivot_df = plot_df.pivot(
        index="Model",
        columns="Target_Temperature_Change_From_1951_1980_C",
        values="First_Year_Predicted_Change_At_Least_Target",
    )
    pivot_df = pivot_df.sort_values(TEMP_CHANGE_TARGETS[0])

    plt.figure(figsize=(12, 6))
    for target_change in TEMP_CHANGE_TARGETS:
        if target_change in pivot_df.columns:
            plt.plot(
                pivot_df.index,
                pivot_df[target_change],
                marker="o",
                linewidth=2,
                label=f"{target_change:.2f} C change",
            )

    plt.xticks(rotation=25, ha="right")
    plt.ylabel("First predicted year")
    plt.title(f"Predicted Years When Temperature Change Exceeds {BASE_PERIOD_LABEL} Targets")
    plt.grid(True)
    plt.legend()
    save_plot(TEMP_OUTPUT_DIR / "plots" / "temperature_change_target_years_comparison.svg")


def run_temperature_forecast_models():
    years, temp = get_temperature_data()
    test_start_index = len(years) - TEST_YEARS_COUNT

    years_train = years[:test_start_index]
    years_test = years[test_start_index:]
    temp_train = temp[:test_start_index]
    temp_test = temp[test_start_index:]

    future_years = np.arange(int(years[-1]) + 1, FUTURE_END_YEAR + 1)

    summary_rows = []
    parameter_rows = []
    fitted_rows = []
    forecast_rows = []
    target_rows = []

    plot_temperature_raw(years, temp)

    raw_df = pd.DataFrame({
        "Year": years,
        "Temperature_Change_From_1951_1980_C": temp,
    })
    raw_df.to_csv(TEMP_OUTPUT_DIR / "temperature_observed_data.csv", index=False, encoding="utf-8-sig")

    for spec in TEMP_TIME_MODELS:
        full_fit = fit_temperature_time_model(spec, years, temp)
        train_fit = fit_temperature_time_model(spec, years_train, temp_train)

        full_pred = full_fit["predict"](years)
        train_pred = train_fit["predict"](years_train)
        test_pred = train_fit["predict"](years_test)

        full_metrics = evaluate_model(temp, full_pred, full_fit["parameter_count"])
        train_metrics = evaluate_model(temp_train, train_pred, train_fit["parameter_count"])
        test_metrics = evaluate_model(temp_test, test_pred, train_fit["parameter_count"])

        summary_row = {
            "Model": spec["name"],
            "Model_Key": spec["key"],
            "Model_Type": spec["type"],
            "Full_Formula": full_fit["formula"],
            "Train_Formula": train_fit["formula"],
        }
        summary_row.update(prefix_metrics("Full", full_metrics))
        summary_row.update(prefix_metrics("Train", train_metrics))
        summary_row.update(prefix_metrics("Test", test_metrics))
        summary_rows.append(summary_row)

        for fit_stage, fit_result in [("full_data", full_fit), ("training_data", train_fit)]:
            parameter_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Model_Type": spec["type"],
                "Fit_Stage": fit_stage,
                "Formula": fit_result["formula"],
                "Parameters_JSON": json.dumps(fit_result["parameters"]),
                "Parameter_Count": fit_result["parameter_count"],
                "Fit_Start_Year": fit_result["fit_start_year"],
                "Fit_End_Year": fit_result["fit_end_year"],
            })

        for year, observed, predicted in zip(years, temp, full_pred):
            residual = observed - predicted
            fitted_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Fit_Stage": "full_data",
                "Year": int(year),
                "Observed_Temperature_Change_From_1951_1980_C": float(observed),
                "Fitted_Temperature_Change_From_1951_1980_C": float(predicted),
                "Residual_C": float(residual),
                "Absolute_Residual_C": float(abs(residual)),
            })

        for year, observed, predicted in zip(years_train, temp_train, train_pred):
            residual = observed - predicted
            fitted_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Fit_Stage": "training_data",
                "Year": int(year),
                "Observed_Temperature_Change_From_1951_1980_C": float(observed),
                "Fitted_Temperature_Change_From_1951_1980_C": float(predicted),
                "Residual_C": float(residual),
                "Absolute_Residual_C": float(abs(residual)),
            })

        for year, observed, predicted in zip(years_test, temp_test, test_pred):
            residual = observed - predicted
            fitted_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Fit_Stage": "test_data",
                "Year": int(year),
                "Observed_Temperature_Change_From_1951_1980_C": float(observed),
                "Fitted_Temperature_Change_From_1951_1980_C": float(predicted),
                "Residual_C": float(residual),
                "Absolute_Residual_C": float(abs(residual)),
            })

        for fit_stage, fit_result in [("full_data", full_fit), ("training_data", train_fit)]:
            future_prediction = fit_result["predict"](future_years)
            model_forecast_rows = []

            for year, value in zip(future_years, future_prediction):
                forecast_row = {
                    "Model": spec["name"],
                    "Model_Key": spec["key"],
                    "Fit_Stage": fit_stage,
                    "Year": int(year),
                    "Forecast_Temperature_Change_From_1951_1980_C": float(value),
                }
                forecast_rows.append(forecast_row)
                model_forecast_rows.append(forecast_row)

            model_forecast_df = pd.DataFrame(model_forecast_rows)

            for target_change in TEMP_CHANGE_TARGETS:
                first_year, first_value = estimate_change_target_from_forecast(model_forecast_df, target_change)
                target_rows.append({
                    "Model": spec["name"],
                    "Model_Key": spec["key"],
                    "Fit_Stage": fit_stage,
                    "Target_Temperature_Change_From_1951_1980_C": target_change,
                    "First_Year_Predicted_Change_At_Least_Target": first_year,
                    "Forecast_Change_At_First_Year_C": first_value,
                })

    summary_df = pd.DataFrame(summary_rows).sort_values("Test_RMSE").reset_index(drop=True)
    parameter_df = pd.DataFrame(parameter_rows)
    fitted_df = pd.DataFrame(fitted_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    target_df = pd.DataFrame(target_rows)

    summary_df.to_csv(TEMP_OUTPUT_DIR / "temperature_model_summary.csv", index=False, encoding="utf-8-sig")
    parameter_df.to_csv(TEMP_OUTPUT_DIR / "temperature_model_parameters.csv", index=False, encoding="utf-8-sig")
    fitted_df.to_csv(TEMP_OUTPUT_DIR / "temperature_fitted_values_and_residuals.csv", index=False, encoding="utf-8-sig")
    forecast_df.to_csv(TEMP_OUTPUT_DIR / "temperature_future_forecasts.csv", index=False, encoding="utf-8-sig")
    target_df.to_csv(TEMP_OUTPUT_DIR / "temperature_change_target_years.csv", index=False, encoding="utf-8-sig")

    temperature_payload = {
        "observed_data": raw_df.to_dict(orient="records"),
        "model_summary": summary_df.to_dict(orient="records"),
        "model_parameters": parameter_df.to_dict(orient="records"),
        "fitted_values_and_residuals": fitted_df.to_dict(orient="records"),
        "future_forecasts": forecast_df.to_dict(orient="records"),
        "temperature_change_target_years": target_df.to_dict(orient="records"),
    }
    (TEMP_OUTPUT_DIR / "temperature_results.json").write_text(
        json.dumps(to_builtin(temperature_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    for spec in TEMP_TIME_MODELS:
        model_dir = TEMP_OUTPUT_DIR / "models" / spec["key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        summary_df[summary_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_summary.csv", index=False, encoding="utf-8-sig"
        )
        parameter_df[parameter_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_parameters.csv", index=False, encoding="utf-8-sig"
        )
        fitted_df[fitted_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_fitted_residuals.csv", index=False, encoding="utf-8-sig"
        )
        forecast_df[forecast_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_future_forecast.csv", index=False, encoding="utf-8-sig"
        )
        target_df[target_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_temperature_change_target_years.csv", index=False, encoding="utf-8-sig"
        )

    plot_temperature_all_models(years, temp, forecast_df)
    plot_temperature_all_models_zoomed(years, temp, forecast_df)
    plot_temperature_error_comparison(summary_df, "Train", "temperature_training_error_comparison.svg", "Temperature Model Training Error Comparison")
    plot_temperature_error_comparison(summary_df, "Test", "temperature_test_error_comparison.svg", "Temperature Model Test Error Comparison")
    plot_temperature_change_targets(target_df)

    return {
        "summary": summary_df,
        "parameters": parameter_df,
        "temperature_change_targets": target_df,
    }


# ============================================================
# 2b. CO2-temperature relationship models
# ============================================================

def fit_relationship_model(spec, co2, temp):
    co2 = np.asarray(co2, dtype=float)
    temp = np.asarray(temp, dtype=float)

    if spec["type"] == "linear_co2":
        x = co2
        coefficients = np.polyfit(x, temp, 1)
        parameters = {"a": float(coefficients[0]), "b": float(coefficients[1])}

        def predict(new_co2):
            return coefficients[0] * np.asarray(new_co2, dtype=float) + coefficients[1]

        formula = f"Temperature change from {BASE_PERIOD_LABEL} base period = a * CO2 + b"
        parameter_count = 2

    elif spec["type"] == "polynomial_centered_co2":
        degree = spec["degree"]
        center = float(np.mean(co2))
        x = co2 - center
        coefficients = np.polyfit(x, temp, degree)
        parameters = {
            label: float(value)
            for label, value in zip(coefficient_labels(len(coefficients)), coefficients)
        }
        parameters["co2_center"] = center

        def predict(new_co2):
            return np.polyval(coefficients, np.asarray(new_co2, dtype=float) - center)

        formula = (
            f"Temperature change from {BASE_PERIOD_LABEL} base period = "
            f"polynomial degree {degree} in (CO2 - {center:.4f})"
        )
        parameter_count = degree + 1

    elif spec["type"] == "log_co2":
        x = np.log(co2)
        coefficients = np.polyfit(x, temp, 1)
        parameters = {"a": float(coefficients[0]), "b": float(coefficients[1])}

        def predict(new_co2):
            return coefficients[0] * np.log(np.asarray(new_co2, dtype=float)) + coefficients[1]

        formula = f"Temperature change from {BASE_PERIOD_LABEL} base period = a * ln(CO2) + b"
        parameter_count = 2

    elif spec["type"] == "log2_co2_280":
        x = np.log2(co2 / 280.0)
        coefficients = np.polyfit(x, temp, 1)
        parameters = {"a": float(coefficients[0]), "b": float(coefficients[1])}

        def predict(new_co2):
            return coefficients[0] * np.log2(np.asarray(new_co2, dtype=float) / 280.0) + coefficients[1]

        formula = f"Temperature change from {BASE_PERIOD_LABEL} base period = a * log2(CO2 / 280) + b"
        parameter_count = 2

    else:
        raise ValueError(f"Unsupported relationship model type: {spec['type']}")

    return {
        "predict": predict,
        "parameters": parameters,
        "formula": formula,
        "parameter_count": parameter_count,
    }


def plot_relationship_raw(years, co2, temp):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(years, co2, color="green", marker="o", markersize=3, label="Annual CO2")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("CO2 concentration / ppm", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    ax2 = ax1.twinx()
    ax2.plot(years, temp, color="red", marker="o", markersize=3, label=f"Temperature change from {BASE_PERIOD_LABEL}")
    ax2.set_ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"Aligned CO2 and Temperature Change from {BASE_PERIOD_LABEL} Over Time")
    fig.tight_layout()
    plt.savefig(REL_OUTPUT_DIR / "plots" / "aligned_co2_temperature_time_series.svg", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(co2, temp, s=35, color="black")
    plt.xlabel("CO2 concentration / ppm")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title(f"CO2 Concentration vs Temperature Change from {BASE_PERIOD_LABEL}")
    plt.grid(True)
    save_plot(REL_OUTPUT_DIR / "plots" / "co2_temperature_scatter.svg")


def plot_relationship_fits(co2, temp, fitted_models):
    co2_grid = np.linspace(min(co2), max(co2), 300)

    plt.figure(figsize=(11, 7))
    plt.scatter(co2, temp, s=35, color="black", label="Observed data")

    for model_name, fit_result in fitted_models:
        plt.plot(co2_grid, fit_result["predict"](co2_grid), linewidth=2, label=model_name)

    plt.xlabel("CO2 concentration / ppm")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title(f"CO2-Temperature Change Relationship Models")
    plt.grid(True)
    plt.legend(fontsize=8)
    save_plot(REL_OUTPUT_DIR / "plots" / "co2_temperature_relationship_fits.svg")


def plot_single_relationship_fit(spec, co2, temp, full_fit, train_fit):
    model_dir = REL_OUTPUT_DIR / "models" / spec["key"]
    model_dir.mkdir(parents=True, exist_ok=True)

    co2_grid = np.linspace(min(co2), max(co2), 300)

    plt.figure(figsize=(10, 6))
    plt.scatter(co2, temp, s=35, color="black", label="Observed data")
    plt.plot(co2_grid, full_fit["predict"](co2_grid), linewidth=2, label="Full-data fit")
    plt.plot(co2_grid, train_fit["predict"](co2_grid), linewidth=2, linestyle="--", label="Training-only fit")
    plt.xlabel("CO2 concentration / ppm")
    plt.ylabel(f"Temperature change from {BASE_PERIOD_LABEL} / C")
    plt.title(f"{spec['name']}: CO2-Temperature Fit")
    plt.grid(True)
    plt.legend()
    save_plot(model_dir / f"{spec['key']}_relationship_fit.svg")


def plot_relationship_model_residuals(spec, years, residuals, stage):
    model_dir = REL_OUTPUT_DIR / "models" / spec["key"]
    model_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.axhline(y=0, color="gray", linestyle="--", label="Zero residual")
    plt.plot(years, residuals, marker="o", linewidth=1.5, markersize=4, label="Residuals")
    plt.xlabel("Year")
    plt.ylabel("Residual / C")
    plt.title(f"{spec['name']}: {stage} Residuals")
    plt.grid(True)
    plt.legend()
    save_plot(model_dir / f"{spec['key']}_{stage}_residuals.svg")


def plot_relationship_error_comparison(summary_df, prefix, filename, title):
    sorted_df = summary_df.sort_values(f"{prefix}_RMSE")
    x = np.arange(len(sorted_df))
    width = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, sorted_df[f"{prefix}_RMSE"], width, label=f"{prefix} RMSE")
    plt.bar(x + width / 2, sorted_df[f"{prefix}_MAE"], width, label=f"{prefix} MAE")
    plt.xticks(x, sorted_df["Model"], rotation=25, ha="right")
    plt.ylabel("Error / C")
    plt.title(title)
    plt.grid(axis="y")
    plt.legend()
    save_plot(REL_OUTPUT_DIR / "plots" / filename)


def plot_relationship_test_residuals(fitted_df):
    test_df = fitted_df[fitted_df["Fit_Stage"] == "test_data"].copy()

    plt.figure(figsize=(12, 6))
    plt.axhline(y=0, color="gray", linestyle="--", label="Zero residual")

    for model_name, model_df in test_df.groupby("Model", sort=False):
        plt.plot(model_df["Year"], model_df["Residual_C"], marker="o", linewidth=1.7, label=model_name)

    plt.xlabel("Year")
    plt.ylabel("Residual / C")
    plt.title("CO2-Temperature Relationship Model Test Residuals")
    plt.grid(True)
    plt.legend(fontsize=8)
    save_plot(REL_OUTPUT_DIR / "plots" / "co2_temperature_test_residual_comparison.svg")


def run_lag_analysis(years, co2, temp):
    rows = []

    for lag in range(0, 11):
        lagged_years = years[lag:]
        lagged_co2 = co2[:-lag] if lag > 0 else co2
        lagged_temp = temp[lag:]

        pearson_r = float(pd.Series(lagged_co2).corr(pd.Series(lagged_temp), method="pearson"))
        spearman_r = float(pd.Series(lagged_co2).corr(pd.Series(lagged_temp), method="spearman"))
        fit = np.polyfit(lagged_co2, lagged_temp, 1)
        pred = np.polyval(fit, lagged_co2)
        metrics = evaluate_model(lagged_temp, pred, 2)

        rows.append({
            "Lag_Years": lag,
            "Meaning": f"Temperature year y vs CO2 year y-{lag}",
            "Observations": len(lagged_years),
            "Pearson_R": pearson_r,
            "Spearman_R": spearman_r,
            "Linear_R2": metrics["R2"],
            "Linear_RMSE": metrics["RMSE"],
            "Linear_Slope_C_per_ppm": float(fit[0]),
            "Linear_Intercept_C": float(fit[1]),
        })

    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(REL_OUTPUT_DIR / "co2_temperature_lag_correlation.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(lag_df["Lag_Years"], lag_df["Pearson_R"], marker="o", linewidth=2, label="Pearson r")
    plt.plot(lag_df["Lag_Years"], lag_df["Spearman_R"], marker="o", linewidth=2, label="Spearman r")
    plt.xlabel("CO2 lag / years")
    plt.ylabel("Correlation")
    plt.title("Lagged CO2-Temperature Correlation")
    plt.grid(True)
    plt.legend()
    save_plot(REL_OUTPUT_DIR / "plots" / "co2_temperature_lag_correlation.svg")

    return lag_df


def run_difference_analysis(years, co2, temp):
    delta_years = years[1:]
    delta_co2 = np.diff(co2)
    delta_temp = np.diff(temp)
    coefficients = np.polyfit(delta_co2, delta_temp, 1)
    predicted = np.polyval(coefficients, delta_co2)
    metrics = evaluate_model(delta_temp, predicted, 2)

    diff_df = pd.DataFrame({
        "Year": delta_years,
        "Delta_CO2_ppm": delta_co2,
        "Delta_Temperature_C": delta_temp,
        "Predicted_Delta_Temperature_C": predicted,
        "Residual_C": delta_temp - predicted,
    })
    diff_df.to_csv(REL_OUTPUT_DIR / "co2_temperature_first_difference_relationship.csv", index=False, encoding="utf-8-sig")

    summary = {
        "Model": "First difference linear relationship",
        "Formula": "Delta temperature = a * Delta CO2 + b",
        "Slope_C_per_ppm": float(coefficients[0]),
        "Intercept_C": float(coefficients[1]),
        "Pearson_R": float(pd.Series(delta_co2).corr(pd.Series(delta_temp), method="pearson")),
        **metrics,
    }

    (REL_OUTPUT_DIR / "co2_temperature_first_difference_summary.json").write_text(
        json.dumps(to_builtin(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    x_grid = np.linspace(min(delta_co2), max(delta_co2), 100)

    plt.figure(figsize=(10, 6))
    plt.scatter(delta_co2, delta_temp, s=35, color="black", label="Annual first differences")
    plt.plot(x_grid, np.polyval(coefficients, x_grid), color="red", linewidth=2, label="Linear fit")
    plt.xlabel("Annual CO2 change / ppm")
    plt.ylabel(f"Annual temperature change from {BASE_PERIOD_LABEL} change / C")
    plt.title("First-Difference CO2-Temperature Relationship")
    plt.grid(True)
    plt.legend()
    save_plot(REL_OUTPUT_DIR / "plots" / "co2_temperature_first_difference_scatter.svg")

    return summary


def run_relationship_models():
    years, co2, temp = get_aligned_co2_temperature_data()
    test_start_index = len(years) - TEST_YEARS_COUNT

    years_train = years[:test_start_index]
    years_test = years[test_start_index:]
    co2_train = co2[:test_start_index]
    co2_test = co2[test_start_index:]
    temp_train = temp[:test_start_index]
    temp_test = temp[test_start_index:]

    raw_df = pd.DataFrame({
        "Year": years,
        "CO2_ppm": co2,
        "Temperature_Change_From_1951_1980_C": temp,
    })
    raw_df.to_csv(REL_OUTPUT_DIR / "aligned_co2_temperature_data.csv", index=False, encoding="utf-8-sig")

    plot_relationship_raw(years, co2, temp)

    summary_rows = []
    parameter_rows = []
    fitted_rows = []
    fitted_models_for_plot = []

    for spec in RELATIONSHIP_MODELS:
        full_fit = fit_relationship_model(spec, co2, temp)
        train_fit = fit_relationship_model(spec, co2_train, temp_train)

        full_pred = full_fit["predict"](co2)
        train_pred = train_fit["predict"](co2_train)
        test_pred = train_fit["predict"](co2_test)

        full_metrics = evaluate_model(temp, full_pred, full_fit["parameter_count"])
        train_metrics = evaluate_model(temp_train, train_pred, train_fit["parameter_count"])
        test_metrics = evaluate_model(temp_test, test_pred, train_fit["parameter_count"])

        summary_row = {
            "Model": spec["name"],
            "Model_Key": spec["key"],
            "Model_Type": spec["type"],
            "Full_Formula": full_fit["formula"],
            "Train_Formula": train_fit["formula"],
        }
        summary_row.update(prefix_metrics("Full", full_metrics))
        summary_row.update(prefix_metrics("Train", train_metrics))
        summary_row.update(prefix_metrics("Test", test_metrics))
        summary_rows.append(summary_row)

        for stage, fit_result in [("full_data", full_fit), ("training_data", train_fit)]:
            parameter_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Fit_Stage": stage,
                "Formula": fit_result["formula"],
                "Parameters_JSON": json.dumps(fit_result["parameters"]),
                "Parameter_Count": fit_result["parameter_count"],
            })

        for stage, stage_years, stage_co2, stage_temp, stage_pred in [
            ("full_data", years, co2, temp, full_pred),
            ("training_data", years_train, co2_train, temp_train, train_pred),
            ("test_data", years_test, co2_test, temp_test, test_pred),
        ]:
            for year, co2_value, observed, predicted in zip(stage_years, stage_co2, stage_temp, stage_pred):
                residual = observed - predicted
                fitted_rows.append({
                    "Model": spec["name"],
                    "Model_Key": spec["key"],
                    "Fit_Stage": stage,
                    "Year": int(year),
                    "CO2_ppm": float(co2_value),
                    "Observed_Temperature_Change_From_1951_1980_C": float(observed),
                    "Fitted_Temperature_Change_From_1951_1980_C": float(predicted),
                    "Residual_C": float(residual),
                    "Absolute_Residual_C": float(abs(residual)),
                })

        fitted_models_for_plot.append((spec["name"], full_fit))

    summary_df = pd.DataFrame(summary_rows).sort_values("Test_RMSE").reset_index(drop=True)
    parameter_df = pd.DataFrame(parameter_rows)
    fitted_df = pd.DataFrame(fitted_rows)

    summary_df.to_csv(REL_OUTPUT_DIR / "co2_temperature_relationship_model_summary.csv", index=False, encoding="utf-8-sig")
    parameter_df.to_csv(REL_OUTPUT_DIR / "co2_temperature_relationship_model_parameters.csv", index=False, encoding="utf-8-sig")
    fitted_df.to_csv(REL_OUTPUT_DIR / "co2_temperature_relationship_fitted_residuals.csv", index=False, encoding="utf-8-sig")

    for spec in RELATIONSHIP_MODELS:
        model_dir = REL_OUTPUT_DIR / "models" / spec["key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        summary_df[summary_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_summary.csv", index=False, encoding="utf-8-sig"
        )
        parameter_df[parameter_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_parameters.csv", index=False, encoding="utf-8-sig"
        )
        fitted_df[fitted_df["Model_Key"] == spec["key"]].to_csv(
            model_dir / f"{spec['key']}_fitted_residuals.csv", index=False, encoding="utf-8-sig"
        )

    correlation_summary = {
        "Aligned_Period": f"{years[0]}-{years[-1]}",
        "Observations": int(len(years)),
        "Pearson_R_CO2_Temp": float(pd.Series(co2).corr(pd.Series(temp), method="pearson")),
        "Spearman_R_CO2_Temp": float(pd.Series(co2).corr(pd.Series(temp), method="spearman")),
        "Pearson_R_Year_Temp": float(pd.Series(years).corr(pd.Series(temp), method="pearson")),
        "Pearson_R_Year_CO2": float(pd.Series(years).corr(pd.Series(co2), method="pearson")),
    }

    (REL_OUTPUT_DIR / "co2_temperature_correlation_summary.json").write_text(
        json.dumps(to_builtin(correlation_summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    plot_relationship_fits(co2, temp, fitted_models_for_plot)
    plot_relationship_error_comparison(summary_df, "Train", "co2_temperature_training_error_comparison.svg", "CO2-Temperature Relationship Training Error Comparison")
    plot_relationship_error_comparison(summary_df, "Test", "co2_temperature_test_error_comparison.svg", "CO2-Temperature Relationship Test Error Comparison")
    plot_relationship_test_residuals(fitted_df)

    relationship_payload = {
        "aligned_data": raw_df.to_dict(orient="records"),
        "model_summary": summary_df.to_dict(orient="records"),
        "model_parameters": parameter_df.to_dict(orient="records"),
        "fitted_values_and_residuals": fitted_df.to_dict(orient="records"),
        "correlation_summary": correlation_summary,
    }
    (REL_OUTPUT_DIR / "co2_temperature_relationship_results.json").write_text(
        json.dumps(to_builtin(relationship_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    return {
        "summary": summary_df,
        "parameters": parameter_df,
        "correlation_summary": correlation_summary,
    }


# ============================================================
# 3. Main workflow
# ============================================================

def main():
    reset_output_dir(OUTPUT_DIR)
    TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (TEMP_OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (REL_OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    temperature_results = run_temperature_forecast_models()
    relationship_results = run_relationship_models()

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    payload = {
        "run_timestamp": run_timestamp,
        "settings": {
            "test_years_count": TEST_YEARS_COUNT,
            "future_end_year": FUTURE_END_YEAR,
            "plot_end_year": PLOT_END_YEAR,
            "temperature_change_targets_from_1951_1980": TEMP_CHANGE_TARGETS,
        },
        "temperature_best_by_test_rmse": temperature_results["summary"].iloc[0].to_dict(),
        "relationship_best_by_test_rmse": relationship_results["summary"].iloc[0].to_dict(),
        "relationship_correlation_summary": relationship_results["correlation_summary"],
    }

    (OUTPUT_DIR / "question_2ab_summary.json").write_text(
        json.dumps(to_builtin(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    output_files = [
        str(path.relative_to(OUTPUT_DIR)).replace("\\", "/")
        for path in sorted(OUTPUT_DIR.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "run_timestamp": run_timestamp,
        "script": Path(__file__).name,
        "settings": payload["settings"],
        "output_directory": str(OUTPUT_DIR),
        "output_files": output_files,
        "temperature_best_model_by_test_rmse": temperature_results["summary"].iloc[0].to_dict(),
        "relationship_best_model_by_test_rmse": relationship_results["summary"].iloc[0].to_dict(),
    }
    (OUTPUT_DIR / "question_2ab_manifest.json").write_text(
        json.dumps(to_builtin(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    console_lines = [
        "=" * 80,
        "QUESTION 2A/2B MODELING COMPLETE",
        "=" * 80,
        "",
        "2a best temperature forecast model by test RMSE:",
        f"  {temperature_results['summary'].iloc[0]['Model']} | Test RMSE = {temperature_results['summary'].iloc[0]['Test_RMSE']:.4f} C",
        "",
        "2b best CO2-temperature relationship model by test RMSE:",
        f"  {relationship_results['summary'].iloc[0]['Model']} | Test RMSE = {relationship_results['summary'].iloc[0]['Test_RMSE']:.4f} C",
        "",
        "CO2-temperature correlation:",
        f"  Pearson r = {relationship_results['correlation_summary']['Pearson_R_CO2_Temp']:.4f}",
        f"  Spearman r = {relationship_results['correlation_summary']['Spearman_R_CO2_Temp']:.4f}",
        "",
        "Outputs saved in:",
        f"  {OUTPUT_DIR}",
    ]
    (OUTPUT_DIR / "question_2ab_console_output.txt").write_text(
        "\n".join(console_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("QUESTION 2A/2B MODELING COMPLETE")
    print("=" * 80)
    print("\n2a best temperature forecast model by test RMSE:")
    best_temp = temperature_results["summary"].iloc[0]
    print(f"  {best_temp['Model']} | Test RMSE = {best_temp['Test_RMSE']:.4f} C")
    print("\n2b best CO2-temperature relationship model by test RMSE:")
    best_relation = relationship_results["summary"].iloc[0]
    print(f"  {best_relation['Model']} | Test RMSE = {best_relation['Test_RMSE']:.4f} C")
    print("\nCO2-temperature correlation:")
    print(f"  Pearson r = {relationship_results['correlation_summary']['Pearson_R_CO2_Temp']:.4f}")
    print(f"  Spearman r = {relationship_results['correlation_summary']['Spearman_R_CO2_Temp']:.4f}")
    print("\nOutputs saved in:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
