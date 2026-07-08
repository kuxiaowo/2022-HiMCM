# question_1b_annual_models.py
# 1b: Unified annual fitting models for annual CO2 data
#
# Required file in the same directory:
#   himcm_data.py

from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from himcm_data import get_co2_data


def reset_output_dir(output_dir):
    """
    Remove old generated output inside the script's own output directory.
    """

    script_dir = Path(__file__).resolve().parent
    resolved_output_dir = (script_dir / output_dir).resolve()

    if script_dir not in resolved_output_dir.parents:
        raise RuntimeError(f"Refusing to clean outside script directory: {resolved_output_dir}")

    resolved_output_dir.mkdir(exist_ok=True)

    for child in resolved_output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


# ============================================================
# 0. Settings
# ============================================================

OUTPUT_DIR = Path("annual_fit_outputs")
reset_output_dir(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_FIGURES = False

# Keep the last 10 years as test data, consistent with the original scripts.
TEST_YEARS_COUNT = 10

MODEL_SPECS = [
    {
        "name": "Quadratic",
        "key": "quadratic",
        "model_type": "polynomial",
        "degree": 2,
        "parameter_count": 3,
    },
    {
        "name": "Cubic",
        "key": "cubic",
        "model_type": "polynomial",
        "degree": 3,
        "parameter_count": 4,
    },
    {
        "name": "Quartic",
        "key": "quartic",
        "model_type": "polynomial",
        "degree": 4,
        "parameter_count": 5,
    },
    {
        "name": "Exponential",
        "key": "exponential",
        "model_type": "exponential",
        "degree": None,
        "parameter_count": 2,
    },
]


# ============================================================
# 1. Helper functions
# ============================================================

console_lines = []


def log(message=""):
    """
    Print a message and keep a copy for annual_console_output.txt.
    """

    text = str(message)
    print(text)
    console_lines.append(text)


def to_builtin(value):
    """
    Convert numpy values into JSON-friendly Python values.
    """

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
    """
    Calculate error metrics.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    residuals = y_true - y_pred

    sse = np.sum(residuals ** 2)
    mse = sse / n
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / y_true)) * 100

    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    if sst == 0:
        r2 = np.nan
    else:
        r2 = 1 - sse / sst

    if n > parameter_count and not np.isnan(r2):
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - parameter_count)
    else:
        adjusted_r2 = np.nan

    return {
        "SSE": float(sse),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "Adjusted_R2": float(adjusted_r2),
    }


def print_metrics(metrics):
    """
    Print error metrics.
    """

    log(f"SSE          = {metrics['SSE']:.4f}")
    log(f"MSE          = {metrics['MSE']:.4f}")
    log(f"RMSE         = {metrics['RMSE']:.4f} ppm")
    log(f"MAE          = {metrics['MAE']:.4f} ppm")
    log(f"MAPE         = {metrics['MAPE']:.4f}%")
    log(f"R^2          = {metrics['R2']:.4f}")
    log(f"Adjusted R^2 = {metrics['Adjusted_R2']:.4f}")


def coefficient_letter(position):
    """
    Return coefficient labels a, b, c, ...
    """

    return chr(ord("a") + position)


def polynomial_formula(coefficients, first_year):
    """
    Build a readable polynomial formula.
    """

    degree = len(coefficients) - 1
    terms = []

    for index, coefficient in enumerate(coefficients):
        power = degree - index

        if power == 0:
            terms.append(f"{coefficient:.10f}")
        elif power == 1:
            terms.append(f"{coefficient:.10f} * t")
        else:
            terms.append(f"{coefficient:.10f} * t^{power}")

    return "CO2 = " + " + ".join(terms) + f", where t = year - {first_year}"


def exponential_formula(parameters, first_year):
    """
    Build a readable exponential formula.
    """

    return (
        f"CO2 = {parameters['A']:.10f} * exp({parameters['B']:.10f} * t), "
        f"where t = year - {first_year}"
    )


def fit_model(spec, t_values, co2_values, first_year):
    """
    Fit one configured model and return prediction metadata.
    """

    if spec["model_type"] == "polynomial":
        coefficients = np.polyfit(t_values, co2_values, spec["degree"])
        parameters = {
            coefficient_letter(index): float(value)
            for index, value in enumerate(coefficients)
        }

        def predict(new_t_values, fitted_coefficients=coefficients):
            return np.polyval(fitted_coefficients, new_t_values)

        return {
            "coefficients": parameters,
            "formula": polynomial_formula(coefficients, first_year),
            "predict": predict,
        }

    log_co2_values = np.log(co2_values)
    linear_coefficients = np.polyfit(t_values, log_co2_values, 1)
    b_value = float(linear_coefficients[0])
    ln_a_value = float(linear_coefficients[1])
    a_value = float(np.exp(ln_a_value))
    parameters = {
        "A": a_value,
        "B": b_value,
        "ln_A": ln_a_value,
    }

    def predict(new_t_values, fitted_a=a_value, fitted_b=b_value):
        return fitted_a * np.exp(fitted_b * new_t_values)

    return {
        "coefficients": parameters,
        "formula": exponential_formula(parameters, first_year),
        "predict": predict,
    }


def safe_filename(model_key):
    """
    Convert a model key into a safe file name.
    """

    return model_key.replace(" ", "_").replace("/", "_")


def model_output_dir(spec):
    """
    Return the per-model output directory.
    """

    return OUTPUT_DIR / "models" / safe_filename(spec["key"])


def save_or_show(filename):
    """
    Save current figure and optionally show it.
    """

    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def plot_fit_and_forecast(
    spec,
    years,
    co2,
    years_train,
    years_test,
    t_grid,
    years_grid,
    full_fit,
    train_fit,
    train_pred,
    test_pred,
):
    """
    Plot original annual data, full-data fitted values, and test forecast.
    """

    plt.figure(figsize=(12, 6))

    plt.scatter(
        years,
        co2,
        color="blue",
        s=28,
        label="Original annual data",
    )

    plt.plot(
        years_grid,
        full_fit["predict"](t_grid),
        color="green",
        linewidth=2,
        label=f"Full-data {spec['name']} fitted curve",
    )

    plt.plot(
        years_train,
        train_pred,
        color="orange",
        linewidth=2,
        label=f"Training-only {spec['name']} fitted values",
    )

    plt.plot(
        years_test,
        test_pred,
        color="orange",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=4,
        label=f"Training-only {spec['name']} forecast",
    )

    plt.axvline(
        x=years_test[0],
        color="gray",
        linestyle="--",
        label="Start of test period",
    )

    plt.xlabel("Year")
    plt.ylabel("CO2 concentration / ppm")
    plt.title(f"{spec['name']} Model: Annual CO2 Fit and Test Forecast")
    plt.legend()
    plt.grid(True)

    filename = model_output_dir(spec) / f"{safe_filename(spec['key'])}_fit_forecast.svg"
    save_or_show(filename)


def plot_test_forecast(spec, years_test, co2_test, test_pred):
    """
    Plot observed test data and forecast data.
    """

    plt.figure(figsize=(10, 5))

    plt.scatter(
        years_test,
        co2_test,
        color="blue",
        s=36,
        label="Observed test data",
    )

    plt.plot(
        years_test,
        test_pred,
        color="orange",
        linewidth=2,
        marker="o",
        markersize=4,
        label=f"{spec['name']} forecast",
    )

    plt.xlabel("Year")
    plt.ylabel("CO2 concentration / ppm")
    plt.title(f"{spec['name']} Model: Test Period Forecast")
    plt.legend()
    plt.grid(True)

    filename = model_output_dir(spec) / f"{safe_filename(spec['key'])}_test_forecast.svg"
    save_or_show(filename)


def plot_residuals_over_time(spec, years_values, residuals, data_label):
    """
    Plot residuals over time.
    """

    plt.figure(figsize=(10, 5))

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        label="Zero residual",
    )

    plt.plot(
        years_values,
        residuals,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="Residuals",
    )

    plt.xlabel("Year")
    plt.ylabel("Residual / ppm")
    plt.title(f"{spec['name']} Model: {data_label} Residuals Over Time")
    plt.legend()
    plt.grid(True)

    filename = model_output_dir(spec) / (
        f"{safe_filename(spec['key'])}_{data_label}_residuals_over_time.svg"
    )
    save_or_show(filename)


def plot_test_error_comparison(summary_df):
    """
    Plot RMSE, MAE, and MAPE comparison on the test set.
    """

    x_values = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        x_values - width,
        summary_df["Test_RMSE"],
        width,
        label="Test RMSE",
    )

    plt.bar(
        x_values,
        summary_df["Test_MAE"],
        width,
        label="Test MAE",
    )

    plt.bar(
        x_values + width,
        summary_df["Test_MAPE"],
        width,
        label="Test MAPE (%)",
    )

    plt.xticks(
        x_values,
        summary_df["Model"],
        rotation=20,
        ha="right",
    )

    plt.ylabel("Error value")
    plt.title("Annual Model Test Error Comparison")
    plt.legend()
    plt.grid(axis="y")

    filename = OUTPUT_DIR / "annual_test_error_comparison.svg"
    save_or_show(filename)


def plot_train_error_comparison(summary_df):
    """
    Plot RMSE, MAE, and MAPE comparison on the training set.
    """

    x_values = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        x_values - width,
        summary_df["Train_RMSE"],
        width,
        label="Train RMSE",
    )

    plt.bar(
        x_values,
        summary_df["Train_MAE"],
        width,
        label="Train MAE",
    )

    plt.bar(
        x_values + width,
        summary_df["Train_MAPE"],
        width,
        label="Train MAPE (%)",
    )

    plt.xticks(
        x_values,
        summary_df["Model"],
        rotation=20,
        ha="right",
    )

    plt.ylabel("Error value")
    plt.title("Annual Model Training Error Comparison")
    plt.legend()
    plt.grid(axis="y")

    filename = OUTPUT_DIR / "annual_train_error_comparison.svg"
    save_or_show(filename)


def plot_test_residual_comparison(test_prediction_df):
    """
    Plot test residual lines for all annual models.
    """

    plt.figure(figsize=(12, 6))

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        label="Zero residual",
    )

    for model_name, model_df in test_prediction_df.groupby("Model", sort=False):
        plt.plot(
            model_df["Year"],
            model_df["Error"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=model_name,
        )

    plt.xlabel("Year")
    plt.ylabel("Residual / ppm")
    plt.title("Annual Model Test Residual Comparison")
    plt.legend()
    plt.grid(True)

    filename = OUTPUT_DIR / "annual_test_residual_comparison.svg"
    save_or_show(filename)


def parameter_row(
    spec,
    fit_stage,
    fit_result,
    data_start_year,
    data_end_year,
    first_year,
    test_start_year,
):
    """
    Build one row for annual_model_parameters.csv.
    """

    row = {
        "Model": spec["name"],
        "Model_Key": spec["key"],
        "Model_Type": spec["model_type"],
        "Fit_Stage": fit_stage,
        "Polynomial_Degree": spec["degree"],
        "First_Year_For_t": int(first_year),
        "Data_Start_Year": int(data_start_year),
        "Data_End_Year": int(data_end_year),
        "Test_Start_Year": int(test_start_year),
        "Formula": fit_result["formula"],
        "Coefficients_JSON": json.dumps(fit_result["coefficients"]),
        "Coefficient_a": np.nan,
        "Coefficient_b": np.nan,
        "Coefficient_c": np.nan,
        "Coefficient_d": np.nan,
        "Coefficient_e": np.nan,
        "Coefficient_A": np.nan,
        "Coefficient_B": np.nan,
        "Coefficient_ln_A": np.nan,
    }

    for key, value in fit_result["coefficients"].items():
        column_name = f"Coefficient_{key}"
        if column_name in row:
            row[column_name] = value

    return row


def metrics_columns(prefix, metrics):
    """
    Prefix metric names for summary rows.
    """

    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def records_from_dataframe(dataframe):
    """
    Convert a DataFrame to plain records for JSON.
    """

    return to_builtin(dataframe.to_dict(orient="records"))


# ============================================================
# 2. Main workflow
# ============================================================

def main():
    years, co2 = get_co2_data()

    first_year = int(years[0])
    t_values = years - first_year

    test_start_index = len(years) - TEST_YEARS_COUNT

    years_train = years[:test_start_index]
    years_test = years[test_start_index:]

    co2_train = co2[:test_start_index]
    co2_test = co2[test_start_index:]

    t_train = t_values[:test_start_index]
    t_test = t_values[test_start_index:]

    t_grid = np.linspace(t_values[0], t_values[-1], 300)
    years_grid = t_grid + first_year

    summary_rows = []
    parameter_rows = []
    fitted_rows = []
    test_prediction_rows = []
    json_models = []

    log("=" * 80)
    log("ANNUAL FITTING MODELS FOR CO2 DATA")
    log("=" * 80)
    log(f"Full data period: {years[0]} to {years[-1]}")
    log(f"Full data observations: {len(years)}")
    log(f"Training period: {years_train[0]} to {years_train[-1]}")
    log(f"Test period: {years_test[0]} to {years_test[-1]}")
    log(f"Test observations: {len(years_test)} years")
    log(f"Number of candidate models: {len(MODEL_SPECS)}")

    for spec in MODEL_SPECS:
        log("")
        log("=" * 80)
        log(f"{spec['name'].upper()} MODEL")
        log("=" * 80)
        log(f"Model type: {spec['model_type']}")
        if spec["degree"] is not None:
            log(f"Polynomial degree: {spec['degree']}")
        log(f"Parameter count: {spec['parameter_count']}")

        full_fit = fit_model(spec, t_values, co2, first_year)
        train_fit = fit_model(spec, t_train, co2_train, first_year)

        full_pred = full_fit["predict"](t_values)
        train_pred = train_fit["predict"](t_train)
        test_pred = train_fit["predict"](t_test)

        full_residuals = co2 - full_pred
        train_residuals = co2_train - train_pred
        test_residuals = co2_test - test_pred

        full_metrics = evaluate_model(
            y_true=co2,
            y_pred=full_pred,
            parameter_count=spec["parameter_count"],
        )
        train_metrics = evaluate_model(
            y_true=co2_train,
            y_pred=train_pred,
            parameter_count=spec["parameter_count"],
        )
        test_metrics = evaluate_model(
            y_true=co2_test,
            y_pred=test_pred,
            parameter_count=spec["parameter_count"],
        )

        log("")
        log("Full-data model")
        log("-" * 35)
        log(full_fit["formula"])
        log("Full-data in-sample error:")
        print_metrics(full_metrics)

        log("")
        log("Training-only model")
        log("-" * 35)
        log(train_fit["formula"])
        log("Training-only in-sample error:")
        print_metrics(train_metrics)

        log("")
        log("Training-only model on test data:")
        print_metrics(test_metrics)

        summary_row = {
            "Model": spec["name"],
            "Model_Key": spec["key"],
            "Model_Type": spec["model_type"],
            "Polynomial_Degree": spec["degree"],
            "Parameter_Count": spec["parameter_count"],
            "Full_Formula": full_fit["formula"],
            "Train_Formula": train_fit["formula"],
        }
        summary_row.update(metrics_columns("Full", full_metrics))
        summary_row.update(metrics_columns("Train", train_metrics))
        summary_row.update(metrics_columns("Test", test_metrics))
        summary_rows.append(summary_row)

        parameter_rows.append(
            parameter_row(
                spec=spec,
                fit_stage="full_data",
                fit_result=full_fit,
                data_start_year=years[0],
                data_end_year=years[-1],
                first_year=first_year,
                test_start_year=years_test[0],
            )
        )
        parameter_rows.append(
            parameter_row(
                spec=spec,
                fit_stage="training_data",
                fit_result=train_fit,
                data_start_year=years_train[0],
                data_end_year=years_train[-1],
                first_year=first_year,
                test_start_year=years_test[0],
            )
        )

        for year, observed, fitted, residual in zip(
            years,
            co2,
            full_pred,
            full_residuals,
        ):
            fitted_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Fit_Stage": "full_data",
                "Year": int(year),
                "Observed_CO2": float(observed),
                "Fitted_CO2": float(fitted),
                "Residual": float(residual),
                "Absolute_Residual": float(abs(residual)),
            })

        for year, observed, fitted, residual in zip(
            years_train,
            co2_train,
            train_pred,
            train_residuals,
        ):
            fitted_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Fit_Stage": "training_data",
                "Year": int(year),
                "Observed_CO2": float(observed),
                "Fitted_CO2": float(fitted),
                "Residual": float(residual),
                "Absolute_Residual": float(abs(residual)),
            })

        for year, observed, predicted, residual in zip(
            years_test,
            co2_test,
            test_pred,
            test_residuals,
        ):
            test_prediction_rows.append({
                "Model": spec["name"],
                "Model_Key": spec["key"],
                "Year": int(year),
                "Observed_CO2": float(observed),
                "Forecast_CO2": float(predicted),
                "Error": float(residual),
                "Absolute_Error": float(abs(residual)),
            })

        plot_fit_and_forecast(
            spec=spec,
            years=years,
            co2=co2,
            years_train=years_train,
            years_test=years_test,
            t_grid=t_grid,
            years_grid=years_grid,
            full_fit=full_fit,
            train_fit=train_fit,
            train_pred=train_pred,
            test_pred=test_pred,
        )

        plot_test_forecast(
            spec=spec,
            years_test=years_test,
            co2_test=co2_test,
            test_pred=test_pred,
        )

        plot_residuals_over_time(
            spec=spec,
            years_values=years,
            residuals=full_residuals,
            data_label="full_data",
        )

        plot_residuals_over_time(
            spec=spec,
            years_values=years_train,
            residuals=train_residuals,
            data_label="training_data",
        )

        plot_residuals_over_time(
            spec=spec,
            years_values=years_test,
            residuals=test_residuals,
            data_label="test",
        )

        json_models.append({
            "model": spec["name"],
            "model_key": spec["key"],
            "model_type": spec["model_type"],
            "polynomial_degree": spec["degree"],
            "parameter_count": spec["parameter_count"],
            "full_data": {
                "formula": full_fit["formula"],
                "coefficients": full_fit["coefficients"],
                "metrics": full_metrics,
            },
            "training_data": {
                "formula": train_fit["formula"],
                "coefficients": train_fit["coefficients"],
                "metrics": train_metrics,
            },
            "test_data": {
                "metrics": test_metrics,
                "predictions": [
                    {
                        "year": int(year),
                        "observed_co2": float(observed),
                        "forecast_co2": float(predicted),
                        "error": float(residual),
                        "absolute_error": float(abs(residual)),
                    }
                    for year, observed, predicted, residual in zip(
                        years_test,
                        co2_test,
                        test_pred,
                        test_residuals,
                    )
                ],
            },
        })

    summary_df = pd.DataFrame(summary_rows)
    parameter_df = pd.DataFrame(parameter_rows)
    fitted_df = pd.DataFrame(fitted_rows)
    test_prediction_df = pd.DataFrame(test_prediction_rows)

    summary_df = summary_df.sort_values(
        by="Test_RMSE",
    ).reset_index(drop=True)

    summary_csv = OUTPUT_DIR / "annual_model_summary.csv"
    parameter_csv = OUTPUT_DIR / "annual_model_parameters.csv"
    fitted_csv = OUTPUT_DIR / "annual_fitted_values.csv"
    test_prediction_csv = OUTPUT_DIR / "annual_test_predictions.csv"
    results_json_path = OUTPUT_DIR / "annual_results.json"
    manifest_json_path = OUTPUT_DIR / "annual_run_manifest.json"
    console_output_path = OUTPUT_DIR / "annual_console_output.txt"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    parameter_df.to_csv(parameter_csv, index=False, encoding="utf-8-sig")
    fitted_df.to_csv(fitted_csv, index=False, encoding="utf-8-sig")
    test_prediction_df.to_csv(test_prediction_csv, index=False, encoding="utf-8-sig")

    per_model_data_files = []
    json_models_by_key = {
        model_payload["model_key"]: model_payload
        for model_payload in json_models
    }

    for spec in MODEL_SPECS:
        model_key = spec["key"]
        model_dir = model_output_dir(spec)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_parameter_csv = model_dir / f"{model_key}_model_parameters.csv"
        model_fitted_csv = model_dir / f"{model_key}_fitted_values.csv"
        model_test_prediction_csv = model_dir / f"{model_key}_test_predictions.csv"
        model_results_json = model_dir / f"{model_key}_results.json"

        parameter_df[parameter_df["Model_Key"] == model_key].to_csv(
            model_parameter_csv,
            index=False,
            encoding="utf-8-sig",
        )
        fitted_df[fitted_df["Model_Key"] == model_key].to_csv(
            model_fitted_csv,
            index=False,
            encoding="utf-8-sig",
        )
        test_prediction_df[test_prediction_df["Model_Key"] == model_key].to_csv(
            model_test_prediction_csv,
            index=False,
            encoding="utf-8-sig",
        )
        model_results_json.write_text(
            json.dumps(
                to_builtin(json_models_by_key[model_key]),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        per_model_data_files.extend([
            str(model_parameter_csv.relative_to(OUTPUT_DIR)),
            str(model_fitted_csv.relative_to(OUTPUT_DIR)),
            str(model_test_prediction_csv.relative_to(OUTPUT_DIR)),
            str(model_results_json.relative_to(OUTPUT_DIR)),
        ])

    plot_train_error_comparison(summary_df)
    plot_test_error_comparison(summary_df)
    plot_test_residual_comparison(test_prediction_df)

    model_plot_files = []
    for spec in MODEL_SPECS:
        model_key = safe_filename(spec["key"])
        model_relative_dir = Path("models") / model_key
        model_plot_files.extend([
            str(model_relative_dir / f"{model_key}_fit_forecast.svg"),
            str(model_relative_dir / f"{model_key}_test_forecast.svg"),
            str(model_relative_dir / f"{model_key}_full_data_residuals_over_time.svg"),
            str(model_relative_dir / f"{model_key}_training_data_residuals_over_time.svg"),
            str(model_relative_dir / f"{model_key}_test_residuals_over_time.svg"),
        ])

    output_files = sorted([
        summary_csv.name,
        parameter_csv.name,
        fitted_csv.name,
        test_prediction_csv.name,
        results_json_path.name,
        manifest_json_path.name,
        console_output_path.name,
        "annual_train_error_comparison.svg",
        "annual_test_error_comparison.svg",
        "annual_test_residual_comparison.svg",
        *model_plot_files,
        *per_model_data_files,
    ])

    best_test_model = summary_df.loc[summary_df["Test_RMSE"].idxmin()]

    results_payload = {
        "metadata": {
            "script": Path(__file__).name,
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(OUTPUT_DIR),
            "data_period": {
                "first_year": int(years[0]),
                "last_year": int(years[-1]),
                "observations": int(len(years)),
            },
            "training_period": {
                "first_year": int(years_train[0]),
                "last_year": int(years_train[-1]),
                "observations": int(len(years_train)),
            },
            "test_period": {
                "first_year": int(years_test[0]),
                "last_year": int(years_test[-1]),
                "observations": int(len(years_test)),
            },
        },
        "models": json_models,
        "summary": records_from_dataframe(summary_df),
        "parameters": records_from_dataframe(parameter_df),
        "fitted_values": records_from_dataframe(fitted_df),
        "test_predictions": records_from_dataframe(test_prediction_df),
    }

    manifest_payload = {
        "script": Path(__file__).name,
        "run_timestamp": results_payload["metadata"]["run_timestamp"],
        "output_dir": str(OUTPUT_DIR),
        "config": {
            "test_years_count": TEST_YEARS_COUNT,
            "show_figures": SHOW_FIGURES,
            "models": [
                {
                    "name": spec["name"],
                    "key": spec["key"],
                    "model_type": spec["model_type"],
                    "degree": spec["degree"],
                    "parameter_count": spec["parameter_count"],
                }
                for spec in MODEL_SPECS
            ],
        },
        "data": results_payload["metadata"],
        "best_models": {
            "best_by_test_rmse": {
                "model": best_test_model["Model"],
                "model_key": best_test_model["Model_Key"],
                "test_rmse": float(best_test_model["Test_RMSE"]),
            }
        },
        "output_files": output_files,
    }

    results_json_path.write_text(
        json.dumps(to_builtin(results_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest_json_path.write_text(
        json.dumps(to_builtin(manifest_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    log("")
    log("=" * 80)
    log("SUMMARY SORTED BY TEST RMSE")
    log("=" * 80)
    log(summary_df.to_string(index=False))

    log("")
    log("=" * 80)
    log("BEST MODEL INDICATORS")
    log("=" * 80)
    log(f"Best by test RMSE: {best_test_model['Model']}")

    log("")
    log("Saved result files:")
    for filename in output_files:
        log(f"  {OUTPUT_DIR / filename}")

    console_output_path.write_text(
        "\n".join(console_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
