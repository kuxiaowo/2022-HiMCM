# question_1c_co2_prediction_from_outputs.py
# Read saved model outputs and answer question 1c using only:
#   1. Quadratic annual model
#   2. SARIMA(1,1,1)(2,1,2,12)
#
# SARIMA is a monthly model, so its future monthly forecasts are converted
# to annual averages before comparing with the annual quadratic model.

import ast
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from himcm_data import get_co2_data, get_monthly_co2_data

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as error:
    raise ImportError(
        "statsmodels is required. Install it with: pip install statsmodels"
    ) from error


# ============================================================
# 0. Settings
# ============================================================

ANNUAL_OUTPUT_DIR = Path("annual_fit_outputs")
SARIMA_OUTPUT_DIR = Path("SARIMA图")
OUTPUT_DIR = Path("question_1c_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FIT_STAGE_FOR_CONCLUSION = "full_data"
ANNUAL_MODEL_KEY = "quadratic"
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (2, 1, 2, 12)
SARIMA_TREND = "c"
SARIMA_MODEL_NAME = "SARIMA(1,1,1)(2,1,2,12)"

TARGET_PPM = 685.0
PREDICTION_YEARS = [2050, 2100]
SEARCH_END_YEAR = 2200
PLOT_END_YEAR = 2100


# ============================================================
# 1. Shared helpers
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def clear_old_plots():
    for path in OUTPUT_DIR.glob("question_1c_*.svg"):
        path.unlink()


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
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    return value


# ============================================================
# 2. Quadratic annual model
# ============================================================

def predict_annual_model(parameter_row, year):
    first_year = int(parameter_row["First_Year_For_t"])
    t_value = year - first_year
    coefficients = json.loads(parameter_row["Coefficients_JSON"])

    coefficient_values = [
        coefficients[key]
        for key in sorted(coefficients.keys())
    ]

    return float(np.polyval(coefficient_values, t_value))


def estimate_annual_threshold_crossing(parameter_row, target_ppm, start_year, end_year):
    previous_year = start_year
    previous_value = predict_annual_model(parameter_row, previous_year) - target_ppm

    if previous_value >= 0:
        return previous_year, float(previous_year)

    for year in range(start_year + 1, end_year + 1):
        current_value = predict_annual_model(parameter_row, year) - target_ppm

        if current_value >= 0:
            low = float(previous_year)
            high = float(year)

            for _ in range(80):
                middle = (low + high) / 2
                middle_value = predict_annual_model(parameter_row, middle) - target_ppm

                if middle_value >= 0:
                    high = middle
                else:
                    low = middle

            return year, high

        previous_year = year

    return None, None


def load_quadratic_result():
    parameter_csv = ANNUAL_OUTPUT_DIR / "annual_model_parameters.csv"
    summary_csv = ANNUAL_OUTPUT_DIR / "annual_model_summary.csv"

    require_file(parameter_csv)
    require_file(summary_csv)

    parameter_df = pd.read_csv(parameter_csv)
    summary_df = pd.read_csv(summary_csv)

    parameter_matches = parameter_df[
        (parameter_df["Model_Key"] == ANNUAL_MODEL_KEY)
        & (parameter_df["Fit_Stage"] == FIT_STAGE_FOR_CONCLUSION)
    ]

    if len(parameter_matches) != 1:
        raise ValueError(
            f"Expected one {ANNUAL_MODEL_KEY!r} parameter row for "
            f"{FIT_STAGE_FOR_CONCLUSION!r}, found {len(parameter_matches)}."
        )

    parameter_row = parameter_matches.iloc[0]
    metric_row = summary_df[summary_df["Model_Key"] == ANNUAL_MODEL_KEY].iloc[0]

    latest_year = int(parameter_row["Data_End_Year"])
    first_integer_year, continuous_year = estimate_annual_threshold_crossing(
        parameter_row=parameter_row,
        target_ppm=TARGET_PPM,
        start_year=latest_year + 1,
        end_year=SEARCH_END_YEAR,
    )

    predictions = {
        year: predict_annual_model(parameter_row, year)
        for year in PREDICTION_YEARS
    }

    forecast_rows = []
    for year in range(latest_year + 1, SEARCH_END_YEAR + 1):
        forecast_rows.append({
            "Model": "Quadratic",
            "Model_Source": "annual_full_data",
            "Year": year,
            "Forecast_CO2_ppm": predict_annual_model(parameter_row, year),
        })

    return {
        "result": {
            "Model": "Quadratic",
            "Model_Source": "annual_full_data",
            "Prediction_2050_ppm": predictions[2050],
            "Prediction_2100_ppm": predictions[2100],
            "Reaches_685_by_2050": predictions[2050] >= TARGET_PPM,
            "First_Integer_Year_Reaches_685": first_integer_year,
            "Estimated_Continuous_Year_Reaches_685": continuous_year,
            "Test_RMSE": float(metric_row["Test_RMSE"]),
            "Selection_Note": "Annual quadratic model from annual_fit_outputs.",
            "Formula": parameter_row["Formula"],
        },
        "forecast_rows": forecast_rows,
    }


# ============================================================
# 3. SARIMA monthly model
# ============================================================

def load_sarima_manifest():
    manifest_path = SARIMA_OUTPUT_DIR / "SARIMA_run_manifest.json"
    require_file(manifest_path)

    with manifest_path.open(encoding="utf-8") as file:
        return json.load(file)


def build_monthly_series(manifest):
    monthly_df = get_monthly_co2_data(as_dataframe=True)

    monthly_df["date"] = pd.to_datetime(
        {
            "year": monthly_df["year"],
            "month": monthly_df["month"],
            "day": 1,
        }
    )
    monthly_df = monthly_df.sort_values("date").reset_index(drop=True)

    use_deseasonalized = manifest["config"]["use_deseasonalized"]
    value_column = "deseasonalized" if use_deseasonalized else "average"

    monthly_df = monthly_df[monthly_df[value_column] > 0].copy()

    data_end_date = manifest["config"]["data_end_date"]
    if data_end_date is not None:
        monthly_df = monthly_df[monthly_df["date"] <= pd.to_datetime(data_end_date)].copy()

    series = pd.Series(
        monthly_df[value_column].values,
        index=monthly_df["date"],
    ).asfreq("MS")

    if series.isna().any():
        series = series.interpolate(method="time")

    return series


def get_selected_sarima_spec():
    summary_csv = SARIMA_OUTPUT_DIR / "SARIMA_model_summary.csv"
    parameter_csv = SARIMA_OUTPUT_DIR / "SARIMA_model_parameters.csv"

    require_file(summary_csv)
    require_file(parameter_csv)

    summary_df = pd.read_csv(summary_csv)
    parameter_df = pd.read_csv(parameter_csv)

    matches = summary_df[
        (summary_df["Order"] == str(SARIMA_ORDER))
        & (summary_df["Seasonal_Order"] == str(SARIMA_SEASONAL_ORDER))
        & (summary_df["Trend"] == SARIMA_TREND)
    ].copy()

    if matches.empty:
        raise ValueError(
            "Requested SARIMA model was not found in SARIMA_model_summary.csv: "
            f"order={SARIMA_ORDER}, seasonal_order={SARIMA_SEASONAL_ORDER}, "
            f"trend={SARIMA_TREND}. Run 1b_arima.py with this model first."
        )

    best_row = matches.iloc[0]
    model_name = best_row["Model"]

    parameter_matches = parameter_df[
        (parameter_df["Model"] == model_name)
        & (parameter_df["Fit_Stage"] == FIT_STAGE_FOR_CONCLUSION)
    ]

    if len(parameter_matches) != 1:
        raise ValueError(
            f"Expected one SARIMA full-data parameter row for {model_name}, "
            f"found {len(parameter_matches)}."
        )

    parameter_row = parameter_matches.iloc[0]

    return {
        "model_name": model_name,
        "order": ast.literal_eval(best_row["Order"]),
        "seasonal_order": ast.literal_eval(best_row["Seasonal_Order"]),
        "trend": best_row["Trend"],
        "parameters": json.loads(parameter_row["Parameters_JSON"]),
        "test_rmse": float(best_row["Test_RMSE"]),
        "test_mae": float(best_row["Test_MAE"]),
        "test_mape": float(best_row["Test_MAPE"]),
        "test_r2": float(best_row["Test_R2"]),
    }


def forecast_sarima_future(series, spec):
    end_year = SEARCH_END_YEAR
    last_date = series.index[-1]
    final_forecast_date = pd.Timestamp(year=end_year, month=12, day=1)
    steps = (
        (final_forecast_date.year - last_date.year) * 12
        + (final_forecast_date.month - last_date.month)
    )

    if steps <= 0:
        raise ValueError("SEARCH_END_YEAR must be after the SARIMA data end year.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            series,
            order=spec["order"],
            seasonal_order=spec["seasonal_order"],
            trend=spec["trend"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        parameter_values = [
            spec["parameters"][parameter_name]
            for parameter_name in model.param_names
        ]

        result = model.filter(parameter_values)
        forecast = result.get_forecast(steps=steps).predicted_mean

    forecast.index = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq="MS",
    )

    return forecast


def load_sarima_result():
    manifest = load_sarima_manifest()
    series = build_monthly_series(manifest)
    spec = get_selected_sarima_spec()
    monthly_forecast = forecast_sarima_future(series, spec)

    annual_forecast = monthly_forecast.resample("YS").mean()
    annual_forecast.index = annual_forecast.index.year

    prediction_2050 = float(annual_forecast.loc[2050])
    prediction_2100 = float(annual_forecast.loc[2100])

    crossing_years = annual_forecast[annual_forecast >= TARGET_PPM]

    if len(crossing_years) > 0:
        first_integer_year = int(crossing_years.index[0])
        continuous_year = float(first_integer_year)
    else:
        first_integer_year = None
        continuous_year = None

    crossing_months = monthly_forecast[monthly_forecast >= TARGET_PPM]
    first_month = (
        crossing_months.index[0].strftime("%Y-%m")
        if len(crossing_months) > 0
        else None
    )

    forecast_rows = []
    for year, value in annual_forecast.items():
        forecast_rows.append({
            "Model": spec["model_name"],
            "Model_Source": "sarima_full_data_annual_mean",
            "Year": int(year),
            "Forecast_CO2_ppm": float(value),
        })

    return {
        "result": {
            "Model": spec["model_name"],
            "Model_Source": "sarima_full_data_annual_mean",
            "Prediction_2050_ppm": prediction_2050,
            "Prediction_2100_ppm": prediction_2100,
            "Reaches_685_by_2050": prediction_2050 >= TARGET_PPM,
            "First_Integer_Year_Reaches_685": first_integer_year,
            "Estimated_Continuous_Year_Reaches_685": continuous_year,
            "First_Month_Reaches_685": first_month,
            "Test_RMSE": spec["test_rmse"],
            "Selection_Note": (
                "Requested SARIMA(1,1,1)(2,1,2,12); monthly forecasts "
                "aggregated to annual means."
            ),
            "Formula": (
                f"order={spec['order']}, seasonal_order={spec['seasonal_order']}, "
                f"trend={spec['trend']}"
            ),
        },
        "forecast_rows": forecast_rows,
    }


# ============================================================
# 4. Plotting
# ============================================================

def plot_forecast_extension(
    result_df,
    forecast_df,
    output_filename,
    title,
    model_names=None,
):
    """
    Plot historical annual CO2 data and extend it with selected model forecasts.
    """

    historical_years, historical_co2 = get_co2_data()
    historical_df = pd.DataFrame({
        "Year": historical_years,
        "CO2": historical_co2,
    })
    historical_df = historical_df[historical_df["Year"] <= PLOT_END_YEAR]

    last_observed_year = int(historical_df["Year"].iloc[-1])
    last_observed_value = float(historical_df["CO2"].iloc[-1])

    if model_names is not None:
        result_df = result_df[result_df["Model"].isin(model_names)].copy()

    plt.figure(figsize=(12, 6))

    plt.plot(
        historical_df["Year"],
        historical_df["CO2"],
        color="black",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Observed annual CO2",
    )

    for _, result_row in result_df.iterrows():
        model_name = result_row["Model"]
        model_source = result_row["Model_Source"]
        model_forecast = forecast_df[
            (forecast_df["Model"] == model_name)
            & (forecast_df["Model_Source"] == model_source)
            & (forecast_df["Year"] <= PLOT_END_YEAR)
        ].copy()

        extension_years = [last_observed_year] + model_forecast["Year"].tolist()
        extension_values = [last_observed_value] + model_forecast["Forecast_CO2_ppm"].tolist()

        plt.plot(
            extension_years,
            extension_values,
            linewidth=2,
            label=f"{model_name} forecast",
        )

        for prediction_year in PREDICTION_YEARS:
            if prediction_year <= PLOT_END_YEAR:
                prediction_value = float(
                    result_row[f"Prediction_{prediction_year}_ppm"]
                )
                plt.scatter(
                    [prediction_year],
                    [prediction_value],
                    s=45,
                    zorder=4,
                )

    plt.axhline(
        y=TARGET_PPM,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{TARGET_PPM:.0f} ppm target",
    )

    plt.axvline(
        x=2050,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="2050",
    )

    plt.axvline(
        x=2100,
        color="gray",
        linestyle=":",
        linewidth=1,
        label="2100",
    )

    plt.xlabel("Year")
    plt.ylabel("CO2 concentration / ppm")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / output_filename
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def plot_all_forecast_extensions(result_df, forecast_df):
    quadratic_plot = plot_forecast_extension(
        result_df=result_df,
        forecast_df=forecast_df,
        output_filename="question_1c_quadratic_forecast_to_2100.svg",
        title="Observed CO2 Data Extended with Quadratic Forecast",
        model_names=["Quadratic"],
    )

    sarima_models = [
        model_name
        for model_name in result_df["Model"].tolist()
        if model_name != "Quadratic"
    ]
    sarima_plot = plot_forecast_extension(
        result_df=result_df,
        forecast_df=forecast_df,
        output_filename="question_1c_sarima_1_1_1_2_1_2_12_forecast_to_2100.svg",
        title="Observed CO2 Data Extended with SARIMA(1,1,1)(2,1,2,12) Forecast",
        model_names=sarima_models,
    )

    combined_plot = plot_forecast_extension(
        result_df=result_df,
        forecast_df=forecast_df,
        output_filename="question_1c_quadratic_and_sarima_forecast_to_2100.svg",
        title="Observed CO2 Data Extended with Quadratic and SARIMA Forecasts",
    )

    return {
        "quadratic_forecast": quadratic_plot,
        "sarima_forecast": sarima_plot,
        "combined_forecast": combined_plot,
    }


# ============================================================
# 5. Main workflow
# ============================================================

def main():
    clear_old_plots()

    quadratic = load_quadratic_result()
    sarima = load_sarima_result()

    result_df = pd.DataFrame([
        quadratic["result"],
        sarima["result"],
    ])

    forecast_df = pd.DataFrame(
        quadratic["forecast_rows"] + sarima["forecast_rows"]
    )

    prediction_csv = OUTPUT_DIR / "question_1c_model_predictions.csv"
    forecast_csv = OUTPUT_DIR / "question_1c_future_forecasts.csv"
    summary_json = OUTPUT_DIR / "question_1c_summary.json"
    forecast_plots = plot_all_forecast_extensions(result_df, forecast_df)

    result_df.to_csv(prediction_csv, index=False, encoding="utf-8-sig")
    forecast_df.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    payload = {
        "settings": {
            "annual_output_dir": str(ANNUAL_OUTPUT_DIR),
            "sarima_output_dir": str(SARIMA_OUTPUT_DIR),
            "fit_stage": FIT_STAGE_FOR_CONCLUSION,
            "annual_model_key": ANNUAL_MODEL_KEY,
            "sarima_order": SARIMA_ORDER,
            "sarima_seasonal_order": SARIMA_SEASONAL_ORDER,
            "sarima_trend": SARIMA_TREND,
            "sarima_model_name": SARIMA_MODEL_NAME,
            "target_ppm": TARGET_PPM,
            "prediction_years": PREDICTION_YEARS,
            "search_end_year": SEARCH_END_YEAR,
            "plot_end_year": PLOT_END_YEAR,
        },
        "plots": {
            key: str(path)
            for key, path in forecast_plots.items()
        },
        "models": result_df.to_dict(orient="records"),
        "conclusion": {
            "models_reaching_685_by_2050": result_df[
                result_df["Reaches_685_by_2050"]
            ]["Model"].tolist(),
            "models_not_reaching_685_by_2050": result_df[
                ~result_df["Reaches_685_by_2050"]
            ]["Model"].tolist(),
        },
    }

    summary_json.write_text(
        json.dumps(to_builtin(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("QUESTION 1C: QUADRATIC AND SARIMA(1,1,1)(2,1,2,12) CO2 PREDICTIONS")
    print("=" * 80)
    print(f"Target claim: CO2 reaches {TARGET_PPM:.0f} ppm by 2050")
    print("SARIMA output uses monthly forecasts aggregated to annual means.")
    print()
    print(result_df[
        [
            "Model",
            "Model_Source",
            "Prediction_2050_ppm",
            "Prediction_2100_ppm",
            "Reaches_685_by_2050",
            "First_Integer_Year_Reaches_685",
            "Test_RMSE",
        ]
    ].to_string(index=False))
    print()
    print("Saved result files:")
    print(f"  {prediction_csv}")
    print(f"  {forecast_csv}")
    print(f"  {summary_json}")
    for plot_path in forecast_plots.values():
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
