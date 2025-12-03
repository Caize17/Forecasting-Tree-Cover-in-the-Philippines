import warnings
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from flask import Flask, jsonify
from flask_cors import CORS

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

app = Flask(__name__)
CORS(app)

tree = pd.read_csv("Annual_Tree_Cover_Analysis_ha.csv")
agri = pd.read_csv("agriculture_ha.csv")
log = pd.read_csv("log_production.csv")
urban = pd.read_csv("urbanization_percentage.csv")
roads = pd.read_csv("national_roads.csv")
actual_tree = pd.read_csv("Actual_Tree_Cover.csv")  # 2021-2024

if "Region" not in roads.columns:
    all_regions = tree["Region"].unique()
    roads = roads.assign(key=1).merge(pd.DataFrame({"Region": all_regions, "key": 1}), on="key").drop("key", axis=1)

df = tree.merge(agri, on=["Region", "Year"], how="left")
df = df.merge(log, on=["Region", "Year"], how="left")
df = df.merge(urban, on=["Region", "Year"], how="left")
df = df.merge(roads, on=["Region", "Year"], how="left")

df = df.rename(columns={
    "tree_cover_ha": "Tree_Cover",
    "extrapolated_urbanization_percentage": "Urbanization",
    "harvested_area_ha": "Agriculture",
    "log_production_cbm": "Logging",
    "Total_km": "Roads_km"
})

regions = df["Region"].unique()
forecast_years = [2021, 2022, 2023, 2024, 2025]

comparison_rows = []
driver_results = {}
sarimax_results = {}
all_forecasts = []

for region in regions:
    reg_df = df[df["Region"] == region].sort_values("Year")
    exog_vars = ["Agriculture", "Logging", "Urbanization", "Roads_km"]
    
    X = sm.add_constant(reg_df[exog_vars])
    y = reg_df["Tree_Cover"]
    lr_model = sm.OLS(y, X).fit()
    driver_results[region] = lr_model
    
    forecast_exog = []
    for var in exog_vars:
        coef_var = np.polyfit(np.arange(len(reg_df)), reg_df[var].values, 1)
        forecast_exog.append(np.polyval(coef_var, np.arange(len(reg_df), len(reg_df)+len(forecast_years))))
    forecast_exog = np.array(forecast_exog).T

    try:
        sarimax_model = sm.tsa.SARIMAX(
            reg_df["Tree_Cover"],
            order=(1,1,0),
            seasonal_order=(0,0,0,0),
            exog=reg_df[exog_vars],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarimax_fit = sarimax_model.fit(disp=False)
        sarimax_results[region] = sarimax_fit
        tree_forecast = sarimax_fit.get_forecast(steps=len(forecast_years), exog=forecast_exog)
        forecast_values = tree_forecast.predicted_mean.values
    except Exception as e:
        print(f"SARIMAX failed for {region}, using LR fallback. Error: {e}")
        forecast_values = []
        for i in range(len(forecast_years)):
            row = [1] + list(forecast_exog[i])
            pred = lr_model.predict(row)[0]
            forecast_values.append(pred)

    
    for i, year in enumerate(forecast_years):
        all_forecasts.append([region, year, forecast_values[i]])

    #Compare with actual if available
    for i, year in enumerate(forecast_years):
        actual_row = actual_tree[(actual_tree["Region"] == region) & (actual_tree["Year"] == year)]
        if not actual_row.empty:
            actual_val = actual_row["tree_cover_ha"].values[0]
            sarimax_val = forecast_values[i]
            ae = abs(sarimax_val - actual_val)
            ape = ae / actual_val * 100
            comparison_rows.append([region, year, actual_val, sarimax_val, ae, ape])

forecast_df = pd.DataFrame(all_forecasts, columns=["Region", "Year", "SARIMAX_Forecast"])
comparison_df = pd.DataFrame(comparison_rows, columns=["Region", "Year", "Actual", "SARIMAX_Forecast", "AE", "APE"])
comparison_df = comparison_df.drop_duplicates(subset=["Region", "Year"])

#Calculate overall metrics
overall_mae = comparison_df['AE'].mean()
overall_mape = comparison_df['APE'].mean()

#Linear regression drivers DataFrame
driver_rows = []
for region, lr_model in driver_results.items():
    for var in lr_model.params.index:
        driver_rows.append([region, var, lr_model.params[var], lr_model.pvalues[var]])
driver_df = pd.DataFrame(driver_rows, columns=["Region", "Driver", "Coefficient", "P-value"])

#Flask API endpoints
@app.route("/api/drivers")
def api_drivers():
    clean_records = driver_df.where(pd.notnull(driver_df), None).to_dict(orient="records")
    
    for row in clean_records:
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v):
                row[k] = None
    
    return jsonify(clean_records)

@app.route("/api/forecasts")
def api_forecasts():
    return jsonify(forecast_df.to_dict(orient="records"))

@app.route("/api/comparison")
def api_comparison():
    return jsonify(comparison_df.to_dict(orient="records"))

@app.route("/api/regions")
def api_regions():
    return jsonify(df["Region"].unique().tolist() if "Region" in df.columns else [])

@app.route("/api/overview")
def api_overview():
    return jsonify({
        "overall_mae": float(overall_mae),
        "overall_mape": float(overall_mape),
        "total_regions": len(df["Region"].unique()),
        "forecast_years": forecast_years
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
