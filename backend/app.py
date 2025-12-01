import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

driver_df = pd.read_csv("driver_df.csv") if os.path.exists("driver_df.csv") else pd.DataFrame()
forecast_df = pd.read_csv("forecast_df.csv") if os.path.exists("forecast_df.csv") else pd.DataFrame()
comparison_df = pd.read_csv("comparison_df.csv") if os.path.exists("comparison_df.csv") else pd.DataFrame()

df = pd.DataFrame()
if not driver_df.empty:
    df["Region"] = driver_df["Region"]

@app.route("/api/drivers")
def api_drivers():
    return jsonify(driver_df.to_dict(orient="records"))

@app.route("/api/forecasts")
def api_forecasts():
    return jsonify(forecast_df.to_dict(orient="records"))

@app.route("/api/comparison")
def api_comparison():
    return jsonify(comparison_df.to_dict(orient="records"))

@app.route("/api/regions")
def api_regions():
    if "Region" in df.columns:
        return jsonify(df["Region"].unique().tolist())
    return jsonify([])

@app.route("/api/overview")
def api_overview():
    if comparison_df.empty or df.empty:
        return jsonify({})
    return jsonify({
        "overall_mae": float(comparison_df['AE'].mean()),
        "overall_mape": float(comparison_df['APE'].mean()),
        "total_regions": len(df["Region"].unique()),
        "forecast_years": [2021, 2022, 2023, 2024, 2025]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)