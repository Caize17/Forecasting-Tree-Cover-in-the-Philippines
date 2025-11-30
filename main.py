import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import statsmodels.api as sm

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Global variables to store data
df = None
driver_df = None
forecast_df = None
comparison_df = None

def prepare_data():
    """Main data preparation and modeling function"""
    global df, driver_df, forecast_df, comparison_df
    
    print("Loading data files...")
    # Load datasets
    tree = pd.read_csv("Annual_Tree_Cover_Analysis_ha.csv")
    agri = pd.read_csv("agriculture_ha.csv")
    log = pd.read_csv("log_production.csv")
    urban = pd.read_csv("urbanization_percentage.csv")
    roads = pd.read_csv("national_roads.csv")
    actual_tree = pd.read_csv("actual_tree_cover.csv")  # 2021-2024
    
    # Handle roads data - ensure it has Region column
    if "Region" not in roads.columns:
        all_regions = tree["Region"].unique()
        roads = roads.assign(key=1).merge(
            pd.DataFrame({"Region": all_regions, "key": 1}), 
            on="key"
        ).drop("key", axis=1)
    
    # Merge all datasets
    print("Merging datasets...")
    df = tree.merge(agri, on=["Region", "Year"], how="left")
    df = df.merge(log, on=["Region", "Year"], how="left")
    df = df.merge(urban, on=["Region", "Year"], how="left")
    df = df.merge(roads, on=["Region", "Year"], how="left")
    
    # Rename columns for clarity
    df = df.rename(columns={
        "tree_cover_ha": "Tree_Cover",
        "extrapolated_urbanization_percentage": "Urbanization",
        "harvested_area_ha": "Agriculture",
        "log_production_cbm": "Logging",
        "Total_km": "Roads_km"
    })
    
    # Get unique regions
    regions = df["Region"].unique()
    forecast_years = [2021, 2022, 2023, 2024, 2025]
    
    comparison_rows = []
    driver_results = {}
    sarimax_results = {}
    all_forecasts = []
    
    print(f"Processing {len(regions)} regions...")
    
    # Process each region
    for region in regions:
        reg_df = df[df["Region"] == region].sort_values("Year")
        exog_vars = ["Agriculture", "Logging", "Urbanization", "Roads_km"]
        
        # Linear Regression for driver importance
        X = sm.add_constant(reg_df[exog_vars])
        y = reg_df["Tree_Cover"]
        lr_model = sm.OLS(y, X).fit()
        driver_results[region] = lr_model
        
        # SARIMAX model for forecasting
        exog = reg_df[exog_vars]
        sarimax_model = sm.tsa.SARIMAX(
            reg_df["Tree_Cover"],
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            exog=exog
        )
        sarimax_fit = sarimax_model.fit(disp=False)
        sarimax_results[region] = sarimax_fit
        
        # Forecast exogenous variables
        forecast_exog = []
        for var in exog_vars:
            coef_var = np.polyfit(np.arange(len(reg_df)), reg_df[var].values, 1)
            forecast_exog.append(
                np.polyval(coef_var, np.arange(len(reg_df), len(reg_df) + len(forecast_years)))
            )
        forecast_exog = np.array(forecast_exog).T
        
        # Generate forecasts
        tree_forecast = sarimax_fit.get_forecast(steps=len(forecast_years), exog=forecast_exog)
        
        for i, year in enumerate(forecast_years):
            all_forecasts.append([
                region,
                year,
                tree_forecast.predicted_mean.iloc[i]
            ])
        
        # Compare with actual data (2021-2024)
        for i, year in enumerate(forecast_years):
            actual_row = actual_tree[
                (actual_tree["Region"] == region) & (actual_tree["Year"] == year)
            ]
            if not actual_row.empty:
                actual_val = actual_row["tree_cover_ha"].values[0]
                sarimax_val = tree_forecast.predicted_mean.iloc[i]
                ae = abs(sarimax_val - actual_val)
                ape = (ae / actual_val) * 100
                comparison_rows.append([region, year, actual_val, sarimax_val, ae, ape])
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame(all_forecasts, columns=[
        "Region", "Year", "SARIMAX_Forecast"
    ])
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_rows, columns=[
        "Region", "Year", "Actual", "SARIMAX_Forecast", "AE", "APE"
    ])
    comparison_df = comparison_df.drop_duplicates(subset=["Region", "Year"])
    
    # Calculate overall metrics
    overall_mae = comparison_df['AE'].mean()
    overall_mape = comparison_df['APE'].mean()
    
    print("\n=== SARIMAX Forecasts (2021-2025) ===")
    print(forecast_df.head(10))
    print(f"\n=== Forecast vs Actual Comparison ===")
    print(comparison_df.head(10))
    print(f"\nOverall MAE: {overall_mae:.2f}")
    print(f"Overall MAPE: {overall_mape:.2f}%")
    
    # Create driver importance dataframe
    driver_rows = []
    for region, lr_model in driver_results.items():
        for var in lr_model.params.index:
            driver_rows.append([
                region,
                var,
                lr_model.params[var],
                lr_model.pvalues[var]
            ])
    
    driver_df = pd.DataFrame(driver_rows, columns=[
        "Region", "Driver", "Coefficient", "P-value"
    ])
    
    print("\n=== Driver Importance (Linear Regression) ===")
    print(driver_df.head(10))
    print("\nData preparation complete!")

# API Endpoints
@app.route("/api/drivers")
def api_drivers():
    """Get driver importance data"""
    return jsonify(driver_df.to_dict(orient="records"))

@app.route("/api/forecasts")
def api_forecasts():
    """Get SARIMAX forecasts (2021-2025)"""
    return jsonify(forecast_df.to_dict(orient="records"))

@app.route("/api/comparison")
def api_comparison():
    """Get forecast vs actual comparison"""
    return jsonify(comparison_df.to_dict(orient="records"))

@app.route("/api/regions")
def api_regions():
    """Get list of all regions"""
    regions = df["Region"].unique().tolist()
    return jsonify(regions)

@app.route("/api/overview")
def api_overview():
    """Get overall statistics"""
    overall_mae = comparison_df['AE'].mean()
    overall_mape = comparison_df['APE'].mean()
    return jsonify({
        "overall_mae": float(overall_mae),
        "overall_mape": float(overall_mape),
        "total_regions": len(df["Region"].unique()),
        "forecast_years": [2021, 2022, 2023, 2024, 2025]
    })

if __name__ == "__main__":
    print("=" * 60)
    print("ForestCastPH Backend Server")
    print("=" * 60)
    prepare_data()
    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print("API will be available at: http://192.168.1.13:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)