from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "daily_combined.csv"
MODELS_DIR = APP_DIR / "models"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.joblib"

MODEL_PATHS = {
    "Bus": MODELS_DIR / "rf_model_bus.joblib",
    "Train": MODELS_DIR / "rf_model_train.joblib",
    "Ferry": MODELS_DIR / "rf_model_ferry.joblib",
    "Cycling": MODELS_DIR / "rf_model_total_cyclists.joblib",
}

DAY_MAPPING = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}

RF_RESULTS = pd.DataFrame(
    {
        "Target": ["Bus", "Train", "Ferry", "Cycling"],
        "MAE": [17793.24, 6283.83, 815.46, 2025.63],
        "RMSE": [27730.33, 8902.30, 1134.83, 2734.27],
        "R²": [0.8398, 0.8032, 0.7568, 0.6594],
    }
)

LR_RESULTS = pd.DataFrame(
    {
        "Target": ["Bus", "Train", "Ferry", "Cycling"],
        "R²": [0.7533, 0.6885, 0.7338, 0.6062],
        "Model": ["Linear Regression"] * 4,
    }
)
RF_R2_RESULTS = RF_RESULTS[["Target", "R²"]].assign(Model="Random Forest")


st.set_page_config(
    page_title="Auckland Green Travel Demand Predictor",
    page_icon="🚲",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the cleaned daily dataset prepared by the data and modelling team."""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["Day"].map(DAY_MAPPING)
    df["is_weekend"] = df["is_weekend"].astype(int)
    return df


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Dict[str, object], List[str]]:
    """Load Random Forest models and the feature order used during training."""
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    return models, feature_cols


def make_prediction_input(
    selected_date,
    is_holiday: int,
    temp_mean: float,
    precipitation: float,
    rain_hours: float,
    windspeed_max: float,
    petrol_price_nzd_cl: float,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Create a one-row feature table in exactly the same format as the training data."""
    date = pd.to_datetime(selected_date)
    day_of_week = int(date.dayofweek)  # Monday=0, Sunday=6
    is_weekend = int(day_of_week >= 5)

    row = pd.DataFrame(
        [
            {
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": int(is_holiday),
                "temp_mean": float(temp_mean),
                "precipitation": float(precipitation),
                "rain_hours": float(rain_hours),
                "windspeed_max": float(windspeed_max),
                "petrol_price_nzd_cl": float(petrol_price_nzd_cl),
            }
        ]
    )

    return row.reindex(columns=feature_cols)


def predict_all_modes(input_df: pd.DataFrame, models: Dict[str, object]) -> pd.DataFrame:
    """Run all four trained models on the same user input."""
    rows = []
    for mode, model in models.items():
        prediction = float(model.predict(input_df)[0])
        rows.append({"Mode": mode, "Predicted daily demand": max(0, round(prediction))})
    return pd.DataFrame(rows)


def get_historical_average_by_mode(data: pd.DataFrame, selected_date) -> pd.DataFrame:
    """Return previous-year average demand for each travel mode.

    If the selected year is the earliest year in the dataset, the function falls back
    to the full historical average so the comparison line is always available.
    """
    selected_year = pd.to_datetime(selected_date).year
    historical_data = data[data["date"].dt.year < selected_year]

    if historical_data.empty:
        historical_data = data
        label = "Historical average"
    else:
        label = f"Previous-year average before {selected_year}"

    averages = {
        "Bus": historical_data["Bus"].mean(),
        "Train": historical_data["Train"].mean(),
        "Ferry": historical_data["Ferry"].mean(),
        "Cycling": historical_data["total_cyclists"].mean(),
    }

    return pd.DataFrame(
        {
            "Mode": list(averages.keys()),
            "Previous-year average": [round(value) for value in averages.values()],
            "Average label": label,
        }
    )


def format_number(value: float | int) -> str:
    return f"{int(round(value)):,}"


df = load_data()
models, feature_cols = load_models()

st.sidebar.title("🚲 Green Travel App")
page = st.sidebar.radio(
    "Navigation",
    [
        "1. Project Overview",
        "2. Data Exploration",
        "3. Prediction Demo",
        "4. Model Performance",
        "5. Demo Script",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Auckland daily public transport, cycling, weather, fuel price and holiday data.")


if page == "1. Project Overview":
    st.title("Auckland Green Travel Demand Predictor")
    st.markdown(
        """
        This app demonstrates a machine learning project that predicts daily demand for four green travel modes in Auckland:
        **bus, train, ferry and cycling**.

        The project combines daily transport demand with weather, fuel price and public holiday information. The app is designed
        for a short executive-style demo: users can explore historical patterns and test how weather or calendar conditions may
        affect predicted travel demand.
        """
    )

    min_date = df["date"].min().strftime("%d %b %Y")
    max_date = df["date"].max().strftime("%d %b %Y")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily records", format_number(len(df)))
    c2.metric("Date range", f"{min_date} – {max_date}")
    c3.metric("Models", "4 Random Forest")
    c4.metric("Best R²", "0.8398 Bus")

    st.subheader("What decision-makers can use it for")
    st.markdown(
        """
        - Understand weekday, weekend and public-holiday effects on green travel demand.
        - Compare how weather conditions affect public transport and cycling differently.
        - Simulate likely travel demand under different weather and fuel-price scenarios.
        - Support operational planning, such as service capacity, station resourcing and cycling infrastructure decisions.
        """
    )

    st.subheader("Data preview by year")
    preview_cols = [
        "date",
        "Day",
        "Bus",
        "Train",
        "Ferry",
        "total_cyclists",
        "temp_mean",
        "precipitation",
        "is_holiday",
        "is_weekend",
    ]
    preview_df = df[preview_cols].copy()
    preview_df["Year"] = preview_df["date"].dt.year
    preview_df = (
        preview_df
        .sort_values("date")
        .groupby("Year", group_keys=False)
        .head(5)
        .drop(columns="Year")
        .reset_index(drop=True)
    )
    preview_df.index = np.arange(1, len(preview_df) + 1)
    preview_df.index.name = "No."
    st.caption("Showing five sample records from each available year so the preview includes different years of the dataset.")
    st.dataframe(preview_df, use_container_width=True)


elif page == "2. Data Exploration":
    st.title("Data Exploration")

    target_options = {
        "Bus": "Bus",
        "Train": "Train",
        "Ferry": "Ferry",
        "Cycling": "total_cyclists",
    }
    selected_mode = st.selectbox("Choose a travel mode", list(target_options.keys()))
    target_col = target_options[selected_mode]

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"Daily {selected_mode} demand over time")
        daily_fig = px.line(
            df,
            x="date",
            y=target_col,
            title=f"Daily {selected_mode} demand",
        )
        st.plotly_chart(daily_fig, use_container_width=True)

    with c2:
        st.subheader("Summary")
        st.metric("Average", format_number(df[target_col].mean()))
        st.metric("Minimum", format_number(df[target_col].min()))
        st.metric("Maximum", format_number(df[target_col].max()))
        st.metric("Standard deviation", format_number(df[target_col].std()))

    st.subheader("14-day rolling average")
    rolling_df = df[["date", "Bus", "Train", "Ferry", "total_cyclists"]].copy()
    rolling_df["Cycling"] = rolling_df["total_cyclists"]
    rolling_df = rolling_df.drop(columns=["total_cyclists"])
    for col in ["Bus", "Train", "Ferry", "Cycling"]:
        rolling_df[col] = rolling_df[col].rolling(14, min_periods=1).mean()
    rolling_long = rolling_df.melt(id_vars="date", var_name="Mode", value_name="14-day rolling average")
    rolling_fig = px.line(
        rolling_long,
        x="date",
        y="14-day rolling average",
        color="Mode",
        title="Smoothed demand trend by travel mode",
    )
    st.plotly_chart(rolling_fig, use_container_width=True)

    st.subheader("Weekday vs weekend demand")
    weekend_df = (
        df.groupby("is_weekend")[["Bus", "Train", "Ferry", "total_cyclists"]]
        .mean()
        .rename(index={0: "Weekday", 1: "Weekend"})
        .rename(columns={"total_cyclists": "Cycling"})
        .reset_index(names="Day type")
    )
    weekend_long = weekend_df.melt(id_vars="Day type", var_name="Mode", value_name="Average demand")
    weekend_fig = px.bar(
        weekend_long,
        x="Mode",
        y="Average demand",
        color="Day type",
        barmode="group",
        title="Average demand: weekday vs weekend",
    )
    st.plotly_chart(weekend_fig, use_container_width=True)

    st.subheader("Weather relationship")
    weather_var = st.selectbox(
        "Choose a weather variable",
        ["temp_mean", "precipitation", "rain_hours", "windspeed_max"],
    )
    scatter_fig = px.scatter(
        df,
        x=weather_var,
        y=target_col,
        trendline="ols",
        title=f"{selected_mode} demand vs {weather_var}",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Correlation matrix")
    corr_cols = [
        "Bus",
        "Train",
        "Ferry",
        "total_cyclists",
        "temp_mean",
        "precipitation",
        "rain_hours",
        "windspeed_max",
        "petrol_price_nzd_cl",
        "is_holiday",
        "is_weekend",
    ]
    corr_df = df[corr_cols].rename(columns={"total_cyclists": "Cycling"}).corr().round(2)
    corr_fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation matrix")
    st.plotly_chart(corr_fig, use_container_width=True)


elif page == "3. Prediction Demo":
    st.title("Prediction Demo")
    st.markdown(
        "Use this page during the presentation. Change the date, weather and fuel price, then show the predicted daily demand for all four modes."
    )

    latest_row = df.sort_values("date").iloc[-1]

    c1, c2 = st.columns(2)
    with c1:
        selected_date = st.date_input("Date", value=pd.Timestamp("2026-06-03"))
        is_holiday = st.selectbox("Public holiday?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
        temp_mean = st.slider(
            "Mean temperature (°C)",
            min_value=0.0,
            max_value=30.0,
            value=float(round(df["temp_mean"].median(), 1)),
            step=0.1,
        )
        petrol_price = st.slider(
            "Regular petrol price (NZ cents/litre)",
            min_value=180.0,
            max_value=360.0,
            value=float(round(latest_row["petrol_price_nzd_cl"], 1)),
            step=0.1,
        )

    with c2:
        precipitation = st.slider(
            "Precipitation (mm)",
            min_value=0.0,
            max_value=80.0,
            value=0.0,
            step=0.1,
        )
        rain_hours = st.slider(
            "Rain hours", min_value=0.0, max_value=24.0, value=0.0, step=0.5
        )
        windspeed_max = st.slider(
            "Maximum wind speed (km/h)",
            min_value=0.0,
            max_value=90.0,
            value=float(round(df["windspeed_max"].median(), 1)),
            step=0.1,
        )
        scenario_name = st.text_input("Scenario label", value="Dry weekday planning scenario")

    input_df = make_prediction_input(
        selected_date=selected_date,
        is_holiday=is_holiday,
        temp_mean=temp_mean,
        precipitation=precipitation,
        rain_hours=rain_hours,
        windspeed_max=windspeed_max,
        petrol_price_nzd_cl=petrol_price,
        feature_cols=feature_cols,
    )

    st.subheader("Model input")
    display_input = input_df.copy()
    display_input["day_name"] = pd.to_datetime(selected_date).day_name()
    st.dataframe(display_input, use_container_width=True)

    if st.button("Predict demand", type="primary"):
        pred_df = predict_all_modes(input_df, models)

        st.subheader(f"Predicted demand: {scenario_name}")
        m1, m2, m3, m4 = st.columns(4)
        for col, (_, row) in zip([m1, m2, m3, m4], pred_df.iterrows()):
            col.metric(row["Mode"], format_number(row["Predicted daily demand"]))

        historical_avg_df = get_historical_average_by_mode(df, selected_date)
        comparison_df = pred_df.merge(historical_avg_df, on="Mode", how="left")

        pred_fig = go.Figure()
        pred_fig.add_trace(
            go.Bar(
                x=comparison_df["Mode"],
                y=comparison_df["Predicted daily demand"],
                text=comparison_df["Predicted daily demand"],
                texttemplate="%{text:,.0f}",
                textposition="outside",
                name="Predicted demand",
                hovertemplate="Mode=%{x}<br>Predicted demand=%{y:,.0f}<extra></extra>",
            )
        )
        pred_fig.add_trace(
            go.Scatter(
                x=comparison_df["Mode"],
                y=comparison_df["Previous-year average"],
                mode="lines+markers",
                name=comparison_df["Average label"].iloc[0],
                hovertemplate="Mode=%{x}<br>Previous-year average=%{y:,.0f}<extra></extra>",
            )
        )
        pred_fig.update_layout(
            title="Predicted daily demand compared with previous-year average",
            xaxis_title="Travel mode",
            yaxis_title="Daily demand",
            legend_title="Series",
            hovermode="x unified",
            margin=dict(t=70),
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        with st.expander("Show prediction vs previous-year average values"):
            table_df = comparison_df.copy()
            table_df["Difference from average"] = (
                table_df["Predicted daily demand"] - table_df["Previous-year average"]
            )
            st.dataframe(table_df, use_container_width=True)

        st.info(
            "Demo interpretation: compare scenarios rather than treating one prediction as a perfect forecast. The model is strongest for Bus and Train, and less certain for Cycling."
        )


elif page == "4. Model Performance":
    st.title("Model Performance")
    st.markdown(
        "The final deployment uses Random Forest models because they outperformed the Linear Regression baseline for all four travel modes."
    )

    st.subheader("Random Forest evaluation results")
    st.dataframe(RF_RESULTS, use_container_width=True)

    comparison = pd.concat([LR_RESULTS, RF_R2_RESULTS], ignore_index=True)
    fig = px.bar(
        comparison,
        x="Target",
        y="R²",
        color="Model",
        barmode="group",
        title="R² comparison: Linear Regression vs Random Forest",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance")
    mode = st.selectbox("Choose model", list(models.keys()))
    model = models[mode]
    importance_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)
    imp_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Feature importance for {mode} model",
    )
    st.plotly_chart(imp_fig, use_container_width=True)

    st.markdown(
        """
        **Key message for presentation:** day-of-week and weekend effects are especially important for public transport demand. Weather variables are more relevant when explaining cycling demand.
        """
    )


elif page == "5. Demo Script":
    st.title("Demo Script for Presentation")
    st.markdown(
        """
        Use this page as your speaking guide during the group presentation.

        ### 1. Introduce the app
        “This app predicts daily green travel demand in Auckland across bus, train, ferry and cycling. It combines historical demand, weather, fuel price and holiday information.”

        ### 2. Show historical patterns
        Go to **Data Exploration** and show that public transport demand is higher on weekdays, while cycling is more weather-sensitive.

        ### 3. Show model performance
        Go to **Model Performance** and explain that Random Forest performed better than Linear Regression. Emphasise that Bus and Train predictions are the strongest.

        ### 4. Run two prediction scenarios
        Go to **Prediction Demo** and run:
        - Scenario A: dry weekday, no public holiday, low rain.
        - Scenario B: wet weekend or public holiday, higher rain and wind.

        ### 5. Business insight
        “The model can help transport planners test demand scenarios and prepare capacity or operational decisions, especially around commuting days, holidays and bad weather.”

        ### 6. Limitations
        “The app is a decision-support prototype. It does not include service disruptions, special events, school terms or real-time weather forecasts, so predictions should be interpreted as scenario estimates.”
        """
    )
