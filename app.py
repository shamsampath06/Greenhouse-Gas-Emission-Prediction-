import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from utils.preprocessor import preprocess_input

# Load model and scaler
model = joblib.load("E:/Greenhouse Gas Emission Prediction/LR_model.pkl")
scaler = joblib.load("E:/Greenhouse Gas Emission Prediction/scaler.pkl")

st.set_page_config(page_title="GHG Emission Prediction", page_icon="🌱", layout="wide")

# Sidebar title & menu
st.sidebar.title("📊 Prediction App")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📈 Visualizations", "🤖 Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("Upload your dataset:")
st.sidebar.file_uploader("Drag & drop file here", type=['csv'])

# ---------------------------
if menu == "🏠 Home":
    st.title("🌍 Supply Chain Emissions Prediction App")
    st.markdown("""
    Welcome to the Greenhouse Gas Emission Prediction App!  
    This tool helps you estimate **Supply Chain Emission Factors with Margins**  
    using machine learning and real-world parameters.

    ✅ **What you can do:**  
    - Predict emissions for a product or industry
    - Visualize data relationships
    - Learn how data quality affects emissions
    """)

# ---------------------------
elif menu == "📈 Visualizations":
    st.title("📊 Visualizations & Correlations")
    st.markdown("""
    Explore how **Supply Chain Emission Factors** relate to margins and data quality.
    (Example demo - replace with real data if you have)
    """)

    # Example demo data to plot
    df = pd.DataFrame({
        "Margin": [0.1, 0.2, 0.3, 0.4, 0.5],
        "PredictedEmission": [1.5, 1.8, 2.0, 2.4, 2.6],
        "DQ Reliability": [0.9, 0.8, 0.85, 0.7, 0.65]
    })

    fig = px.scatter(df, x="Margin", y="PredictedEmission", size="DQ Reliability",
                     title="Predicted Emission vs. Margin", color="DQ Reliability")
    st.plotly_chart(fig)

    st.markdown("👉 Use real data to explore deeper insights!")

# ---------------------------
elif menu == "🤖 Prediction":
    st.title("🤖 Predict Supply Chain Emission Factor")

    with st.form("prediction_form"):
        substance = st.selectbox(
            "Substance",
            ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'],
            help="Choose the greenhouse gas to analyze."
        )
        unit = st.selectbox(
            "Unit",
            ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'],
            help="Standardizes emissions across gases."
        )
        source = st.selectbox(
            "Source",
            ['Commodity', 'Industry'],
            help="Commodity: single product; Industry: whole sector."
        )
        supply_wo_margin = st.number_input(
            "Supply Chain Emission Factors without Margins",
            min_value=0.0,
            help="Base emission factor before adding margin."
        )
        margin = st.number_input(
            "Margins of Supply Chain Emission Factors",
            min_value=0.0,
            help="Buffer for uncertainty."
        )
        dq_reliability = st.slider("DQ Reliability", 0.0, 1.0)
        dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0)
        dq_geo = st.slider("DQ Geographical Correlation", 0.0, 1.0)
        dq_tech = st.slider("DQ Technological Correlation", 0.0, 1.0)
        dq_data = st.slider("DQ Data Collection", 0.0, 1.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = {
            'Substance': substance,
            'Unit': unit,
            'Supply Chain Emission Factors without Margins': supply_wo_margin,
            'Margins of Supply Chain Emission Factors': margin,
            'DQ ReliabilityScore of Factors without Margins': dq_reliability,
            'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
            'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
            'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
            'DQ DataCollection of Factors without Margins': dq_data,
            'Source': source
        }

        input_df = preprocess_input(pd.DataFrame([input_data]))
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        st.success(f"✅ Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**")
        st.markdown("""
        **What this means:**  
        Estimated GHG emissions per 2018 USD, considering margin and data quality.
        """)

# ---------------------------
st.sidebar.markdown("---")
st.sidebar.info("Built with 👾 using Streamlit")
