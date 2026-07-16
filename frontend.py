"""
Streamlit real-time interface for the Child Malnutrition Risk Prediction system.
Run with: streamlit run frontend.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
from inference import MalnutritionPredictor

st.set_page_config(page_title="Child Malnutrition Risk Screening", page_icon="🩺", layout="wide")

@st.cache_resource
def load_predictor():
    return MalnutritionPredictor()

predictor = load_predictor()

st.title("🩺 Child Malnutrition Risk Screening")
st.caption(
    "Multi-task deep learning model predicting stunting, wasting, and underweight risk "
    "from household and demographic factors — with uncertainty estimation and explainability. "
    "Trained on DHS Children's Recode survey data using WHO Child Growth Standards."
)

with st.sidebar:
    st.header("Child & Household Information")
    age_months = st.slider("Child's age (months)", 0, 59, 18)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    wealth_index = st.select_slider(
        "Household wealth quintile", options=[1, 2, 3, 4, 5],
        value=1, format_func=lambda x: {1: "1 - Poorest", 2: "2 - Poor", 3: "3 - Middle",
                                          4: "4 - Richer", 5: "5 - Richest"}[x],
    )
    residence = st.radio("Residence type", ["Urban", "Rural"], horizontal=True)
    mother_education = st.select_slider(
        "Mother's education", options=[0, 1, 2, 3],
        format_func=lambda x: {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"}[x],
    )
    mother_age = st.slider("Mother's age (years)", 15, 49, 25)
    region = st.selectbox("Region", options=list(predictor.region_encoder.classes_))

    mc_samples = st.slider("MC Dropout samples (uncertainty precision)", 10, 100, 30, step=10)
    run = st.button("Run Real-Time Prediction", type="primary", use_container_width=True)

if run:
    record = {
        "age_months": age_months,
        "sex": 0 if sex == "Male" else 1,
        "wealth_index": wealth_index,
        "residence_type": 1 if residence == "Urban" else 2,
        "mother_education": mother_education,
        "mother_age": mother_age,
        "region": region,
    }

    with st.spinner("Running model inference..."):
        result = predictor.predict(record, mc_samples=mc_samples, explain=True)

    st.success("Prediction complete")

    cols = st.columns(3)
    severity_colors = {"normal": "#2ecc71", "moderate": "#f39c12", "severe": "#e74c3c"}
    labels = {"stunting": "Stunting (Height-for-Age)", "underweight": "Underweight (Weight-for-Age)",
              "wasting": "Wasting (Weight-for-Height)"}

    for col, key in zip(cols, ["stunting", "underweight", "wasting"]):
        r = result[key]
        with col:
            st.metric(
                labels[key],
                value=r["severity"].upper(),
                delta=f"z = {r['z_score_mean']} ± {r['z_score_std']}",
                delta_color="off",
            )
            severity_color = severity_colors[r["severity"]]
            severity_label = r["severity"].upper()
            st.markdown(
                f"<div style='background-color:{severity_color};"
                f"padding:8px;border-radius:6px;text-align:center;color:white;font-weight:bold'>"
                f"{severity_label}</div>", unsafe_allow_html=True,
            )
            st.caption(f"MC Dropout confidence: {r['mc_dropout_confidence']*100:.0f}%")
            if not r["classifier_cross_check"]["agrees_with_regression"]:
                st.caption("⚠️ Classifier head disagrees with regression estimate — treat with extra caution.")

    st.divider()
    st.subheader("Why this prediction? (SHAP feature contributions)")
    st.caption("Positive values push the risk (worse z-score) up; negative values push it down (better).")

    explain_cols = st.columns(3)
    for col, key in zip(explain_cols, ["stunting", "underweight", "wasting"]):
        expl = result["explanations"][key]
        sorted_feats = sorted(expl.items(), key=lambda x: abs(x[1]), reverse=True)
        names = [f[0] for f in sorted_feats]
        vals = [f[1] for f in sorted_feats]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h", marker_color=colors,
        ))
        fig.update_layout(
            title=labels[key], height=280, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="SHAP contribution",
        )
        with col:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption(
        "⚕️ This tool is a research/screening aid trained on survey data and does not replace "
        "clinical assessment. Predictions are based on demographic/household risk factors, "
        "not direct anthropometric measurement of this specific child."
    )
else:
    st.info("Enter the child's information in the sidebar and click **Run Real-Time Prediction**.")
