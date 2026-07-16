"""
Child Malnutrition Risk Screening — full app with login gate + multi-section navigation.
Hugging Face Spaces / Streamlit Cloud entry point (app.py).
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
from inference import MalnutritionPredictor

st.set_page_config(page_title="Child Malnutrition Risk Screening", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Demo credentials (portfolio/demo project — no real user database)
# ---------------------------------------------------------------------------
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "unicef2026"

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background: linear-gradient(180deg, #f7faf9 0%, #ffffff 100%); }
    .hero-title {
        font-size: 2.6rem; font-weight: 800; color: #14532d;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.1rem; color: #4b5563; margin-bottom: 1.5rem; max-width: 700px;
    }
    .nav-card {
        background: white; border-radius: 14px; padding: 1.6rem;
        border: 1px solid #e5e7eb; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        height: 100%;
    }
    .nav-card h3 { margin-top: 0; color: #14532d; }
    .nav-card p { color: #6b7280; font-size: 0.92rem; }
    .login-box {
        max-width: 420px; margin: 3rem auto; padding: 2.2rem;
        background: white; border-radius: 16px; border: 1px solid #e5e7eb;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    }
    .badge {
        display: inline-block; background: #dcfce7; color: #166534;
        padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "welcome"


def go_to(page_name):
    st.session_state.page = page_name


# ---------------------------------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------------------------------
def render_login():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown("### 🩺 Child Malnutrition Risk Screening")
    st.caption("Please sign in to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

    if submitted:
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.page = "welcome"
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    st.info(f"**Demo access:** username `{DEMO_USERNAME}` · password `{DEMO_PASSWORD}`")
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# WELCOME / LANDING PAGE
# ---------------------------------------------------------------------------
def render_welcome():
    st.markdown('<div class="hero-title">🩺 Child Malnutrition Risk Screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">A real-time multi-task deep learning system that predicts '
        'stunting, wasting, and underweight risk in children under 5 — from household and '
        'demographic factors alone, with uncertainty estimation and explainability. Trained on '
        'DHS Bangladesh 2022 survey data using WHO Child Growth Standards.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="badge">Real-time inference</span> &nbsp; '
                '<span class="badge">MC Dropout uncertainty</span> &nbsp; '
                '<span class="badge">SHAP explainability</span>', unsafe_allow_html=True)
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="nav-card"><h3>🩺 Risk Screening Tool</h3>'
            '<p>Enter a child\'s household and demographic details to get an instant, '
            'explainable malnutrition risk prediction.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("Open Screening Tool →", key="btn_screening", use_container_width=True,
                   on_click=go_to, args=("screening",))

    with col2:
        st.markdown(
            '<div class="nav-card"><h3>📊 Model & Methodology</h3>'
            '<p>Architecture, training approach, performance metrics, and design decisions '
            'behind the model.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("View Methodology →", key="btn_methodology", use_container_width=True,
                   on_click=go_to, args=("methodology",))

    with col3:
        st.markdown(
            '<div class="nav-card"><h3>ℹ️ About This Project</h3>'
            '<p>Motivation, data source, limitations, and ethical considerations.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("Read More →", key="btn_about", use_container_width=True,
                   on_click=go_to, args=("about",))

    st.divider()
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.page = "welcome"
        st.rerun()


# ---------------------------------------------------------------------------
# SCREENING TOOL PAGE
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return MalnutritionPredictor()


def render_screening():
    if st.button("← Back to Home"):
        go_to("welcome")
        st.rerun()

    predictor = load_predictor()

    st.title("🩺 Child Malnutrition Risk Screening")
    st.caption(
        "Multi-task deep learning model predicting stunting, wasting, and underweight risk "
        "from household and demographic factors — with uncertainty estimation and explainability."
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

            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
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


# ---------------------------------------------------------------------------
# METHODOLOGY PAGE
# ---------------------------------------------------------------------------
def render_methodology():
    if st.button("← Back to Home"):
        go_to("welcome")
        st.rerun()

    st.title("📊 Model & Methodology")

    st.subheader("Architecture")
    st.markdown(
        "A single shared-trunk neural network with three task-specific heads — "
        "**stunting (HAZ)**, **wasting (WHZ)**, and **underweight (WAZ)** — reflecting the real "
        "clinical correlation between these conditions. Each head outputs a continuous z-score "
        "regression and a 3-class severity classification (normal / moderate / severe)."
    )

    st.subheader("Uncertainty & Explainability")
    st.markdown(
        "- **Monte Carlo Dropout** (~30 stochastic forward passes) produces a mean and standard "
        "deviation for every prediction, so the model communicates when it's uncertain.\n"
        "- **SHAP** (KernelExplainer) attributes each prediction to the input features that drove it."
    )

    st.subheader("Test Set Performance (n=614, Bangladesh DHS 2022)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stunting MAE", "0.94 SD")
    c2.metric("Underweight MAE", "0.81 SD")
    c3.metric("Wasting MAE", "0.87 SD")

    st.subheader("Design Decisions")
    st.markdown(
        "- **Severity is derived from the WHO z-score threshold**, not the classifier head — this "
        "guarantees the displayed severity always matches the displayed z-score. The classifier's "
        "vote is kept as an independent cross-check signal instead.\n"
        "- **Class-weighted loss** deliberately trades precision for recall on severe cases: a false "
        "positive costs a follow-up checkup, a false negative on a severely malnourished child costs "
        "far more."
    )


# ---------------------------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------------------------
def render_about():
    if st.button("← Back to Home"):
        go_to("welcome")
        st.rerun()

    st.title("ℹ️ About This Project")

    st.subheader("Motivation")
    st.markdown(
        "Community health workers often need to identify at-risk children before formal "
        "anthropometric screening is possible. This project asks whether a model can flag "
        "malnutrition risk using only the household and demographic information a health worker "
        "already has — age, sex, household wealth, region, residence, and mother's education/age — "
        "rather than requiring a direct physical measurement."
    )

    st.subheader("Data")
    st.markdown(
        "DHS (Demographic and Health Survey) Bangladesh 2022 Children's Recode data, using "
        "pre-computed WHO Child Growth Standards z-scores as ground truth."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Predicting from demographic proxies alone is inherently harder than using direct "
        "measurements — this is a screening/triage aid, not a diagnostic tool.\n"
        "- Regional and household-level factors here don't include some clinically important "
        "variables (recent illness, breastfeeding practices, dietary diversity).\n"
        "- This tool does not replace clinical assessment or physical measurement."
    )

    st.subheader("Ethical Considerations")
    st.markdown(
        "This project uses de-identified survey data and is intended as a research/portfolio "
        "project demonstrating applied ML methodology for humanitarian health screening, not as "
        "a deployed clinical system."
    )


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------
if not st.session_state.logged_in:
    render_login()
else:
    if st.session_state.page == "welcome":
        render_welcome()
    elif st.session_state.page == "screening":
        render_screening()
    elif st.session_state.page == "methodology":
        render_methodology()
    elif st.session_state.page == "about":
        render_about()
