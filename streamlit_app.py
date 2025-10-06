import streamlit as st
import pandas as pd
import joblib
import json

@st.cache_resource
def load_model():
    model = joblib.load("../models/rf_churn.joblib")
    with open("../models/model_metadata.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

st.title("Telco Customer Churn Prediction")

st.markdown("## Single Customer Prediction")
numeric_cols = meta['numeric_cols']
categorical_cols = meta['categorical_cols']

with st.form("single_form"):
    input_data = {}
    for c in numeric_cols:
        v = st.number_input(c, value=0.0)
        input_data[c] = v
    for c in categorical_cols:
        val = st.text_input(c, value="")
        input_data[c] = val
    submitted = st.form_submit_button("Predict")
    if submitted:
        df_single = pd.DataFrame([input_data], columns=meta['X_columns'])
        for n in numeric_cols:
            df_single[n] = pd.to_numeric(df_single[n], errors='coerce').fillna(0)
        prob = model.predict_proba(df_single)[:,1][0]
        pred = model.predict(df_single)[0]
        st.write(f"Predicted churn: **{'Yes' if pred==1 else 'No'}** — probability {prob:.3f}")

st.markdown("---")
st.markdown("## Batch Predictions")
uploaded = st.file_uploader("Upload CSV with same feature columns", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df_req = df[meta['X_columns']].copy()
    for n in numeric_cols:
        df_req[n] = pd.to_numeric(df_req[n], errors='coerce').fillna(0)
    probs = model.predict_proba(df_req)[:,1]
    df['churn_prob'] = probs
    st.dataframe(df.head(50))
    st.download_button("Download results CSV", df.to_csv(index=False), file_name="predictions.csv")
