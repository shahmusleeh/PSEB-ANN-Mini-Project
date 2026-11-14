import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Configure Streamlit page attributes
st.set_page_config(
    page_title="Bank Customer Churn Risk Portal",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model_artifact():
    """Load and cache model pipeline, threshold, and feature structures."""
    model_path = Path("models/churn_pipeline.joblib")
    if not model_path.exists():
        return None
    return joblib.load(model_path)

artifact = load_model_artifact()

st.title("🏦 Bank Customer Churn Risk Portal")
st.markdown("""
This operational decision support tool predicts whether a bank customer is at risk of churn (**Exited**).
It leverages an optimized Decision Tree model trained with class-balancing weights to emphasize high recall.
""")

if artifact is None:
    st.error("⚠️ Model artifact not found at `models/churn_pipeline.joblib`. Please run your training notebook pipeline first.")
else:
    pipeline = artifact['pipeline']
    threshold = artifact['threshold']
    features = artifact['features']
    categories = artifact['categories']
    numeric_cols = artifact['training_ranges'].keys()
    medians = artifact['training_medians']
    ranges = artifact['training_ranges']

    # --- Streamlit Sidebar Menu ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["Single Customer Evaluation", "Batch File Processing", "Model Specifications Summary"])

    if app_mode == "Single Customer Evaluation":
        st.header("👤 Single Customer Risk Profiler")
        st.markdown("Enter attributes below to retrieve an on-demand risk assessment.")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
            geography = st.selectbox("Geography", categories['Geography'])
            gender = st.selectbox("Gender", categories['Gender'])
        
        with col2:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=38, step=1)
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5, step=1)
            balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)

        with col3:
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2, step=1)
            has_cr_card = st.selectbox("Has Credit Card?", ["1", "0"], format_func=lambda x: "Yes" if x == "1" else "No")
            is_active_member = st.selectbox("Is Active Member?", ["1", "0"], format_func=lambda x: "Yes" if x == "1" else "No")
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)

        if st.button("Run Risk Assessment"):
            # Assemble row matching exact model structures
            input_df = pd.DataFrame([{
                'CreditScore': credit_score,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'EstimatedSalary': estimated_salary,
                'Geography': geography,
                'Gender': gender,
                'HasCrCard': int(has_cr_card),
                'IsActiveMember': int(is_active_member)
            }])

            # Execute prediction pipeline
            score = pipeline.predict_proba(input_df)[0, 1]
            flagged = "At Risk (High Churn Probability)" if score >= threshold else "Stable (Low Churn Probability)"
            
            st.markdown("--- ")
            st.subheader("📊 Risk Assessment Summary")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Churn Score", f"{score:.1%}")
            metric_col2.metric("Classification Status", flagged)
            metric_col3.metric("Operational Threshold Target", f"{threshold:.1%}")

            # Provide targeted recommendations based on primary features
            st.subheader("📋 Prescriptive Account Recommendations")
            recs = []
            if score >= threshold:
                if age > 40:
                    recs.append("**Tailored Wealth Advisory Plan:** Customer is in a mature age bracket. Introduce them to exclusive wealth preservation advisory services.")
                if num_products == 1:
                    recs.append("**Cross-Selling Offer:** Customer holds only 1 active product. Introduce targeted cross-selling offers (e.g. specialized credit cards or premium savings accounts) to improve product stickiness.")
                if int(is_active_member) == 0:
                    recs.append("**Re-engagement Initiative:** Customer is currently marked inactive. Launch localized marketing outreach or personal relationship manager contact.")
                if len(recs) == 0:
                    recs.append("**General Retention Review:** Plan a routine relationship call to ensure service satisfaction.")
            else:
                recs.append("**Proactive Nurturing:** Customer is stable. Continue baseline loyalty rewards tracking and service support.")

            for idx, r in enumerate(recs, 1):
                st.write(f"{idx}. {r}")

    elif app_mode == "Batch File Processing":
        st.header("📁 Batch File Processing")
        st.markdown("Upload a CSV file matching the schema of `Churn_Modelling.csv` to score and classify multiple accounts in bulk.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                batch_raw = pd.read_csv(uploaded_file)
                
                # Validate that necessary features exist in the uploaded batch
                missing_cols = [c for c in features if c not in batch_raw.columns]
                if missing_cols:
                    st.error(f"❌ Missing columns in file: {', '.join(missing_cols)}")
                else:
                    batch_clean = batch_raw[features].copy()
                    
                    # Compute scores
                    scores = pipeline.predict_proba(batch_clean)[:, 1]
                    predictions = (scores >= threshold).astype(int)
                    
                    # Append results to raw data
                    results_df = batch_raw.copy()
                    results_df['Churn_Score'] = scores
                    results_df['Predicted_Exited'] = predictions
                    
                    st.success(f"✅ Successfully processed {len(results_df)} customer accounts!")
                    
                    # Summary statistics
                    flagged_pct = predictions.mean()
                    st.metric("Flagged accounts percentage", f"{flagged_pct:.1%}")
                    
                    st.dataframe(results_df[['CustomerId', 'Surname', 'Geography', 'Age', 'Churn_Score', 'Predicted_Exited']].head(100))
                    
                    # Download predictions link
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Prediction File",
                        data=csv,
                        file_name="batch_churn_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing batch file: {e}")

    elif app_mode == "Model Specifications Summary":
        st.header("⚙️ Model Parameters & Metadata")
        st.markdown("Below are the validation metrics and pipeline attributes recorded during training:")
        st.json({
            'Selected Classifier': artifact['model_name'],
            'Operational Decision Threshold': artifact['threshold'],
            'Generalization Test Recall': artifact['metrics']['recall'],
            'Generalization Test Precision': artifact['metrics']['precision'],
            'Evaluation Dataset Size': artifact['test_size'],
            'Variables Trained On': artifact['features']
        })
