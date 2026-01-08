import streamlit as st
import pickle
import numpy as np
from supabase import create_client

# Clear cache to ensure fresh assets are loaded
st.cache_resource.clear()

# --- 1. SUPABASE SETUP ---
URL = "https://ixwvplxnfdjbmdsvdpu.supabase.co"
KEY = "sb_publishable_666yE2Qkv09Y5NQ_QlQaEg_L8fneOgL"
supabase = create_client(URL, KEY)

# --- 2. SECURE AUTHENTICATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🛡️ Enterprise Security Portal")
    user_key = st.text_input("Enter License Key", type="password", help="Example: PREMIUM-BANK-2026")
    
    if st.button("Activate System"):
        try:
            # Cleaning the input to prevent matching errors
            clean_key = user_key.strip()
            res = supabase.table("licenses").select("*").eq("key_value", clean_key).eq("is_active", True).execute()
            
            if len(res.data) > 0:
                st.session_state.authenticated = True
                st.success("Authentication Successful!")
                st.rerun()
            else:
                st.error("Access Denied: Invalid or Expired License Key.")
        except Exception as e:
            st.error(f"Supabase Connection Error: {e}")
    st.stop()

# --- 3. AI ENGINE (Fixed for 3 Features) ---
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("model.pkl", "rb") as f: model = pickle.load(f)
        with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Critical Asset Error: {e}")
        return None, None

model, scaler = load_model_and_scaler()

if model and scaler:
    st.title("Customer AI Assessment Terminal")
    st.sidebar.success("License: ACTIVE ✅")
    
    # Strictly defining only the 3 expected features
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 95, 35)
        balance = st.number_input("Yearly Balance ($)", 0, 1000000, 250000)
    with col2:
        tenure = st.number_input("Relationship Tenure (Years)", 0, 50, 8)
        # Note: Day is collected but NOT used in the calculation to avoid error
        _ = st.slider("Reference Day (System Info Only)", 1, 31, 15)

    if st.button("Generate AI Decision"):
        try:
            # This is the exact shape the Scaler expects
            final_input = np.array([[age, balance, tenure]])
            
            # Transformation and Prediction
            scaled_features = scaler.transform(final_input)
            prediction = model.predict(scaled_features)
            probabilities = model.predict_proba(scaled_features)[0]
            confidence = max(probabilities) * 100
            
            # Displaying Confidence Metric
            st.markdown("---")
            if prediction[0] == 1:
                st.success(f"Result: ELIGIBLE | Confidence: {confidence:.2f}%")
            else:
                st.warning(f"Result: NOT ELIGIBLE | Confidence: {confidence:.2f}%")
            
            st.progress(confidence / 100) #
            
        except Exception as e:
            st.error(f"System Error: {e}")
            st.info("Technical Note: Ensure model.pkl expects exactly 3 features.")
