import streamlit as st
import requests
import joblib

# ================= CONFIG =================
API_URL = "https://orange-goggles-r46wjrwqwx7v3xpxw-8000.app.github.dev/predict"

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("💻 Laptop Price Predictor")
st.caption("Powered by FastAPI + ML Model")

# ================= LOAD PREPROCESSOR =================
preprocessor = joblib.load("preprocessor.pkl")

# Extract OneHotEncoder
ohe = preprocessor.named_transformers_['cat']

# Extract categories safely
company_list = list(ohe.categories_[0])
type_list = list(ohe.categories_[1])
cpu_name_list = list(ohe.categories_[2])
cpu_brand_list = list(ohe.categories_[3])
gpu_list = list(ohe.categories_[4])
os_list = list(ohe.categories_[5])

# ================= UI =================
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Company", company_list)
        type_name = st.selectbox("Type", type_list)
        ram = st.number_input("RAM (GB)", 2, 64, 8)
        weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0)
        touchscreen = st.selectbox("Touchscreen", [0, 1])
        ips = st.selectbox("IPS Display", [0, 1])

    with col2:
        ppi = st.number_input("PPI", 90.0, 400.0, 141.0)
        cpu_name = st.selectbox("CPU Name", cpu_name_list)
        cpu_speed = st.number_input("CPU Speed (GHz)", 1.0, 5.0, 2.5)
        cpu_brand = st.selectbox("CPU Brand", cpu_brand_list)
        ssd = st.number_input("SSD (GB)", 0, 2000, 256)
        gpu = st.selectbox("GPU Brand", gpu_list)
        os = st.selectbox("Operating System", os_list)

    submit = st.form_submit_button("🔍 Predict Price")

# ================= API CALL =================
if submit:
    payload = {
        "Company": company,
        "TypeName": type_name,
        "Ram_GB": ram,
        "Weight": weight,
        "Touchscreen": touchscreen,
        "IPS": ips,
        "ppi": ppi,
        "Cpu_Name": cpu_name,
        "Cpu_Speed_GHz": cpu_speed,
        "Cpu_brand": cpu_brand,
        "SSD": ssd,
        "Gpu_Brand": gpu,
        "OS": os
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.subheader("💰 Predicted Laptop Price")
            st.success(f"₹ {result['predicted_price']:,.2f}")
        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")