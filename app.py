import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\asus\OneDrive\Desktop\Admission prediction\Admission_Predict.csv")
    df = df.drop(columns=['Serial No.'], errors='ignore')
    return df

df = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = df.drop("Chance of Admit ", axis=1)
y = df["Chance of Admit "]
df['CGPA'] = df['CGPA'] / 10 * 4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎓 Graduate Admission Prediction")
st.write("Enter your academic details to estimate your chance of admission.")

# Input fields
gre = st.slider("GRE Score", 260, 340, 320)
toefl = st.slider("TOEFL Score", 80, 120, 110)
univ_rating = st.slider("University Rating (1-5)", 1, 5, 3)
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, 3.5, 0.5)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0, 0.5)
cgpa = st.number_input("CGPA (out of 4.0)", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
research = st.selectbox("Research Experience", [0, 1])

# Prediction
if st.button("Predict Admission Chance"):
    student_data = np.array([[gre, toefl, univ_rating, sop, lor, cgpa, research]])
    prediction = model.predict(student_data)[0]
    st.success(f"Estimated Chance of Admission: {prediction*100:.2f}%")

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("📊 Model Performance on Test Data")
y_pred = model.predict(X_test)
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Show dataset preview
with st.expander("🔍 See Dataset Preview"):
    st.dataframe(df.head())
