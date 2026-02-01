# Graduate-Admission-Prediction-System
This is a easy prediction system for calculating the chances of admission for masters based on previous years experience
A machine learning web application built with Streamlit that predicts a student’s probability of getting into a graduate program based on academic metrics.
Project Overview
This project uses a Random Forest Regressor to analyze various factors such as GRE scores, TOEFL scores, and CGPA to estimate the “Chance of Admit” for prospective graduate students.
Tech Stack
•	Web Framework: Streamlit
•	Machine Learning: Scikit-learn (Random Forest Regressor)
•	Data Handling: Pandas, NumPy
•	Dataset: Admission_Predict.csv (400 records)
Features Analyzed
1.	GRE Score: Range 260–340
2.	TOEFL Score: Range 80–120
3.	University Rating: Scale of 1–5
4.	SOP/LOR Strength: Scale of 1–5
5.	CGPA: Scale of 1–10 (Input as 4.0 scale in UI)
6.	Research Experience: Binary (0 or 1)
How to Run Locally
1.	Clone this repository
2.	Install the required dependencies
3.	Run the Streamlit app
