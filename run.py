import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\asus\OneDrive\Desktop\Admission prediction\Admission_Predict.csv")

# Drop unwanted columns if any (like Serial No.)
df = df.drop(columns=['Serial No.'], errors='ignore')

# Check missing values
print(df.isnull().sum())

# Features and target
X = df.drop("Chance of Admit ", axis=1)  # Features
y = df["Chance of Admit "]        

df['CGPA'] = df['CGPA'] / 10 * 4

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel("Actual Chance")
plt.ylabel("Predicted Chance")
plt.title("Graduate Admission Prediction")
plt.show()
sample_student = np.array([[320, 110, 4, 4.5, 4.0, 9.0, 1]])
# GRE, TOEFL, UnivRating, SOP, LOR, CGPA, Research
print("Predicted Chance of Admission:", rf.predict(sample_student.reshape(1, -1))[0])
