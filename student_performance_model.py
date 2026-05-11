import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split


data = pd.read_csv('Students_Performance_Synthetic_5000.csv')

FEATURES = ['Study_Hours_per_Week', 'Attendance (%)',
            'Midterm_Score', 'Assignments_Avg', 'Quizzes_Avg']
TARGET   = 'Final_Score'

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly   = PolynomialFeatures(degree=2, include_bias=True)
scaler = StandardScaler()

X_train_poly   = poly.fit_transform(X_train)
X_test_poly    = poly.transform(X_test)          # use fit from train only

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled  = scaler.transform(X_test_poly)   # use fit from train only


model = LinearRegression()
model.fit(X_train_scaled, y_train)
predicted_scores = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, predicted_scores)
mse = mean_squared_error(y_test, predicted_scores)
r2  = r2_score(y_test, predicted_scores)

print("=" * 40)
print("       MODEL PERFORMANCE METRICS")
print("=" * 40)
print(f"  Mean Absolute Error (MAE) : {mae:.2f}")
print(f"  Mean Squared Error  (MSE) : {mse:.2f}")
print(f"  R² Score                  : {r2:.4f}")
print("=" * 40)


plt.figure(figsize=(10, 6))
plt.hist(data[TARGET], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Final Exam Scores')
plt.xlabel('Final Score')
plt.ylabel('Number of Students')
plt.grid(True)
# NOTE: no plt.legend() here — histogram has no labelled artists
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
idx = range(len(y_test))
plt.scatter(idx, y_test,          color='blue', s=10, label='Actual Final Scores')
plt.plot(idx,   predicted_scores, color='red',  lw=1, label='Predicted Final Scores')
plt.title('Actual vs Predicted Final Scores')
plt.xlabel('Sample Index')
plt.ylabel('Final Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Predict Final Score for a New Student ---")
new_study_hours    = float(input("Enter Study Hours per Week (5–30)     : "))
new_attendance     = float(input("Enter Attendance %       (50–100)    : "))
new_midterm_score  = float(input("Enter Midterm Score      (40–100)    : "))
new_assignments    = float(input("Enter Assignments Avg    (50–100)    : "))
new_quizzes        = float(input("Enter Quizzes Avg        (50–100)    : "))

new_data = np.array([[new_study_hours, new_attendance,
                       new_midterm_score, new_assignments, new_quizzes]])

new_poly   = poly.transform(new_data)    # reuse fitted poly
new_scaled = scaler.transform(new_poly)  # reuse fitted scaler

predicted_final = model.predict(new_scaled)[0]
print(f"\n  Predicted Final Score : {predicted_final:.2f}")
