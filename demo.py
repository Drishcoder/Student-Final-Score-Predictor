import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Student Final Score Predictor",
    page_icon="🎓",
    layout="wide"

)

@st.cache_resource
def load_and_train_model():
    data = pd.read_csv('Students_Performance_Synthetic_5000.csv')
    Features = ["Study_Hours_per_Week","Attendance (%)","Midterm_Score","Assignments_Avg","Quizzes_Avg"]
    Target = "Final_Score"
    
    X = data[Features]
    y = data[Target]
    
    
     

