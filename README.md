# 🎓 Student Score Predictor

A **Streamlit-based machine learning application** that predicts a student's final exam score based on various academic performance metrics using Polynomial Regression.

## 📋 Features

- **Interactive Sliders**: Adjust student parameters in real-time to see predicted score changes
- **Smart Insights**: Personalized recommendations based on current input values
- **Model Metrics**: View R² score, MAE, and MSE performance metrics
- **Dataset Overview**: Statistics about the training dataset (5,000 synthetic records)
- **Grade Badges**: Visual grade classification (Excellent, Good, Average, Below Average, Needs Improvement)
- **Progress Visualization**: Score progress bar with comparison to dataset average

## 🎯 Input Parameters

The predictor uses five key student metrics:

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Study Hours per Week** | 5 - 30 hours | Hours dedicated to studying outside class |
| **Attendance (%)** | 50 - 100% | Percentage of classes attended |
| **Midterm Score** | 40 - 100 | Score from midterm examination |
| **Assignments Average** | 50 - 100 | Average score across all assignments |
| **Quizzes Average** | 50 - 100 | Average score across in-class quizzes |

## 🤖 Machine Learning Model

- **Algorithm**: Polynomial Regression (Degree 2)
- **Features**: 5 student performance metrics
- **Training Data**: 5,000 synthetic student records
- **Train/Test Split**: 80% / 20%
- **Preprocessing**:
  - Polynomial Feature Expansion (degree=2)
  - Standard Scaling normalization

### Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | ~0.85 (varies based on training data) |
| **MAE** | ~3-5 points |
| **MSE** | ~15-25 points |

## 📦 Installation

### Requirements
```bash
Python 3.8+
```

### Dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### Setup
1. Clone or download the project
2. Navigate to the `advproject` directory
3. Ensure the dataset is available at:
   ```
   ../student_final_score_predictor/Students_Performance_Synthetic_5000.csv
   ```

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Output

### Predicted Final Score
- **Large numerical display** of the predicted score (0-100)
- **Grade badge** with color-coded classification
- **Comparison** to dataset average

### Smart Insights
- Provides targeted recommendations based on student parameters
- Highlights areas for improvement
- Alerts when key metrics are below optimal thresholds

### Dataset Overview Metrics
- **Total Students**: Number of records in training dataset
- **Average Final Score**: Mean final score across all students
- **Highest Score**: Maximum score achieved
- **Model R² Score**: Model accuracy metric

## 🎨 Design Features

- **Modern UI**: Clean, professional interface with card-based layout
- **Responsive Layout**: Two-column design (inputs | predictions)
- **Dark Text on Light Background**: Optimized for readability
- **Color-Coded Insights**: Visual categorization by performance area
- **Emoji Icons**: Visual indicators for different sections

## 📁 File Structure

```
advproject/
├── app.py                              # Main Streamlit application
├── README.md                           # This file
├── streamlitwebapp.py                  # Alternative implementation
├── student_performance_model.py        # Model training script
└── student_final_score_predictor/
    └── Students_Performance_Synthetic_5000.csv   # Training dataset
```

## 🔧 Technical Details

### Model Training Pipeline
1. Load synthetic student dataset (5,000 records)
2. Extract features: Study Hours, Attendance, Midterm, Assignments, Quizzes
3. Split data: 80% training, 20% testing
4. Apply Polynomial Features transformation (degree=2)
5. Apply StandardScaler normalization
6. Train LinearRegression model
7. Calculate performance metrics (R², MAE, MSE)

### Prediction Pipeline
1. User adjusts slider values
2. Input values are compiled into array
3. Polynomial transformation applied
4. Scaling applied (using training scaler)
5. Model predicts final score
6. Score clipped to valid range (0-100)
7. Grade and insights generated

## 💡 Interpretation Guide

### Grade Classifications

| Grade | Score Range | Background Color |
|-------|-------------|-----------------|
| **Excellent** | 85-100 | Green (#d1fae5) |
| **Good** | 75-84 | Blue (#dbeafe) |
| **Average** | 65-74 | Yellow (#fef9c3) |
| **Below Average** | 55-64 | Orange (#ffedd5) |
| **Needs Improvement** | 0-54 | Red (#f7f3f3) |

### Key Insights Logic

The app provides specific recommendations:
- Study hours < 12/week: Recommend increasing study time
- Attendance < 70%: Alert about correlation with lower scores
- Midterm < 60: Risk signal for final exam
- Assignments < 65: Emphasize assignment importance
- Quizzes < 65: Highlight retention importance

## 🔄 Caching & Performance

- **@st.cache_resource**: Model is trained once and cached in memory
- **No retraining**: Sliders update predictions instantly without retraining
- **Responsive UI**: Sub-second prediction latency

## 🐛 Troubleshooting

### Text Not Visible
- Clear browser cache and reload
- Ensure CSS styling is properly applied
- Check browser console for CSS errors

### Dataset Not Found
- Verify dataset path: `../student_final_score_predictor/Students_Performance_Synthetic_5000.csv`
- Ensure correct working directory when running the app

### Slow Performance
- Restart Streamlit to ensure cached model is fresh
- Check system resources and available memory

## 📝 License

This project is provided as-is for educational purposes.

## 👨‍💼 Author

Created as part of Machine Learning coursework for student performance prediction.

---

**Last Updated**: May 2026  
**Version**: 1.0
