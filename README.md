# 🏥 Medical Insurance Charges Prediction App

This project is a **Streamlit-based web application** that predicts **medical insurance charges** using machine learning. It is powered by a trained **Random Forest Regression model** on the `insurance.csv` dataset.

---

## 🚀 Live Features

- 🧠 Machine learning model predicts medical charges
- 📊 Interactive visualizations using Matplotlib & Seaborn
- 📷 Custom background image with stylish UI
- 📁 Model + Preprocessor saved for real-time inference
- 🧪 Detailed EDA and feature analysis
- 📦 Neatly organized folder structure for production-readiness

---

## 📂 Project Structure

```
├── app/
│   └── app.py                  # Streamlit app
├── data/
│   ├── insurance.csv           # Original dataset
│   ├── X_train.csv             # Saved training features
│   ├── X_test.csv              # Saved test features
├── images/
│   └── background_image.jpg    # Background image for Streamlit UI
├── models/
│   ├── final_model.pkl         # Trained ML model
│   └── preprocessor.pkl        # Preprocessing pipeline
├── src/
│   ├── preprocessing.py        # Preprocessing function
│   ├── model_training.py       # Model training + saving script
│   ├── eda.py                  # EDA with correlation and histograms
│   ├── *.ipynb                 # Jupyter notebooks (EDA, model dev)
├── outputs/
│   └── *.png                   # EDA plots (auto-saved)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---

## 📊 Dataset Overview

- **Source:** [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Features:**
  - `age`: Age of the individual
  - `sex`: Gender (`male`, `female`)
  - `bmi`: Body mass index
  - `children`: Number of children covered by insurance
  - `smoker`: Smoking status (`yes`, `no`)
  - `region`: Residential region
  - `charges`: Medical cost billed by health insurance (Target)

---

## 🧠 Model Details

- **Model:** Random Forest Regressor
- **Pipeline:** Custom preprocessing with `ColumnTransformer` from `scikit-learn`
- **Evaluation Metrics:**
  - R² Score
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)

---

## 📺 App Demo (Streamlit UI)

- Navigate between:
  - **Home**: Introduction
  - **Medical Charges Prediction**: Input fields to get predictions
  - **About**: Project details and author credits
- Beautiful sidebar with styled radio buttons and local background image

---

## 📈 EDA Highlights

- Summary statistics
- Missing values
- Correlation heatmap (saved as: `outputs/correlation_heatmap.png`)
- Feature distributions (`outputs/distribution_<feature>.png`)

---

## 🛠️ How to Run Locally

### 🔧 Prerequisites

Make sure you have Python ≥ 3.8 and `pip` installed.

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Launch the App

```bash
streamlit run app/app.py
```

---

## ✍️ Authors

- **Rayhan Mahmud Ansari**  
  Dept. of CSE, Sylhet Engineering College  
  📧 rayhan_mahmud@sec.ac.bd

- **Nurul Islam Opu**  
  Dept. of CSE, Sylhet Engineering College  
  📧 nurulislamopu1@gmail.com

---

## 📌 Future Improvements

- Add cancer impact analysis (Bangladesh-specific)
- Include hospital recommendations based on location
- Deploy app using Streamlit Cloud or HuggingFace Spaces
- Add ML explainability (e.g., SHAP values)
