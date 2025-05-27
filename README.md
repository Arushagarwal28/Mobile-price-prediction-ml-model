# 📱 Mobile Price Prediction

This project predicts the price of mobile phones based on features like RAM, storage, battery, camera specs, and more using machine learning.  
The dataset was cleaned and preprocessed to extract numeric values from text-based specifications (e.g., "4 GB RAM" → 4).  
A Random Forest Regressor was used to train the model, achieving an **R² score of 0.90** and a low **RMSE of 0.24**.  
The workflow includes data cleaning, feature encoding, model training, and performance evaluation.  
This project demonstrates practical ML techniques for real-world price prediction tasks using Python and Scikit-learn.

## 🛠 Tools & Technologies
- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn (for visualization)

## 🚀 How to Run
1. Clone this repo and install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the Jupyter notebook or Python script to train and evaluate the model.

## 📁 Dataset
The dataset includes features such as:
- RAM, ROM, Battery, Processor, Display Size
- Front & Rear Camera, Ratings, Number of Ratings
- Price (target variable)

## 📊 Model Performance
- **Mean Squared Error (MSE):** 0.06
- **Root Mean Squared Error (RMSE):** 0.24
- **R² Score:** 0.90

---

> This project can be extended to support live prediction via a web interface using Streamlit or Flask.
