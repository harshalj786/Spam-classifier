# 📧 Spam Classifier

A machine learning project that classifies emails and SMS messages as **Spam** or **Ham**.  
Built using **Python, scikit-learn, and Streamlit**, with my own reusable toolkits for data cleaning and PCA.

---

## 🚀 Features
- Trains on the **SMS Spam Collection Dataset** (`spam.csv`).  
- Text preprocessing using **TF-IDF vectorization** with uni- and bi-grams.  
- Logistic Regression classifier with balanced weights for imbalanced data.  
- Interactive **Streamlit app** (`app.py`) for real-time spam detection.  
- Shows predictions with **confidence scores** and probability bars.  
- Includes personal toolkits:
  - `data_clean.py`: custom **DataCleaner (DC)** class for NaNs, duplicates, text cleaning, and exports.  
  - `my_Pca.py`: custom **MY_PCA** class for scaling, dimensionality reduction, and explained variance.  

---

## 📂 Project Structure
├── app.py # Streamlit app for predictions
├── model_.py # Training script with pipeline & evaluation
├── spam.csv # Dataset (SMS Spam Collection)
├── spam_model.pkl # Saved trained model
├── data_clean.py # Personal data cleaning toolkit
├── my_Pca.py # Personal PCA toolkit
└── README.md # Documentation
