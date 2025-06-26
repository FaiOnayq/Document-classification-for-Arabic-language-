# Document Classification for Arabic Language Using KALIMAT Arabic Corpus 

This project involves classifying Arabic text documents from the **[KALIMAT Multipurpose Arabic Corpus](https://sourceforge.net/projects/kalimat/files/kalimat/document-collection/)** using machine learning techniques. The dataset consists of articles collected from an Omani newspaper, categorized into six classes: culture, economy, local news, international news, religion, and sports.

## 📁 Dataset Overview

- **Total Articles**: 20,291  
- **Categories**:  
  - Culture  
  - Economy  
  - Local News  
  - International News  
  - Religion  
  - Sports  

## 🔧 Data Preprocessing

Text preprocessing includes:
- **Stopword removal** (using multiple sources)
- **Normalization** (standardizing Arabic characters)
- **Stemming**
- **Tokenization**
- **TF-IDF Vectorization** with:
  - `max_df=0.8`, `min_df=5`
  - `max_features=5000`
  - `ngram_range=(1,2)`
- **Hand-crafted Features**: Document length & unique word ratio
- **Feature Selection**: Top 100 features using `SelectKBest`

## 🤖 Model Selection
After preprocessing the Corpus with `preprocessing.ipynb` with the output features file `TFIDF.arff`
Multiple classifiers were tested using **Weka** with the f:
- Naive Bayes
- SVM (SVO)
- Random Forest ✅ *(Best performing model)*
- Multi-class Classifier
- AdaBoost M1

**Random Forest with 10-fold cross-validation** showed the most consistent performance.

## 🧪 Prediction Results

- **Sports** and **Religion** categories had the highest accuracy.

## 🌐 Deployment 

A basic Flask web application is implemented for deployment:
- Input text is preprocessed
- TF-IDF and feature selection models are applied
- A trained RandomForest model predicts the category
- Run locally via: `http://127.0.0.1:5000`

## ✔️ Samples 
(samples/sample1.jpeg)
(samples/sample2.jpeg)
(samples/sample3.jpeg)


## 📦 Requirements

To run the web app:
- Python 3.x
- Flask
- scikit-learn
- nltk
- arabicstopwords
- Weka (for training)

## 🚀 How to Use

1. Clone the repository
2. Install dependencies
3. Run `app.py`
4. Open browser at `http://127.0.0.1:5000`
5. Enter Arabic text and get classification result
