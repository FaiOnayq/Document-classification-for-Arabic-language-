from flask import Flask, request, render_template
import joblib
import pandas as pd
import pickle
import re
import nltk
from nltk.stem import ISRIStemmer
from nltk.tokenize import word_tokenize
import arabicstopwords.arabicstopwords as stp
nltk.download('punkt_tab')
nltk.download("stopwords")
# Load the pre-trained model
model = joblib.load('random_forest_model.pkl')



#arabic stop words from nlkt which contain 706 words
arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
arb_stopwords = ' '.join(arb_stopwords)
# Apply normalizing to the stopwords list because it do not contain many words
arb_stopwords = re.sub(r'[إأٱآا]', 'ا', arb_stopwords)
arb_stopwords = re.sub(r'ى', 'ي', arb_stopwords)
arb_stopwords = re.sub(r'ؤ', 'و', arb_stopwords)
arb_stopwords = re.sub(r'ئ', 'ي', arb_stopwords)
arb_stopwords = re.sub(r'[a-zA-Z]', '', arb_stopwords)
arb_stopwords = arb_stopwords.split()

# Function to remove stop words using arabicstopwords library  which contain 13465 words
def remove_stopWords(sentence):
    terms=[]
    stopWords= set(stp.stopwords_list())
    for term in sentence.split() :
        if term not in stopWords :
            terms.append(term)
    return " ".join(terms)

# Another arabic stop words from local text file which contain 769 words
ar_stop_list = open("ArabicStop.txt", encoding="utf-8")
stop_words = ar_stop_list.read().split('\n')
stop_words = ' '.join(stop_words)
# Apply normalizing to the stopwords list because it do not contain many words
stop_words = re.sub(r'[إأٱآا]', 'ا', stop_words)
stop_words = re.sub(r'ى', 'ي', stop_words)
stop_words = re.sub(r'ؤ', 'و', stop_words)
stop_words = re.sub(r'ئ', 'ي', stop_words)
stop_words = re.sub(r'[a-zA-Z]', '', stop_words)
stop_words = stop_words.split()


# reducing Arabic words to their root by removing prefixes, suffixes, and infixes using ISRIStemmer from nlkt
def stemm(text):
    text=word_tokenize(text)
    #stemming of each word
    stem=ISRIStemmer()
    text=[stem.stem(i) for i in text]
    return ' '.join(text)


# Load the selector from the pickle file
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)
    
# Load the selector from the pickle file
with open('tfidf_selector3.pkl', 'rb') as f:
    selector = pickle.load(f)

# function that preprocesses the input text
def preprocess_text(text):    
    text = remove_stopWords(text) # remove stop words 
    text = text.split()                                           # Split the cleaned text back into tokens
    text = [w for w in text if not w in arb_stopwords]   # remove stop words 
    text = [w for w in text if not w in stop_words]  # remove stop words 
    text = ' '.join(text)
    text = re.sub(r'\W|\d', ' ', text)                            # Remove punctuation and digits
    text = re.sub(r'[إأٱآا]', 'ا', text)                          # Normalize the Arabic text
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'وو', 'و',text)
    text = re.sub(r'يي', 'ي',text)
    text = re.sub(r'ييي', 'ي',text)
    text = re.sub(r'اا', 'ا',text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = stemm(text)
    text = remove_stopWords(text) # remove stop words 
    text = text.split()                                           # Split the cleaned text back into tokens
    meaningful_words = [w for w in text if not w in arb_stopwords]   # remove stop words 
    corpus = [w for w in meaningful_words if not w in stop_words]   # remove stop words 
    cleaned_text = ' '.join(corpus)
    # Transform the new inpu
    X_new_tfidf = loaded_vectorizer.transform([cleaned_text])
    
    # Tokenize the document into words
    words = cleaned_text.split()
    
    # Create handcrafted features for the document
    test_doc_length = len(words)
    test_unique_words_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
    # Create a DataFrame for the handcrafted features
    test_handcrafted_features_df = pd.DataFrame({
        'doc_length': [test_doc_length],
        'unique_words_ratio': [test_unique_words_ratio]
    })
    
    # Convert to TF-IDF matrix
    test_docterm_tfidf_df = pd.DataFrame(X_new_tfidf.todense(), columns=loaded_vectorizer.get_feature_names_out())

    # Combine the TF-IDF matrix with the handcrafted features
    test_docterm_handcrafted = pd.concat([test_docterm_tfidf_df.reset_index(drop=True), test_handcrafted_features_df], axis=1)
    
    # Apply feature selection on the document 
    X_test_selected = selector.transform(test_docterm_handcrafted)
    return X_test_selected[0]



# Initialize Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the form
    input_text = request.form['input_text']
    
    processed_features = preprocess_text(input_text)
    
    # Convert to DataFrame 
    input_data = pd.DataFrame([processed_features], columns=model.feature_names_in_)
    
    prediction = model.predict(input_data)
    
    return render_template('index.html', prediction=prediction[0],input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
