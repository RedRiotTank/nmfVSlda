# !pip install pdfplumber scikit-learn nltk transformers

from google.colab import drive
drive.mount('/content/drive')

import os
import pdfplumber
import re
import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

folder = "/content/drive/MyDrive/Master/TID/Teoría/TopicAnalysisTexts"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

doc_list = []
file_list = []

# --------------- TEXT PROCESSING ---------------

# 1. PARSER
for f in os.listdir(folder):
    if f.endswith(".pdf"):
        file_path = os.path.join(folder, f)
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + " "

            # 2. LEXIC + STOPWORDS + TOKENIZATION
            tokenized_text = tokenize(text)
            doc_list.append(tokenized_text)
            file_list.append(f)

n_topics = 5

# --------------- NMF MODEL ---------------


tfidf_vectorizer = TfidfVectorizer(max_df=0.85)
tfidf = tfidf_vectorizer.fit_transform(doc_list)

start_time_nmf = time.time()
nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=25)
nmf_features = nmf_model.fit_transform(tfidf)
end_time_nmf = time.time()

words = tfidf_vectorizer.get_feature_names_out()
nmf_topic_list = []
for idx, topic in enumerate(nmf_model.components_):
    nmf_topic_list.append([words[i] for i in topic.argsort()[-n_topics:]])

print("------------- NMF -------------")
for i, doc in enumerate(nmf_features):
    tema_principal = doc.argmax()
    print(f"{file_list[i]} related topics: {' '.join(nmf_topic_list[tema_principal])}")
print(f"Time NMF: {end_time_nmf - start_time_nmf:.2f} s\n")


# --------------- LDA MODEL ---------------

count_vectorizer = CountVectorizer(max_df=0.85)
count_data = count_vectorizer.fit_transform(doc_list)

start_time_lda = time.time()
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=25)
lda_features = lda_model.fit_transform(count_data)
end_time_lda = time.time()

lda_words = count_vectorizer.get_feature_names_out()
lda_topic_list = []
for idx, topic in enumerate(lda_model.components_):
    lda_topic_list.append([lda_words[i] for i in topic.argsort()[-n_topics:]])

print("------------- LDA -------------")
for i, doc in enumerate(lda_features):
    tema_principal = doc.argmax()
    print(f"{file_list[i]} related topics: {' '.join(lda_topic_list[tema_principal])}")
print(f"Time LDA: {end_time_lda - start_time_lda:.2f} s")
