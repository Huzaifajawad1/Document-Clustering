import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import gensim.downloader as api

# Download and load the Google Word2Vec model
model = api.load('word2vec-google-news-300')

# Load the documents
doc_dir = "Doc50"
docs = []
for filename in sorted(os.listdir(doc_dir)):
    with open(os.path.join(doc_dir, filename), "r") as f:
        docs.append(f.read().lower())

# Load the ground truth labels
label_dir = "Doc50 GT"
labels = []
for i in range(1, 6):
    label_subdir = os.path.join(label_dir, "c" + str(i))
    for filename in sorted(os.listdir(label_subdir)):
        labels.append(i)

# Preprocess the documents
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

for i, doc in enumerate(docs):
    # Remove punctuation
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stop words and stem the words
    words = [ps.stem(word) for word in doc.split() if word not in stop_words]
    docs[i] = ' '.join(words)

# Vectorize the documents
vectorizer1 = CountVectorizer()
vectorizer2 = TfidfVectorizer(use_idf=True, max_df=1)
X1 = vectorizer1.fit_transform(docs)
X2 = vectorizer2.fit_transform(docs)

# Perform K-Means clustering for Baseline 1
kmeans1 = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X1)
y_pred1 = kmeans1.labels_

# Perform K-Means clustering for Baseline 2
kmeans2 = KMeans(n_clusters=5, random_state=45, n_init=10).fit(X2)
y_pred2 = kmeans2.labels_

# Evaluate the clustering performance using purity for Baseline 1
purity1 = 0
for i in range(1, 6):
    max_count = 0
    for j in range(50):
        if labels[j] == i:
            count = np.sum(y_pred1 == j)
            if count > max_count:
                max_count = count
    purity1 += max_count
purity1 /= 50

# Evaluate the clustering performance using purity for Baseline 2
purity2 = 0
for i in range(1, 6):
    max_count = 0
    for j in range(50):
        if labels[j] == i:
            count = np.sum(y_pred2 == j)
            if count > max_count:
                max_count = count
    purity2 += max_count
purity2 /= 50

print("Purity using Baseline 1 with preprocessing:", purity1)
print("Purity using Baseline 2 with preprocessing:", purity2)
