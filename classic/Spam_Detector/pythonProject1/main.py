import pandas as pd
import os

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from scipy.sparse import hstack

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ML_Loggs")

path = "D:/NLP and LLM projects/NLP-and-LLM-Pet-projects/classic/Spam_Detector"
# defining a function to load the data

logger_handler = logging.FileHandler(os.path.join(path, 'pythonProject1/logs'), "w")

logger.addHandler(logger_handler)

def get_data(path):
    logger.debug("Geting data")
    data = []
    files = os.listdir(path)
    for file in files:
        f = open(path+file, encoding = "ISO-8859-1")
        words_list = f.read()
        data.append(words_list)
        f.close()
    logger.info("Data have loaded")
    return data

easy_ham_path = os.path.join(path, 'easy_ham/easy_ham/')
hard_ham_path = os.path.join(path, 'hard_ham/hard_ham/')
spam_path = os.path.join(path, 'spam_2/spam_2/')

easy_ham_data = get_data(easy_ham_path)
#hard_ham_data = get_data(hard_ham_path)
spam_data = get_data(spam_path)
print(easy_ham_data[2])


nltk.download("stopwords")
nltk.download("worldnet")
nltk.download('punkt')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words("english"))


def extract_num_features(text):
    logger.debug("Extracting numbers")
    features = {}
    numbers = re.findall(r'\d+', text)
    features["has_number"] = int(len(numbers)>0)
    features["number_count"] = len(numbers)
    features["is_time"] = int(bool(re.search(r"\b\d{1,2}[:]\d{2}\b", text)))
    features["is_price"] = int(bool(re.search(r"\$?\d+[\.,]?\d*\b", text)))
    return features

def clean_text(text):
    logger.debug("Cleaning text")
    features = extract_num_features(text)
    text = re.sub(r"<[^>]+>", "",text) # html
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # only words and spaces
    text = text.lower()
    tokens = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in tokens if not word in stopwords]
    features["text"] = " ".join(text)
    features = pd.Series(features)
    features[["has_number", "number_count", "is_time", "is_price"]] = features[["has_number", "number_count", "is_time", "is_price"]].astype(int)
    return features




non_spam_data = pd.DataFrame(map(clean_text, easy_ham_data))
non_spam_data["Spam"] = 0
spam_data = pd.DataFrame(map(clean_text, spam_data))
spam_data["Spam"] = 1

data = pd.concat([non_spam_data, spam_data], axis=0)

X = data.drop("Spam", axis=1)
y = data["Spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

class TextNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        logger.debug("Tokenization")
        self.tfdf = TfidfVectorizer(max_features=5000)

    def fit(self, X, y=None):
        self.tfdf.fit(X["text"])
        return self

    def transform(self, X):
        logger.info("Tokenizes")
        X_text = self.tfdf.transform(X["text"])
        X_num = X.select_dtypes([int])
        return hstack([X_text, X_num])



pipeline = Pipeline([
    ("TextNumTransf", TextNumericTransformer()),
    ("Classifier", DecisionTreeClassifier())
]
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

params = {
    "Classifier__max_depth": range(5, 10),
    "Classifier__min_samples_split": range(2, 5)
}



grid_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=params,
    cv=cv,
    scoring='f1',
    n_iter=10,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_pred, y_test))
logger.info("Model result: \n" + str(classification_report(y_pred, y_test)))
print(confusion_matrix(y_pred, y_test))
logger.info("Confusion matrix: \n" + str(confusion_matrix(y_pred, y_test)))








