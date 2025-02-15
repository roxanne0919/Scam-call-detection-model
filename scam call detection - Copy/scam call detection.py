import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('XPLORE AI\scam call detection\phonecall_data.csv')


# cleaning up dataset 
df = df.replace('neutral', '0', regex=True)
df = df.replace('legitimate', '0', regex=True)
df = df.replace('scam_response', '1', regex=True)
df = df.replace('highly_suspicious', '1', regex=True)
df = df.replace('slightly_suspicious', '1', regex=True)
df = df.replace('suspicious', '1', regex=True)
df = df.replace('scam', '1', regex=True)
df = df.replace('Step:', '', regex=True)


df = df.sample(frac=1)
df.reset_index(inplace=True)


from tqdm import tqdm
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud

# cleaning up text to remove punctuation
def preprocess_text(text_data):
    preprocessed_text = []
    
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                  for token in str(sentence).split()
                                  if token not in stopwords.words('english')))

    return preprocessed_text


preprocessed_review = preprocess_text(df['TEXT'].values)
df['TEXT'] = preprocessed_review






from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(df['TEXT'], 
                                                    df['LABEL'], 
                                                    test_size=0.25)

# vectorising the data
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)


# fitting the data using logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)



# testing the model
print(accuracy_score(y_train, model.predict(x_train)))
print(accuracy_score(y_test, model.predict(x_test)))



# testing new data

def preprocess_new_text(text_data):
    preprocessed_text = []
    
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
        preprocessed_text.append(' '.join(token.lower()
                                  for token in str(sentence).split()
                                  if token not in stopwords.words('english')))  # Remove stopwords

    return preprocessed_text

new_text = ["I ' m calling from the bank. We ' ve suspicious noticed activity. .. [Attack Formulation] which For account is this verification needed? Can you specify the nature the of activity suspicious?"]
new_text_processed = preprocess_new_text(new_text)

# Convert to TF-IDF vector using the already trained vectorizer
new_text_vectorized = vectorization.transform(new_text_processed)


prediction = model.predict(new_text_vectorized)
print("Prediction:", prediction[0])  # Output will be '0' (legit) or '1' (scam)
