import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('XPLORE AI\scam call detection\BETTER30.csv')


# cleaning up dataset 
df = df.replace('neutral', '0', regex=True)
df = df.replace('legitimate', '0', regex=True)
df = df.replace('scam_response', '1', regex=True)
df = df.replace('highly_suspicious', '1', regex=True)
df = df.replace('slightly_suspicious', '1', regex=True)
df = df.replace('suspicious', '1', regex=True)
df = df.replace('scam', '1', regex=True)
df = df.replace('Step:', '', regex=True)


df = df.drop(['CONTEXT', 'FEATURES', 'ANNOTATIONS'], axis=1)
df = df.sample(frac=1)
df.reset_index(inplace=True)
df.drop(["CONVERSATION_ID"], axis=1, inplace=True)



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


# Visualing word cloud for legitimate calls
consolidated = ' '.join(
    word for word in df['TEXT'][df['LABEL'] == '0'].astype(str))
wordCloud = WordCloud(width=1600,
                      height=800,
                      random_state=21,
                      max_font_size=110,
                      collocations=False)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


# Visualising word cloud for suspected scam calls 
consolidated = ' '.join(
    word for word in df['TEXT'][df['LABEL'] == '1'].astype(str))
wordCloud = WordCloud(width=1600,
                      height=800,
                      random_state=21,
                      max_font_size=110,
                      collocations=False)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


common_words = get_top_n_words(df['TEXT'], 20)
df1 = pd.DataFrame(common_words, columns=['Review', 'count'])

df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(10, 6),
    xlabel="Top Words",
    ylabel="Count",
    title="Bar Chart of Top Words Frequency"
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(df['TEXT'], 
                                                    df['LABEL'], 
                                                    test_size=0.25)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

# testing the model
print(accuracy_score(y_train, model.predict(x_train)))
print(accuracy_score(y_test, model.predict(x_test)))

