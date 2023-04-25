import pandas as pd
import re
import io
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
import time
from nltk.stem.porter import PorterStemmer
from keras.layers import Embedding
from nltk.tag import pos_tag_sents
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
c=0
l=0
f=0
v=0
JJ_count=0
NN_count=0
VB_count=0
RB_count=0
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
#reading dataframe
df = pd.read_csv('measuring_hate_speech (1).csv')
print("DATA LOADED\n")
def print_star():
    print('*'*50, '\n')
print(df.head(10))
print_star()
#selecting required columns
df=df[["tweet","label"]]
print("selecting required columns\n")
print(df.head(10))
print_star()
#Dropping null columns
df=df.dropna( axis=0)
print(df.head(10))
print_star()

print(df['label'].value_counts())


print(df)



# Seperating data and labels
data=df["tweet"]
labels=df["label"]

print("LABEL COUNT :\n",labels.value_counts())

print("DATAS\n")
print(data.head(10))
print(labels.head(10))
print_star()

print("Preprocessing Started")


port_stem = PorterStemmer()
stop_words = set(stopwords.words('english')) - set(['not','no','don\'t','n\'t'])
def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()

    review = [port_stem.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    return review



#text preprocessing
def cleantext(text):
  x=str(text).lower().replace('\\','').replace('_','')
  tag=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
  spcl=tag.replace('[^\w\s]','')
  return spcl



data=data.apply(lambda x:cleantext(x))
print("After removal of special characters\n")
print(data.head(10))
print_star()
data= data.apply(stemming)
print("After stemming and stopwords removal\n")
print(data.head(10))
print_star()
print('\nFINAL OUTPUT AFTER PREPROCESSING\n')
print(data.head(10))
print_star()
print("Preprocessing Completed")
print_star()





# Split data into train and test sets

# Define parameters
max_features = 10000
max_len = 26114
embedding_dim = 100

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(data)
 

# Pad sequences
#train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_features, maxlen=max_len)
#test_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_features, maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(train_features, labels, test_size = 0.2, random_state=101)
# Define model architecture
X_train = X_train.toarray()
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_features, embedding_dim, input_length=max_len),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train)
model.save('cnn_lstm_model.h5')
# Evaluate model
X_test=X_test.toarray()
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', accuracy)
