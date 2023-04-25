import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle
import re
import keras
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
pickle_in=open("LR_model.pickle","rb")
classifier=pickle.load(pickle_in)

pickle_in=open("vectorized_data1.pickle","rb")
vectorizer=pickle.load(pickle_in)
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english')) - set(['not','no','don\'t','n\'t'])

def cleantext(text):
  x=str(text).lower().replace('\\','').replace('_','')
  tag=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
  spcl=tag.replace('[^\w\s]','')
  return spcl

def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()

    review = [port_stem.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    return review
data=cleantext(data)

data= stemming(data)

print(data)


# Fit and transform the input string
#vectorized_string = tfidf_vectorizer.fit_transform([data])
y=vectorizer.transform([str(data)])
prediction=classifier.predict(y)
return prediction 
if prediction==0:
  return render_template('index.html',prediction.text="non-hate speech") 
else:
 return render_template('index.html',prediction.text="non-hate speech") 
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():

