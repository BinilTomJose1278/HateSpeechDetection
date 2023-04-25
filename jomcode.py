import streamlit as st
import re
import keras
import tensorflow as tf
from nltk.corpus import stopwords
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
#vectorizer = TfidfVectorizer()

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



#inputcode

st.title("Hate speech detection by Thankamma")
html_temp = """
<div style="background-color:crimson;padding:10px;margin:0px 0px 10px 0px">
<h2 style="color:white;text-align:center;"></h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
data = st.text_input("Input tweet","Type Here")
    
data=cleantext(data)

data= stemming(data)

print(data)


# Fit and transform the input string
#vectorized_string = tfidf_vectorizer.fit_transform([data])
y=vectorizer.transform([str(data)])
st.text("")
prediction=classifier.predict(y)
if prediction==0:
   st.text("non hate speech")
else:
   st.text("hate speech")