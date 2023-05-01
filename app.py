from flask import Flask, render_template, request, redirect, url_for, flash
from nltk.corpus import stopwords
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
app.secret_key = 'many random bytes'
usr=""
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english')) - set(['not','no','don\'t','n\'t'])

@app.route('/')
def home():
    return render_template('index1.html')

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

def model1(user):
    pickle_in=open("SVM_model.pickle","rb")
    classifier=pickle.load(pickle_in)
    prediction=classifier.predict(user)
    if prediction==0:
        return("non hate speech")
    else:
        return("hate speech")

def model2(user):
    pickle_in=open("RF_model (1).pickle","rb")
    classifier=pickle.load(pickle_in)
    prediction=classifier.predict(user)
    if prediction==0:
        return("non hate speech")
    else:
        return("hate speech")

def model3(user):
    pickle_in=open("LR_model.pickle","rb")
    classifier=pickle.load(pickle_in)
    prediction=classifier.predict(user)
    if prediction==0:
        return("non hate speech")
    else:
        return("hate speech")




@app.route('/Predict',methods=['GET','POST'])
def pre():
    return render_template('index.html')

@app.route('/Tpred',methods=['GET','POST'])
def pred():
    user=request.form["tweet"]
    pickle_in=open("vectorized_data1.pickle","rb")
    vectorizer=pickle.load(pickle_in)
    user=cleantext(user)
    user= stemming(user)
    user=vectorizer.transform([str(user)])
    mod1=model1(user)
    mod2=model2(user)
    mod3=model3(user)
    return render_template('index.html',mod1=mod1,mod2=mod2,mod3=mod3)



 
@app.route('/Fuzzy',methods=['POST'])
def fuz():
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=True)
