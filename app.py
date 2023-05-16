from flask import Flask, render_template, request, redirect, url_for, flash,send_file
import nltk

import hypernetx as hnx
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import pickle
import re
import io
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
app.secret_key = 'many random bytes'
usr=""
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english')) - set(['not','no','don\'t','n\'t'])


#divide words to cases
#case 1 :  Not hate word
#case 2 : Hate word
#case 3 : Hate word, but depends on context
#case 4 : possibility of hate word, nbut high indeterminancy
#case 5 : not hate hate word, but depends on context

case1=['want','gonna','stop','apart','done','insid','bye','felicia','use','would','king','kong','insert','basebal','bat','male','domest','cell','she','still','thing','think','absolut','already','sun','readi','share','complet','meet','heaven','eat','abl','sayin','start','might','yo','forget','sooooooo','anoth','urself','privladg','ur','finish','read','amaz','book','recommend','love','spend','time','famili','friend','grate','opportun','far','great','workout','gym','today','excit','try','new','recip','dinner','tonight','music','favorit','artist','proud','hard','work','put','achieve','goal','kind','respect','norm','import','take','care','health','watch','inspir','ted','talk','feel','motiv','support','life','walk','alway','help','clear','mind','educ','key','travel','see','world','one','day','other','let','focu','unit','us','rather','divid','beauti','like','make','differ','someon','good','idea','learn','skill','keep','hous','apt','cold','hobbi','fulfil','break','must','bodi','refresh','benefici','cracker','somehow','avoid','get','shone','fr','bought','ah','sub','da','corner','sto','easi','dream','rememb','joyou','nazi','tell','bath','sorri','desir','say','abolish','gettin','gang','bred','beta',
'rough','ball','yeah','cleanin','deep','load','sweat','drippin','total','final','url','gettin','pleas','send','video','surround','also','follow','drool','bad','commun']

case2=['rape','boipussi','white','goat','fucker','brutalis','suck','slutti','whore','sex','fat','homo','dumb','shit','dead','racist','hoe','bitch','pussi','stupid','ugli','ass','divers','give','back','nigga','gay','fuck','bash','cunt','filthi','dirti','cock','cum','floodin','slut','dildo']

case3=['bull','terrorist','kill','thick','breed','swear','useless','spit','clout','drug','feend','bore','social','part','death','run','power','countri','greedi','bottom']


case4=['hole','cri','german','disgust','pound','smash','capri','sever','femal','choke','gal','men','hang','beat','lil','closet','mental','spread','inclus','penalti','short','first','freaki','punch','face','muslim','illeg','christian','daddi']

case5=['wish','huge','no','call','damn','photo','free','enjoy','deserv','not','except','peopl','go','outsid','unlock','potenti','small','act','make','out','father','big','posit','black','way',
]


#create subhypergraph from a threshold
c={}
speech=[]
erosion=[] 
hatespeech=[]
nonhatespeech=[]
listx=[]
listy=[]
listk=[]
listr=[]
listg=[]
@app.route('/')
def home():
    return render_template('homepage.html')


x={}
a=-1
def graph(s):
  global a,x
  word_list = s.split()
  a+=1
  x[a]=word_list


x1={}

def graph3(s):
  global a,x
  word_list = s.split()
  a+=1
  x1[a]=word_list

def graph1(s):

  global a,c
  
  word_list = s.split()
  a+=1
  c[a]=word_list

def weights(x):
  t=0
  y=0
  f=0
  c=0
  words=x.split()
  for i in words:
    c=c+1
    if i in case1:
      r=0
      if(t<r):
        t=r
      y=y+0
      f=f+1
    elif i in case2:
      t=1

    elif i in case3:
      t=1
      y=y+0.5
    elif i in case4:
      r=0.5
      if(t<r):
        t=r
      y=y+1
    elif i in case5:
      y=y+0.5
      f=f+1
  speech.append(x)
  erosion.append(x)
  if((t>0.8 and y>0.3)):
    print(x," : hate speech")
    graph1(x)
    hatespeech.append(x)
   
  else:
    print(x," : non hate speech")
    nonhatespeech.append(x)

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
        return("Non Hate Speech")
    else:
        return("Hate Speech")

def model2(user):
    pickle_in=open("RF_model (1).pickle","rb")
    classifier=pickle.load(pickle_in)
    prediction=classifier.predict(user)
    if prediction==0:
        return("Non Hate Speech")
    else:
        return("Hate Speech")

def model3(user):
    pickle_in=open("LR_model.pickle","rb")
    classifier=pickle.load(pickle_in)
    prediction=classifier.predict(user)
    if prediction==0:
        return("Non Hate Speech")
    else:
        return("Hate Speech")

@app.route('/Home')
def hoe():
    return render_template('homepage.html')

@app.route('/Predict',methods=['GET','POST'])
def pre():
    return render_template('prediction.html')

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
    return render_template('modeloutput.html',mod1=mod1,mod2=mod2,mod3=mod3)


@app.route('/upload', methods=['POST'])
def upload():
    global a
    a=-1
    csv_file=request.files['csv']
    df = pd.read_csv(csv_file)
    data=df.iloc[:, 0]
    data=data.apply(lambda x:cleantext(x))
    data= data.apply(stemming)
    d=data.apply(lambda y:graph(y))
    d=data.apply(lambda z:weights(z))
    return('',204)
   
@app.route('/hypergraph1', methods=['POST'])
def hypergraph():
    # Generate the URL for the image file
    G = hnx.Hypergraph(x)
    # Draw the graph using matplotlib
    fig, ax = plt.subplots()
    hnx.draw(G, ax=ax)

    # Save the plot to an image file
    png_output = io.BytesIO()
    fig.savefig(png_output, format='png')
    png_output.seek(0)

    # Send the PNG file to the client
    return send_file(png_output, mimetype='image/png')

@app.route('/hypergraph2', methods=['POST'])
def hypergraph2():
    G = hnx.Hypergraph(c)
    # Draw the graph using matplotlib
    fig, ax = plt.subplots()
    hnx.draw(G, ax=ax)

    # Save the plot to an image file
    png_output = io.BytesIO()
    fig.savefig(png_output, format='png')
    png_output.seek(0)

    # Send the PNG file to the client
    return send_file(png_output, mimetype='image/png')

@app.route('/Dilationnodes')
def dilation():
   #w.r.t nodes
   global listx
   for i in hatespeech:
        words=i.split()
   for j in words:
        if j not in listx:
            listx.append(j)
   return render_template('dilationnodes.html',listx1=listx)

@app.route('/Dilationedges')
def dilation1():
   #w.r.t edges
   global listx,listg
   for i in listx:
    for j in speech:

      words=j.split()
      if i in words:
        if j not in listg:
          listg.append(j)
   return render_template('dilationedges.html',listx1=listg)
    
@app.route('/Erosionnodes')
def erosion1():  
   #erosion w.r.t nodes
    global nonhatespeech,listk,listx,erosion,listy
    for i in nonhatespeech:
        words=i.split()
        for j in words:
            if j not in listk:
                listk.append(j)
    for i in listx:
        if i not in listk:
            listr.append(i)
    return render_template('erosionnodes.html',listy1=listr)

@app.route('/Erosionedges')
def erosion2(): 
    for i in listx:
        for j in erosion:
            words=j.split()
            if i in words:
              if j not in listy:
                listy.append(j)
    return render_template('erosionedges.html',speech1=listy)
@app.route('/output')
def output1():
  for i in hatespeech:
    words=i.split()
    for j in words:
      for y in speech:
        wordslist=y.split()
        if j in wordslist:
          speech.remove(y)
  return render_template('output.html',speech1=speech)
   
@app.route('/outputgraph')
def output2():
  global a

  for i in hatespeech:
    words=i.split()
    for j in words:
      for y in speech:
        wordslist=y.split()
        if j in wordslist:
          speech.remove(y)
  for i in speech:
    graph3(i)
  G = hnx.Hypergraph(x1)
    # Draw the graph using matplotlib
  fig, ax = plt.subplots()
  hnx.draw(G, ax=ax)

    # Save the plot to an image file
  png_output = io.BytesIO()
  fig.savefig(png_output, format='png')
  png_output.seek(0)

    # Send the PNG file to the client
  return send_file(png_output, mimetype='image/png')
   
@app.route('/Fuzzy',methods=['POST'])
def fuz():
    return render_template('fuzzy.html')

if __name__ == "__main__":
    app.run(debug=True)
