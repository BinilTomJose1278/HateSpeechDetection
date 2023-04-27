from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'many random bytes'
usr=""

@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/Predict',methods=['GET','POST'])
def pre():
    return render_template('index.html')

@app.route('/Tpred',methods=['GET','POST'])
def twe():
     user=request.form["tweet"]    
     return render_template('index.html',usr=user)

 

          


@app.route('/Fuzzy',methods=['POST'])
def fuz():
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=True)
