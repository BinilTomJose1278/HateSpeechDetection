from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'many random bytes'

#app.config['MYSQL_HOST'] = 'localhost'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = ''
#app.config['MYSQL_DB'] = 'trackmybus'
#mysql = MySQL(app)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/Predict',methods=['POST'])
def pre():
    return render_template('index.html')

@app.route('/Fuzzy',methods=['POST'])
def fuz():
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=True)
