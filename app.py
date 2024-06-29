from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/post')
def post():
    return render_template('post.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/tester')
def tester():
    return render_template('tester.html')

if __name__ == "__main__":
    app.run(debug=True)