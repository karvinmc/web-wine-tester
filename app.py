from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import SubmitField, DecimalField, SelectField
from wtforms.validators import DataRequired, ValidationError

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret123"

# Custom validator to ensure a selection is made
def validate_selection(form, field):
    if field.data == '':
        raise ValidationError('Please select a target.')

# Form Class
class WineTest(FlaskForm):
    alcohol = DecimalField("Alcohol", validators=[DataRequired()])
    malicAcid = DecimalField("Malic Acid", validators=[DataRequired()])
    ash = DecimalField("Ash", validators=[DataRequired()])
    ashAlcanity = DecimalField("Ash Alcanity", validators=[DataRequired()])
    magnesium = DecimalField("Magnesium", validators=[DataRequired()])
    totalPhenols = DecimalField("Total Phenols", validators=[DataRequired()])
    flava = DecimalField("Flavanoids", validators=[DataRequired()])
    nonFlava = DecimalField("Nonflavanoid Phenols", validators=[DataRequired()])
    pro = DecimalField("Proanthocyanins", validators=[DataRequired()])
    color = DecimalField("Color Intensity", validators=[DataRequired()])
    hue = DecimalField("Hue", validators=[DataRequired()])
    od280 = DecimalField("OD280", validators=[DataRequired()])
    proline = DecimalField("Proline", validators=[DataRequired()])
    target = SelectField("Target Attribute", choices=[
        ("", "Select a target"),
        ("alcohol", "Alcohol"),
        ("malicAcid", "Malic Acid"),
        ("ash", "Ash"),
        ("magnesium", "Magnesium"),
        ("totalPhenols", "Total Phenols"),
        ("flava", "Flavanoids"),
        ("nonFlava", "Nonflavanoid Phenols"),
        ("pro", "Proanthocyanins"),
        ("color", "Color Intensity"),
        ("hue", "Hue"),
        ("od280", "OD280"),
        ("proline", "Proline")
        ], validators=[DataRequired(), validate_selection])
    submit = SubmitField("Submit")

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

@app.route('/tester', methods=['GET', 'POST'])
def tester():
    form = WineTest()
    alcohol = None
    malicAcid = None
    ash = None
    ashAlcanity = None
    magnesium = None
    totalPhenols = None
    flava = None
    nonFlava = None
    pro = None
    color = None
    hue = None
    od280 = None
    proline = None
    target = 'Choose a target attribute'

    # Validate
    if form.validate_on_submit():
        alcohol = form.alcohol.data
        form.alcohol.data = ''
        malicAcid = form.malicAcid.data
        form.malicAcid.data = ''
        ash = form.ash.data
        form.ash.data = ''
        ashAlcanity = form.ashAlcanity.data
        form.ashAlcanity.data = ''
        magnesium = form.magnesium.data
        form.magnesium.data = ''
        totalPhenols = form.totalPhenols.data
        form.totalPhenols.data = ''
        flava = form.flava.data
        form.flava.data = ''
        nonFlava = form.nonFlava.data
        form.nonFlava.data = ''
        pro = form.pro.data
        form.pro.data = ''
        color = form.color.data
        form.color.data = ''
        hue = form.hue.data
        form.hue.data = ''
        od280 = form.od280.data
        form.od280.data = ''
        proline = form.proline.data
        form.proline.data = ''
        target = form.target.data
        form.target.data = 'Choose a target attribute'

    return render_template('tester.html',
                           form = form,
                           alcohol = alcohol,
                           malicAcid = malicAcid,
                           ash = ash,
                           ashAlcanity = ashAlcanity,
                           magnesium = magnesium,
                           totalPhenols = totalPhenols,
                           flava = flava,
                           nonFlava = nonFlava,
                           pro = pro,
                           color = color,
                           hue = hue,
                           od280 = od280,
                           proline = proline,
                           target = target)

if __name__ == "__main__":
    app.run(debug=True)