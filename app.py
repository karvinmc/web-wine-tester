from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SubmitField, DecimalField, SelectField
from wtforms.validators import DataRequired, ValidationError
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret123"

# Custom validator to ensure a selection is made
def validate_selection(form, field):
    if field.data == "":
        raise ValidationError("Please select a target.")

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
    target = SelectField(
        "Target Attribute",
        choices=[
            ("", "Select a target"),
            ("alcohol", "Alcohol"),
            ("malicAcid", "Malic Acid"),
            ("ash", "Ash"),
            ("ashAlcanity", "Ash Alcanity"),
            ("magnesium", "Magnesium"),
            ("totalPhenols", "Total Phenols"),
            ("flava", "Flavanoids"),
            ("nonFlava", "Nonflavanoid Phenols"),
            ("pro", "Proanthocyanins"),
            ("color", "Color Intensity"),
            ("hue", "Hue"),
            ("od280", "OD280"),
            ("proline", "Proline"),
        ],
        validators=[DataRequired(), validate_selection],
    )
    submit = SubmitField("Submit")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/post")
def post():
    return render_template("post.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/tester", methods=["GET", "POST"])
def tester():
    form = WineTest()
    if form.validate_on_submit():
        # Extracting form data
        data = {
            "alcohol": form.alcohol.data,
            "malicAcid": form.malicAcid.data,
            "ash": form.ash.data,
            "ashAlcanity": form.ashAlcanity.data,
            "magnesium": form.magnesium.data,
            "totalPhenols": form.totalPhenols.data,
            "flava": form.flava.data,
            "nonFlava": form.nonFlava.data,
            "pro": form.pro.data,
            "color": form.color.data,
            "hue": form.hue.data,
            "od280": form.od280.data,
            "proline": form.proline.data
        }
        target = form.target.data

        # Convert form data to DataFrame
        df = pd.DataFrame([data])

        # Load dataset
        file_path = 'wine-clustering.csv'  # Update with the actual path
        dataset = pd.read_csv(file_path)

        # Prepare the data
        X = dataset.drop(columns=[target])
        y = dataset[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Predict with user input
        user_input_scaled = scaler.transform(df)
        user_prediction = model.predict(user_input_scaled)

        return render_template(
            "result.html",
            user_prediction=user_prediction[0],
            mse=mse,
            r2=r2,
            form=form
        )

    return render_template("tester.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)
