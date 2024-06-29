from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from wtforms import SubmitField, DecimalField, SelectField
from wtforms.validators import DataRequired, ValidationError
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
    alcohol = DecimalField("Alcohol")
    malicAcid = DecimalField("Malic Acid")
    ash = DecimalField("Ash")
    ashAlcanity = DecimalField("Ash Alcanity")
    magnesium = DecimalField("Magnesium")
    totalPhenols = DecimalField("Total Phenols")
    flava = DecimalField("Flavanoids")
    nonFlava = DecimalField("Nonflavanoid Phenols")
    pro = DecimalField("Proanthocyanins")
    color = DecimalField("Color Intensity")
    hue = DecimalField("Hue")
    od280 = DecimalField("OD280")
    proline = DecimalField("Proline")
    target = SelectField(
        "Target Attribute",
        choices=[
            ("", "Select a target"),
            ("Alcohol", "Alcohol"),
            ("Malic_Acid", "Malic Acid"),
            ("Ash", "Ash"),
            ("Ash_Alcanity", "Ash Alcanity"),
            ("Magnesium", "Magnesium"),
            ("Total_Phenols", "Total Phenols"),
            ("Flavanoids", "Flavanoids"),
            ("Nonflavanoid_Phenols", "Nonflavanoid Phenols"),
            ("Proanthocyanins", "Proanthocyanins"),
            ("Color_Intensity", "Color Intensity"),
            ("Hue", "Hue"),
            ("OD280", "OD280"),
            ("Proline", "Proline"),
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

@app.route("/blog1")
def blog1():
    return render_template("blog1.html")

@app.route("/blog2")
def blog2():
    return render_template("blog2.html")

@app.route("/blog3")
def blog3():
    return render_template("blog3.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/result")
def result():
    alcohol = request.args.get("alcohol")
    malicAcid = request.args.get("malicAcid")
    ash = request.args.get("ash")
    ashAlcanity = request.args.get("ashAlcanity")
    magnesium = request.args.get("magnesium")
    totalPhenols = request.args.get("totalPhenols")
    flava = request.args.get("flava")
    nonFlava = request.args.get("nonFlava")
    pro = request.args.get("pro")
    color = request.args.get("color")
    hue = request.args.get("hue")
    od280 = request.args.get("od280")
    proline = request.args.get("proline")
    user_prediction = request.args.get("user_prediction")
    mse = request.args.get("mse")
    r2 = request.args.get("r2")

    return render_template(
        "result.html",
        alcohol=alcohol,
        malicAcid=malicAcid,
        ash=ash,
        ashAlcanity=ashAlcanity,
        magnesium=magnesium,
        totalPhenols=totalPhenols,
        flava=flava,
        nonFlava=nonFlava,
        pro=pro,
        color=color,
        hue=hue,
        od280=od280,
        proline=proline,
        user_prediction=user_prediction,
        mse=mse,
        r2=r2,
    )

@app.route("/tester", methods=["GET", "POST"])
def tester():
    form = WineTest()
    if form.validate_on_submit():

        # Extracting form data
        data = {
            # "Alcohol": form.alcohol.data,
            "Malic_Acid": form.malicAcid.data,
            "Ash": form.ash.data,
            "Ash_Alcanity": form.ashAlcanity.data,
            "Magnesium": form.magnesium.data,
            "Total_Phenols": form.totalPhenols.data,
            "Flavanoids": form.flava.data,
            "Nonflavanoid_Phenols": form.nonFlava.data,
            "Proanthocyanins": form.pro.data,
            "Color_Intensity": form.color.data,
            "Hue": form.hue.data,
            "OD280": form.od280.data,
            "Proline": form.proline.data,
        }
        target = form.target.data

        # Convert form data to DataFrame
        df = pd.DataFrame([data])

        # Load dataset
        dataset = pd.read_csv("data/wine-clustering.csv")
        
        # Prepare the data
        X = dataset.drop(columns=[target])
        y = dataset[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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

        return redirect(
            url_for(
                "result",
                # alcohol=form.alcohol.data,
                malicAcid=form.malicAcid.data,
                ash=form.ash.data,
                ashAlcanity=form.ashAlcanity.data,
                magnesium=form.magnesium.data,
                totalPhenols=form.totalPhenols.data,
                flava=form.flava.data,
                nonFlava=form.nonFlava.data,
                pro=form.pro.data,
                color=form.color.data,
                hue=form.hue.data,
                od280=form.od280.data,
                proline=form.proline.data,
                user_prediction=user_prediction,
                mse=mse,
                r2=r2,
            )
        )
    return render_template("tester.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)