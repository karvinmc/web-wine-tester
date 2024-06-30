from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from wtforms import SubmitField, DecimalField, SelectField
from wtforms.validators import DataRequired, ValidationError, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret123"


# Custom validator to ensure a selection is made
def validate_selection(form, field):
    if field.data == "":
        raise ValidationError("Please select a target.")


# Form Class
class WineTest(FlaskForm):
    alcohol = DecimalField("Alcohol", validators=[Optional()])
    malicAcid = DecimalField("Malic Acid", validators=[Optional()])
    ash = DecimalField("Ash", validators=[Optional()])
    ashAlcanity = DecimalField("Ash Alcanity", validators=[Optional()])
    magnesium = DecimalField("Magnesium", validators=[Optional()])
    totalPhenols = DecimalField("Total Phenols", validators=[Optional()])
    flava = DecimalField("Flavanoids", validators=[Optional()])
    nonFlava = DecimalField("Nonflavanoid Phenols", validators=[Optional()])
    pro = DecimalField("Proanthocyanins", validators=[Optional()])
    color = DecimalField("Color Intensity", validators=[Optional()])
    hue = DecimalField("Hue", validators=[Optional()])
    od280 = DecimalField("OD280", validators=[Optional()])
    proline = DecimalField("Proline", validators=[Optional()])
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

@app.route("/blog4")
def blog4():
    return render_template("blog4.html")

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

        target = form.target.data
        alcohol = form.alcohol.data if target != "Alcohol" else "Target"
        malicAcid = form.malicAcid.data if target != "Malic_Acid" else "Target"
        ash = form.ash.data if target != "Ash" else "target"
        ashAlcanity = form.ashAlcanity.data if target != "Ash_Alcanity" else "Target"
        magnesium = form.magnesium.data if target != "Magnesium" else "Target"
        totalPhenols = form.totalPhenols.data if target != "Total_Phenols" else "Target"
        flava = form.flava.data if target != "Flavanoids" else "Target"
        nonFlava = form.nonFlava.data if target != "Nonflavanoid_Phenols" else "Target"
        pro = form.pro.data if target != "Proanthocyanins" else "Target"
        color = form.color.data if target != "Color_Intensity" else "Target"
        hue = form.hue.data if target != "Hue" else "Target"
        od280 = form.od280.data if target != "OD280" else "Target"
        proline = form.proline.data if target != "Proline" else "Target"

        # Extracting form data
        data = {
            "Alcohol": alcohol,
            "Malic_Acid": malicAcid,
            "Ash": ash,
            "Ash_Alcanity": ashAlcanity,
            "Magnesium": magnesium,
            "Total_Phenols": totalPhenols,
            "Flavanoids": flava,
            "Nonflavanoid_Phenols": nonFlava,
            "Proanthocyanins": pro,
            "Color_Intensity": color,
            "Hue": hue,
            "OD280": od280,
            "Proline": proline,
        }

        # Convert form data to DataFrame
        df = pd.DataFrame([data])
        df = df.drop(columns=[target])

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
        )
    return render_template("tester.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
