from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SubmitField, DecimalField, SelectField
from wtforms.validators import DataRequired, ValidationError
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret123"

# Load dataset
data = pd.read_csv("data/wine-clustering.csv")
data_columns = data.columns.tolist()

# Custom validator to ensure a selection is made
def validate_selection(form, field):
    if field.data == "":
        raise ValidationError("Please select a target.")

# Create a dynamic form based on available columns
def create_form(exclude_columns=[]):
    class WineTest(FlaskForm):
        if "Alcohol" not in exclude_columns:
            alcohol = DecimalField("Alcohol", validators=[DataRequired()])
        if "Malic_Acid" not in exclude_columns:
            malicAcid = DecimalField("Malic Acid", validators=[DataRequired()])
        if "Ash" not in exclude_columns:
            ash = DecimalField("Ash", validators=[DataRequired()])
        if "Ash_Alcanity" not in exclude_columns:
            ashAlcanity = DecimalField("Ash Alcanity", validators=[DataRequired()])
        if "Magnesium" not in exclude_columns:
            magnesium = DecimalField("Magnesium", validators=[DataRequired()])
        if "Total_Phenols" not in exclude_columns:
            totalPhenols = DecimalField("Total Phenols", validators=[DataRequired()])
        if "Flavanoids" not in exclude_columns:
            flava = DecimalField("Flavanoids", validators=[DataRequired()])
        if "Nonflavanoid_Phenols" not in exclude_columns:
            nonFlava = DecimalField("Nonflavanoid Phenols", validators=[DataRequired()])
        if "Proanthocyanins" not in exclude_columns:
            pro = DecimalField("Proanthocyanins", validators=[DataRequired()])
        if "Color_Intensity" not in exclude_columns:
            color = DecimalField("Color Intensity", validators=[DataRequired()])
        if "Hue" not in exclude_columns:
            hue = DecimalField("Hue", validators=[DataRequired()])
        if "OD280" not in exclude_columns:
            od280 = DecimalField("OD280", validators=[DataRequired()])
        if "Proline" not in exclude_columns:
            proline = DecimalField("Proline", validators=[DataRequired()])

        target = SelectField(
            "Target Attribute",
            choices=[("", "Select a target")] + [(col, col.capitalize()) for col in data_columns],
            validators=[DataRequired(), validate_selection],
        )
        submit = SubmitField("Submit")
    return WineTest

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
    target = request.args.get('target')
    exclude_columns = [target] if target else []
    form_class = create_form(exclude_columns)
    form = form_class(request.form)

    result = None

    if request.method == "POST" and form.validate():
        target = form.target.data

        # Prepare input data
        input_data = {}
        for field in form:
            if isinstance(field, DecimalField) and field.name != target:
                input_data[field.name] = [float(field.data)]

        input_df = pd.DataFrame(input_data)

        # Select features and target
        X = data.drop(columns=[target])
        y = data[target]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        input_scaled = scaler.transform(input_df)

        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Predict the target
        prediction = model.predict(input_scaled)
        result = prediction[0]

        # Save the model
        joblib.dump(model, f'{target}_regression_model.pkl')

        # Redirect to the same route with target parameter to remove target field
        return redirect(url_for('tester', target=target, result=result))

    return render_template("tester.html", form=form, result=result, target=target)

if __name__ == "__main__":
    app.run(debug=True)
