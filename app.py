from flask import Flask, render_template, request
from ai_advice import generate_advice
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        age = float(request.form["age"])
        education = float(request.form["education"])
        sex = float(request.form["sex"])
        cigsPerDay = float(request.form["cigsPerDay"])
        BPMeds = float(request.form["BPMeds"])
        prevalentStroke = float(request.form["prevalentStroke"])
        prevalentHyp = float(request.form["prevalentHyp"])
        diabetes = float(request.form["diabetes"])
        totChol = float(request.form["totChol"])
        sysBP = float(request.form["sysBP"])
        diaBP = float(request.form["diaBP"])
        BMI = float(request.form["BMI"])
        heartRate = float(request.form["heartRate"])
        glucose = float(request.form["glucose"])

        features = np.array([[
            age,
            education,
            sex,
            cigsPerDay,
            BPMeds,
            prevalentStroke,
            prevalentHyp,
            diabetes,
            totChol,
            sysBP,
            diaBP,
            BMI,
            heartRate,
            glucose
        ]])

        # Apply scaling
        features_scaled = scaler.transform(features)

        # Prediction
        prediction = model.predict(features_scaled)[0]

        # Probability
        probability = round(
            model.predict_proba(features_scaled)[0][1] * 100,
            2
        )
        advice = generate_advice(
    probability,

    age,

    BMI,

    totChol,

    sysBP,

    glucose,

    cigsPerDay
)

        if probability < 20:
            result = "🟢 Low Risk"
        elif probability < 50:
            result = "🟡 Moderate Risk"
        else:
            result = "🔴 High Risk"

        return render_template(
        "index.html",
        prediction=result,
        probability=probability,
        advice=advice
)

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            probability=0
        )


if __name__ == "__main__":
    app.run(debug=True)