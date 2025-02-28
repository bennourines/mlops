from flask import Flask, render_template, request
import requests

app = Flask(__name__)

FASTAPI_URL = "http://localhost:8000/predict"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        form_data = {key: float(value) if key != "state" and key not in ["international_plan", "voice_mail_plan"] else value
                     for key, value in request.form.items()}

        response = requests.post(FASTAPI_URL, json=form_data)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
        else:
            prediction = "Error in prediction"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

