
from  flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# load the train model
model_data = joblib.load("model/model.pkl")

model = model_data["model"]
encoder = model_data["encoder"]

@app.route("/", methods=["GET"])

def home():
    return jsonify({
        "message": "Students Performance Prediction API ",
        "status": "Running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    study_hours = data["study_hours"]
    attendance = data["attendance"]
    previous_marks = data["previous_marks"]
    assignment_score = data["assignment_score"]

    features =[[
        study_hours,
        attendance,
        previous_marks,
        assignment_score
    ]]

    prediction = model.predict(features)
    result = encoder.inverse_transform(prediction)[0]
    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(
        host = "0.0.0.0",
        port = 5000
    )
