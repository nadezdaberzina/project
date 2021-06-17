import numpy as np
import dill
import pandas as pd
dill._dill._reverse_typemap['ClassType'] = type
import flask


app = flask.Flask(__name__)
model = None


def load_model(model_path):
    global model
    with open('models/pipeline.dill', 'rb') as f:
        model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
    return "Welcome to cardio disease prediction process"


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active = "", "", "", "", "", "", "", "", "", "", ""
        request_json = flask.request.get_json()
        if request_json["age"]:
            age = request_json['age']
        if request_json["gender"]:
            gender = request_json['gender']
        if request_json["height"]:
            height = request_json['height']
        if request_json["weight"]:
            weight = request_json['weight']
        if request_json["ap_hi"]:
            ap_hi = request_json['ap_hi']
        if request_json["ap_lo"]:
            ap_lo = request_json['ap_lo']
        if request_json["cholesterol"]:
            cholesterol = request_json['cholesterol']
        if request_json["gluc"]:
            gluc = request_json['gluc']
        if request_json["smoke"]:
            smoke = request_json['smoke']
        if request_json["alco"]:
            alco = request_json['alco']
        if request_json["active"]:
            active = request_json['active']

        preds = model.predict_proba(pd.DataFrame({"age": [age],
                                                  "gender": [gender],
                                                  "height": [height],
                                                  "weight": [weight],
                                                  "ap_hi": [ap_hi],
                                                  "ap_lo": [ap_lo],
                                                  "cholesterol": [cholesterol],
                                                  "gluc": [gluc],
                                                  "smoke": [smoke],
                                                  "alco": [alco],
                                                  "active": [active]}))
        data["predictions"] = preds[:, 1][0]
        data["age"] = age
        data["success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
        "please wait until server has fully started"))
    modelpath = 'models/pipeline.dill'
    load_model(modelpath)
    app.run()
