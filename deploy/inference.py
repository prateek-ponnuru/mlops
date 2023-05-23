from flask import Flask, request, jsonify
import mlflow


MODEL = 'baseline_rf'
STAGE = 'Staging'
MLFLOW_TRACKING_URI = 'http://18.157.175.19:5000/'

logged_model = f"models:/{MODEL}/{STAGE}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

app = Flask("data_profile_predictor")
app.model = loaded_model


@app.route("/predict/", methods=['POST'])
def predict_profile():
    if request.method == 'POST':
        body = request.json

        degree = body.get('diplome', 'No diploma')
        city = body.get('ville', 'Paris')
        entrp = body.get('entreprise', 'UNKNOWN')
        exper = float(body.get('experience', 0))
        skills = body.get('skills', [])

        if degree in ['PhD', 'PHD', 'Phd']:
            degree = 'Phd'
        elif degree in ['NO', 'None', 'No Degree', 'No diploma']:
            degree = 'No diploma'
        elif degree in ['MASTER', 'MSc', 'Mastere', 'msc', 'master', 'Master']:
            degree = 'Master'
        elif degree in ['Bachelor', 'bachelor', 'BSc']:
            degree = 'Bachelor'

        city = city.capitalize()

        skills = list(map(lambda x: x.lower(), skills))

        x = {'Diplome': degree, 'Ville': city,
             'Techlist': skills, 'Experience': exper}
        pred = list(app.model.predict(x))[0]
        send_to_evidently(x, pred)
        return jsonify({'prediction': list(app.model.predict(x))[0]})


def send_to_evidently(x, pred):
    # x['prediction'] = pred
    # requests.request('POST', f"{EVIDENTLY_SERVICE_ADDRESS}/interate/data_profiles", json=[x])
    # print(resp)
    pass


if __name__ == 'main':
    app.run(debug=True, host="0.0.0.0", port=8080)
