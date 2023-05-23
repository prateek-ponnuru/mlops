import json
import pandas as pd

import mlflow
from prefect import flow, task

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, ClassificationPerformanceProfileSection


@task
def load_data(filename):
    data = pd.read_csv(filename)
    data['Techlist'] = data.Technologies.apply(lambda x: x.split('/'))
    return data.copy()


@task
def get_predictions(data, features):
    MODEL = 'baseline_rf'
    STAGE = 'Staging'
    MLFLOW_TRACKING_URI = 'http://18.157.175.19:5000/'

    logged_model = f"models:/{MODEL}/{STAGE}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    loaded_model = mlflow.sklearn.load_model(logged_model)

    X = data[features].to_dict(orient='records')
    data['prediction'] = loaded_model.predict(X)

    return data[features+['prediction']].copy()


@task
def run_evidently(ref_data, data):

    profile = Profile(sections=[DataDriftProfileSection()])
    mapping = ColumnMapping(prediction="prediction", numerical_features=['Experience'],
                            categorical_features=['Diplome', 'Ville'],
                            datetime_features=[])
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


# @task
# def save_report(result):
#     client = MongoClient("mongodb://localhost:27018/")
#     client.get_database("prediction_service").get_collection("report").insert_one(result[0])


@task
def save_html_report(result):
    result[1].save("evidently_report_example.html")


@flow
def batch_analyze():
    new_data = load_data("data/data_sim.csv")
    new_data = get_predictions(new_data, ['Diplome', 'Ville', 'Techlist', 'Experience'])

    ref_data = load_data("data/data_clean.csv").sample(frac=0.3, random_state=143)
    ref_data = get_predictions(ref_data, ['Diplome', 'Ville', 'Techlist', 'Experience'])

    result = run_evidently(ref_data, new_data)
    # save_report(result)
    save_html_report(result)


batch_analyze()
