from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import shap
import base64
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

@app.route("/predictgood", methods=['POST'])
def do_prediction_good():
    json_data = request.get_json()
    df = pd.DataFrame(json_data, index=[0])

    # predict
    model = load_model('Model/diabetes_good_model.keras')
    y_pred = model.predict(df)
    pred_diabetes = int(y_pred[0])
    
    # shap
    explainer = joblib.load(filename="Shap/explainer_good.bz2")
    shap_values = explainer.shap_values(df)

    i = 0
    shap_plot = shap.force_plot(explainer.expected_value[i], shap_values[i], df.iloc[i], matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])

    # Save the plot as a Base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")

    # Return the Base64-encoded image string in the response
    result_map = {0: 'No', 1: 'Yes'}
    return jsonify({'diabetes': result_map[pred_diabetes], 'image_base64': base64_image})

@app.route("/predictbad", methods=['POST'])
def do_prediction_bad():
    json_data = request.get_json()
    df = pd.DataFrame(json_data, index=[0])

    # predict
    model = load_model('Model/diabetes_bad_model.keras')
    y_pred = model.predict(df)
    pred_diabetes = int(y_pred[0])
    
    # shap
    explainer = joblib.load(filename="Shap/explainer_bad.bz2")
    shap_values = explainer.shap_values(df)

    i = 0
    shap_plot = shap.force_plot(explainer.expected_value[i], shap_values[i], df.iloc[i], matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])

    # Save the plot as a Base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")

    # Return the Base64-encoded image string in the response
    result_map = {0: 'No', 1: 'Yes'}
    return jsonify({'diabetes': result_map[pred_diabetes], 'image_base64': base64_image})

if __name__ == "__main__":
    port = 5000
    app.run(host='0.0.0.0', port=port)