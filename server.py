from flask import Flask, jsonify, request, send_file
import pandas as pd
import joblib
import shap
import base64
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()
    df = pd.DataFrame(json, index=[0])

    # predict
    model = load_model('Model/diabetes_model_1500.keras')
    y_pred = model.predict(df)
    pred_diabetes = int(y_pred[0])
    
    # shap
    explainer = joblib.load(filename="Shap/explainer.bz2")
    shap_values = explainer.shap_values(df)

    i = 0
    shap.force_plot(explainer.expected_value[i], shap_values[i], df.iloc[i], matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)

    # Save the image to a temporary file
    tmp_filename = "force_plot.png"
    with open(tmp_filename, "wb") as tmp_file:
        tmp_file.write(buf.read())

    # Return the image URL in the response
    result_map = {0: 'No', 1: 'Yes'}
    image_url = "/get_image"
    return jsonify({'diabetes': result_map[pred_diabetes], 'image_url': image_url})

@app.route('/get_image')
def get_image():
    # Return the saved image file
    return send_file("force_plot.png", mimetype='image/png')

if __name__ == "__main__":
    port = 5000
    app.run(host='0.0.0.0', port=port)
