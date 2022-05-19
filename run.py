from flask import Flask, request, jsonify, copy_current_request_context
import numpy as np
import fbrs_predict  # import library to your project

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        checkpoint = 'resnet34_dh128_sbd'

        data = request.json
        img = np.array(data['image'], np.float)
        x_coord = data['x_coord']
        y_coord = data['y_coord']
        is_pos = data['is_pos']

        engine = fbrs_predict.fbrs_engine(checkpoint)

        #x_coord = []  # x image coordinates seed pts
        #y_coord = []  # y image coordinates seed pts
        #is_pos = []  # Either 1 or 0 for + or - seed pts
        mask_pred = engine.predict(x_coord, y_coord, is_pos, img)  # Get segmentation mask

        return {'prediction': mask_pred.tolist()}


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
