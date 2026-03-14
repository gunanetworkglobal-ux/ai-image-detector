from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.applications.MobileNetV2(weights="imagenet")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).resize((224,224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)
        prediction = decoded[0][0][1]

    return f'''
    <h2>AI Image Detector</h2>
    <form method="POST" enctype="multipart/form-data">
    <input type="file" name="image">
    <input type="submit">
    </form>
    <h3>Prediction: {prediction}</h3>
    '''

if __name__ == "__main__":
    app.run(debug=True)