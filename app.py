from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Load your trained models
model_cnn = load_model('models/model_cnn.h5')
model_resnet50 = load_model('models/model_resnet50.h5')
model_vgg16 = load_model('models/model_vgg16.h5')

# Define the classes for your models
class_labels = ["Covid", "Normal", "Viral Pneumonia"]

def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image array
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('get_prediction', filename=file.filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def get_prediction(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    img_cnn = preprocess_image(image_path, target_size=(64, 64))
    img_resnet50 = preprocess_image(image_path, target_size=(64, 64))
    img_vgg16 = preprocess_image(image_path, target_size=(64, 64))

    prediction_cnn = model_cnn.predict(img_cnn)
    prediction_resnet50 = model_resnet50.predict(img_resnet50)
    prediction_vgg16 = model_vgg16.predict(img_vgg16)

    predicted_label_cnn = class_labels[np.argmax(prediction_cnn)]
    predicted_label_resnet50 = class_labels[np.argmax(prediction_resnet50)]
    predicted_label_vgg16 = class_labels[np.argmax(prediction_vgg16)]

    return jsonify({
        'filename': filename,
        'predicted_label_cnn': predicted_label_cnn,
        'predicted_label_resnet50': predicted_label_resnet50,
        'predicted_label_vgg16': predicted_label_vgg16
    })

if __name__ == '__main__':
    app.run(debug=True)
