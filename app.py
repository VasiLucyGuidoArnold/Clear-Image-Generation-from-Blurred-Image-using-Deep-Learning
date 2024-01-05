from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import joblib
import webbrowser
import matplotlib.pyplot as plt

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def deblur_richardson_lucy(image, psf, iterations=30):
    psf /= psf.sum()
    deconvolved = np.copy(image).astype(np.float64)
    for _ in range(iterations):
        relative_blur = cv2.filter2D(deconvolved, -1, psf)
        ratio = image / (relative_blur + 1e-10)
        deconvolved *= cv2.filter2D(ratio.astype(np.float64), -1, psf)
    return np.clip(deconvolved, 0, 255).astype(np.uint8)

blur_kernel_size = 5  # Adjust the kernel size as needed
blur_kernel = np.ones((blur_kernel_size, blur_kernel_size), np.float32) / (blur_kernel_size * blur_kernel_size)
def blur_image(image, kernel_size=5):
    blur_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, blur_kernel)

def load_scientist_descriptions():
    # Dictionary mapping scientist names to their descriptions
    scientist_descriptions = {
        "Alan Turing": "Alan Turing was a mathematician and logician. He is often considered the father of theoretical computer science.",
        "Albert Einstein": "Albert Einstein was a theoretical physicist known for developing the theory of relativity.",
        "Marie Curie": "Marie Curie was a physicist and chemist, famous for her pioneering research on radioactivity.",
        "Isac Newton": "Isaac Newton was a mathematician, physicist, and astronomer who formulated the laws of motion and universal gravitation."
        # Add more scientists as needed
    }
    return scientist_descriptions

scientist_descriptions = load_scientist_descriptions()

# Load the trained model for face recognition
knn_clf = joblib.load('scientist_face_recognition_model.pkl')

def predict_person(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (100, 100))
    face_encoding = resized_image.flatten()
    face_encoding = face_encoding.reshape(1, -1)
    prediction = knn_clf.predict(face_encoding)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    original_img = None
    deblurred_img = None
    scientist_description = None

    if request.method == 'POST':
        if 'img' not in request.files:
            return redirect(request.url)

        file = request.files['img']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(file_path)

            # Load the blurred image
            blurred_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

            # Deblur the image using Richardson-Lucy
            deblurred_image = deblur_richardson_lucy(blurred_image, blur_kernel)

            # Save the original and deblurred images
            original_img = f'/static/uploads/{filename}'
            deblurred_filename = f'deblurred_{filename}'
            deblurred_path = os.path.join(app.config['UPLOAD'], deblurred_filename)
            cv2.imwrite(deblurred_path, cv2.cvtColor(deblurred_image, cv2.COLOR_RGB2BGR))
            deblurred_img = f'/static/uploads/{deblurred_filename}'

            # Predict the scientist after deblurring
            predicted_scientist = predict_person(deblurred_path)
            scientist_description = scientist_descriptions.get(predicted_scientist, "Description not available.")

    return render_template('index.html', original_img=original_img, deblurred_img=deblurred_img, scientist_description=scientist_description)

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
