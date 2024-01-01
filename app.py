from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
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

@app.route('/', methods=['GET', 'POST'])
def index():
    original_img = None
    deblurred_img = None

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

            # Define the blurring PSF
            kernel_size = 5
            blur_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

            # Deblur the image using Richardson-Lucy
            deblurred_image = deblur_richardson_lucy(blurred_image, blur_kernel)

            # Save the original and deblurred images
            original_img = f'/static/uploads/{filename}'
            deblurred_filename = f'deblurred_{filename}'
            deblurred_path = os.path.join(app.config['UPLOAD'], deblurred_filename)
            cv2.imwrite(deblurred_path, cv2.cvtColor(deblurred_image, cv2.COLOR_RGB2BGR))
            deblurred_img = f'/static/uploads/{deblurred_filename}'

    return render_template('index.html', original_img=original_img, deblurred_img=deblurred_img)

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
