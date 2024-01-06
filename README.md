# Clear-Image-Generation-from-Blurred-Image-using-Deep-Learning
A deeplearning method used to convert the blurred image into a clear image on the local system.

Clear Image Generation from Blurred Image is a web application that showcases the process of generating clear images from blurred or unclear images using deep learning techniques.


## Demo Images

Here are some demo images showcasing the functionality of your project:

![Demo Image 1](demo_images/Demo_image(1).jpg)
![Demo Image 2](demo_images/Demo_image(2).jpg)
![Demo Image 3](demo_images/Demo_image(3).jpg)
![Demo Image 3](demo_images/Demo_image(4).jpg)


## Features

- Upload a blurred or unclear image.
- View the original blurred image and the generated clear image.
- Download the generated clear image.

## How to Use

Sure, here are step-by-step instructions to install Python on Windows, add the path to pip, clone a GitHub repository, navigate to the folder using CMD, and run the Flask application:

### Install Python on Windows and Add Path to Pip:
1. Download the latest Python installer for Windows from the official website: [Python Downloads](https://www.python.org/downloads/).
2. Run the installer, and during installation, make sure to check the box that says "Add Python to PATH."

### Clone GitHub Repository using Git:

3. Open the Command Prompt (CMD) on your computer. Clone the GitHub repository using the following command:
   ```bash
   git clone https://github.com/username/repository.git
   ```   
### OR Download Zip from GitHub and Unzip:
1. Visit the GitHub repository in your web browser.
2. Click on the "Code" button and select "Download ZIP."
3. Extract the contents of the downloaded ZIP file to your desired directory.

### Navigate to the Repository Folder:

1. The donwloaded folder will be in the downloads of the C:/Downloads unzip that zip file and place in the download folder itself.

2. Open the Command Prompt (CMD) on your computer.
   
3. Navigate to the directory where you want to clone the repository using the `cd` command:
   ```bash
   cd Clear-Image-Generation-from-Blurred-Image-using-Deep-Learning
   ```
#If you have the Github command enabled you can directly clone the folder and run to goo
     

### Install Dependencies:
1. Make sure you have Python and pip installed.
2. Run the following command to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Flask Application:
1. After installing the dependencies, run the Flask application using the following command:
   ```bash
   python app.py
   ```

### Access the Web Application:
1. It will Open automatically on your local broswer otherwise, Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
2. Use the provided form to upload a blurred image for deblurring.

### Predict Scientist from Blurred Image:
1. If you want to predict the scientist from a blurred image, go to the "test_input" folder and select a blurred image.
2. Click on the "Upload" button to submit the image and see the deblurred result along with the predicted scientist.

### Download the Deblurred Image:
1. If you are satisfied with the deblurred result, you can download the deblurred image by clicking the "Download Deblurred Image" button.

Follow these steps, and you should be able to set up the environment and run the Flask application successfully. Adjust paths and commands based on your specific setup.

## Technologies Used

- Python
- Opencv
- Scipy
- Scikit-learn
- Flask (web framework)
- HTML, CSS (front-end)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
