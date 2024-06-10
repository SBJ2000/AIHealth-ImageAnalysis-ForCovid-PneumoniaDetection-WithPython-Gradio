# AI-Health-Image-Analysis-For-Covid-&-Pneumonia-Detection-With-Python-&-Gradio
![Project Logo](https://github.com/SBJ2000/AIHealth-ImageAnalysis-ForCovid-PneumoniaDetection-WithPython-Gradio/blob/main/Images/Logo.jpg)
## Project Description
AIHealth is an artificial intelligence-based application designed for automated detection and classification of COVID-19, pneumonia, and normal chest X-ray images. The project leverages deep learning techniques to provide a valuable tool for healthcare professionals in the diagnosis of respiratory diseases.
## Project Architecture :
Before going into the application, we need to understand the architecture that the developer adapted to build the project & the tools needed.

### Model Architecture :
First, we have the Backend architecture for the CNN model:
Conv2D Layer: The Conv2D layer performs a 2D convolution operation on the input image. In this model, the first Conv2D layer applies 32 filters (or kernels) of size 3x3 to the input image. The purpose of this layer is to detect local patterns such as edges and textures. The 'relu' activation function (Rectified Linear Unit) is applied to introduce non-linearity into the model by transforming all negative values to zero and keeping positive values unchanged.

* MaxPooling2D Layer: The MaxPooling2D layer performs down-sampling (subsampling) by selecting the maximum value from each 2x2 window. This reduces the spatial dimensions of the feature maps, which helps to decrease the computational load and reduce overfitting by providing a form of translation invariance.

* Flatten Layer: The Flatten layer transforms the 2D matrix of features into a 1D vector. This step is necessary before feeding the data into fully connected layers. It essentially reshapes the pooled feature map into a vector that can be used by the dense layers.

* Dense Layer: The Dense layer, also known as a fully connected layer, consists of neurons that are fully connected to all the neurons in the previous layer. In this model, the first Dense layer has 128 neurons with a 'relu' activation function, which allows the model to learn complex representations. The final Dense layer has a number of neurons equal to the number of classes (3 in this case) and uses the 'softmax' activation function to output a probability distribution over the classes.

The combination of these layers allows the CNN to learn hierarchical representations of the input images, from simple features like edges in the initial convolutional layers to more complex patterns in the deeper layers, ultimately enabling accurate classification of chest X-ray images into COVID-19, NORMAL, and PNEUMONIA categories.

### Front-end Architecture :
The frontend architecture of this application is built using Gradio, a Python library that simplifies the creation of web-based user interfaces for machine learning models. The interface allows users to upload chest X-ray images and receive real-time classification results. The frontend consists of the following components:

* Gradio Interface: Gradio provides a high-level API to create interactive web interfaces. In this project, a Gradio interface is created using gr.Interface which links the uploaded image to the prediction function. The interface specifies the type of input (an image) and the type of output (a label with the top three class predictions).

* Image Preprocessing: When an image is uploaded, it is first converted to a numpy array and resized to match the input dimensions expected by the CNN model (224x224 pixels). This step ensures that the image is in the correct format for prediction.

* Prediction Function: The core of the frontend is the prediction function, which loads the pre-trained CNN model and processes the input image. The image is normalized by scaling pixel values to the range [0, 1], and then it is fed into the model to generate predictions. The function returns the predicted class label based on the highest probability score.

* Pre-trained Model: The pre-trained CNN model, saved during the training phase, is loaded using tensorflow.keras.models.load_model. This model is used to make predictions on the new images uploaded through the interface.

* Example Images: Gradio allows specifying example images that users can click on to see how the model performs. This provides a quick way to test the interface without needing to upload new images.

Together, these components provide a seamless and interactive user experience, enabling users to easily classify chest X-ray images into COVID-19, NORMAL, and PNEUMONIA categories using the trained CNN model.

## Installation & Usage :
###Prerequisites :

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.1-orange)](https://www.tensorflow.org/)
    
    TensorFlow: Required for building and training the CNN model.

[![Python](https://img.shields.io/badge/Python-3.8.5-blue)](https://www.python.org/)

    Python: The frontend interface and backend model are both written in Python.

[![Gradio](https://img.shields.io/badge/Gradio-2.3.1-green)](https://gradio.app/)

    Gradio: Used to create the web-based user interface.

[![Pillow](https://img.shields.io/badge/Pillow-8.0.1-yellow)](https://pillow.readthedocs.io/en/stable/)

    Pillow: Python Imaging Library required for image processing.

[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/SBJ2000/AIHealth-ImageAnalysis-ForCovid-PneumoniaDetection-WithPython-Gradio)
    
    Git: Used to clone the project repository and manage the code.

###Installation :

To install and set up the AIHealth project, follow these steps:

1- Clone this Repository using this command:

    git clone https://github.com/SBJ2000/AIHealth-ImageAnalysis-ForCovid-PneumoniaDetection-WithPython-Gradio.git

Then navigate to the project directory:

    cd AIHealth-ImageAnalysis-ForCovid-PneumoniaDetection-WithPython-Gradio

2- Prepare the dataset by splitting it into training and validation sets:

    import splitfolders
    splitfolders.ratio('Dataset', seed=1337, output='Dataset-Splitted', ratio=(0.8, 0.2))

3- Train the model:

    python IA Model.py

4- Start the frontend interface:

    python Front End.py

###Usage :

After installing the project, you can now run and use the application through the graphical interface provided by Gradio. Simply launch the interface and upload a chest X-ray image to get the classification result.

![Example of usage](https://github.com/SBJ2000/AIHealth-ImageAnalysis-ForCovid-PneumoniaDetection-WithPython-Gradio/blob/main/Images/ExampleOfUsage.jpg)

## Conclusion :

AIHealth is a deep learning-based application for the classification of chest X-ray images into COVID-19, pneumonia, and normal categories. It consists of a CNN model built with TensorFlow and a user-friendly frontend interface created with Gradio. The project demonstrates the application of artificial intelligence in the healthcare domain, providing a valuable tool for disease detection.

In summary, AIHealth facilitates the automated detection of COVID-19 and pneumonia using chest X-ray images, with a Python backend and Gradio frontend. Install the prerequisites, set up the project, and utilize the user-friendly interface for image classification.