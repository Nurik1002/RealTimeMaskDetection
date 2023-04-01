# Real-Time Mask Detection

Real-Time Mask Detection is a project that aims to detect whether a person is wearing a mask or not in real-time using computer vision techniques. The project uses a Convolutional Neural Network (CNN) to detect masks, with an accuracy of 0.9764.


## Features

- Real-time detection of masks in video streams or from a webcam
- Built using OpenCV, Keras, and TensorFlow
- Easy to use and integrate with other projects
- CNN model architecture for mask detection

## Model Architecture

The model architecture used for mask detection is a CNN that consists of the following layers:

- Conv2D layer with 4 filters of size 3x3, padding "same", and ReLU activation
- BatchNormalization layer
- MaxPool2D layer with pool size of 2x2
- Dropout layer with 0.2 probability
- Conv2D layer with 8 filters of size 3x3, padding "same", and ReLU activation
- BatchNormalization layer
- MaxPool2D layer with pool size of 2x2
- Dropout layer with 0.2 probability
- Conv2D layer with 16 filters of size 3x3, padding "same", and ReLU activation
- BatchNormalization layer
- MaxPool2D layer with pool size of 2x2
- Dropout layer with 0.2 probability
- Conv2D layer with 32 filters of size 3x3, padding "same", and ReLU activation
- BatchNormalization layer
- MaxPool2D layer with pool size of 2x2
- Dropout layer with 0.2 probability
- Conv2D layer with 64 filters of size 3x3, padding "same", and ReLU activation
- BatchNormalization layer
- MaxPool2D layer with pool size of 2x2
- Dropout layer with 0.2 probability
- Flatten layer
- Dense layer with 128 units and ReLU activation
- BatchNormalization layer
- Dropout layer with 0.2 probability
- Dense layer with 10 units and ReLU activation
- BatchNormalization layer
- Dropout layer with 0.2 probability
- Dense layer with 2 units and sigmoid activation for binary classification (mask vs. no mask)

## Installation

1. Clone the repository: `git clone https://github.com/Nurik1002/RealTimeMaskDetection.git`
2. Navigate to the project directory: `cd RealTimeMaskDetection`
3. Install the dependencies: `pip install -r requirements.txt`

## Usage

To detect masks in real-time from a webcam:

`python detect_mask_video.py`

To detect masks in real-time from a video stream:

`python detect_mask_video.py --stream <stream_url>`

In the template above, be sure to replace <stream_url> with the actual URL of the video stream you want to use.
