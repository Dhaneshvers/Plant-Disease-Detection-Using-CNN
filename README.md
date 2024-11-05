# Plant Disease Detection Using CNN

This project focuses on detecting plant diseases using a Convolutional Neural Network (CNN) model, utilizing the PlantVillage dataset.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Performance Graphs](#performance-graphs)
- [Prediction](#prediction)
- [Future Work](#future-work)

## Dataset
The dataset used is the **PlantVillage** dataset, containing various plant leaf images labeled as healthy or diseased. The images are categorized based on disease types, allowing the model to learn and classify different plant conditions accurately.

- **Directory Structure**: 
    - The dataset is organized by folders for each plant disease category.
- **Image Details**:
    - Image Size: 256 x 256 pixels
    - Channels: 3 (RGB)
    - Format: PNG or JPG

## Installation
1. Clone this repository and navigate to the project directory:
    ```
    git clone <repository_url>
    cd <project_directory>
    ```
2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```
3. Set up the dataset in the specified directory (modify the code to point to your dataset location).

## Model Architecture
The CNN model for plant disease detection consists of several layers, each designed to capture specific features from the images.

- **Input Shape**: `(256, 256, 3)` for image dimensions and RGB channels.
- **Layers**:
    - **Convolutional Layers**: Apply multiple convolution operations to extract features.
    - **Batch Normalization**: Normalizes inputs, accelerating convergence.
    - **MaxPooling Layers**: Downsample feature maps to reduce dimensionality.
    - **Dense Layers**: Final layers to classify based on extracted features.

### Layer Summary:
1. **Conv2D Layers**: 32-64 filters with a kernel size of (3,3) and ReLU activation.
2. **BatchNormalization**: Used after initial Conv2D layer.
3. **MaxPooling2D**: Reduces spatial dimensions.
4. **Flatten**: Transforms 2D feature maps into a 1D vector for classification.
5. **Dense Layers**: Includes a fully connected layer and the output layer with softmax activation.

## Training
The model is trained using the following parameters:

- **Batch Size**: 32
- **Image Size**: 256 x 256
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

**Train/Validation Split**:
- **Train Set**: 80% of data
- **Validation Set**: 10% of data
- **Test Set**: 10% of data


