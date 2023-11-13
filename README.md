# Kaggle Dog Breeds Identification Project

This project addresses the Kaggle competition [Dog Breeds Identification](https://www.kaggle.com/competitions/dog-breed-identification) using feature-based transfer learning. The following three main steps are described:

1. Evaluate a set of pre-trained models based on ImageNet through transfer learning on the dog breeds dataset.
2. Select the top two models with the highest validation accuracy as feature extractors to obtain the distinctive features of the dog breeds.
3. Concatenate the extracted features and input them into a Deep Neural Network (DNN) classifier for training and predicting the dog breeds.

## How to run the project

- **For Step 1, please use the command to run the python file `python models_evaluation.py`.**

  - This code file designs a complete transfer learning process from pre-trained models to the target dataset.
  - The models chosen to be evaluated in the "models_list". The results will be saved to "outputs/models_compare.csv" and printed on the command line. The results display train loss and valid loss of each model.
  - Due to the high demand of computation of the large-scale model training, runtime errors like "CUDNN_STATUS_EXECUTION_FAILED" may occur during long time running. It is suggested to restart and seperate the whole list into several gruops and test in progression.
- **For Step 2 and 3, please use the command to run the python file `python main.py`.**

  - This code file develops a simple DNN classifier trained with the features extracted by the best 2 models in Step 2. Files in src folder `preprocessing.py`, `visualization.py`, and  `model.py` provides supporting functions for the main file.
  - The extracted features and produced images are stored in the outputs folder.
  - The results display the loss curve and final valid loss of the trained DNN model. A `submission.csv` is created for test results.
- **In addition, resnet151 and densenet162 can be trained individually by running  `single_train.ipynb`.**

  - This notebook contains a complete process of transfer learning from data preprocessing to model evaluation.

## Packages required

The following lists present the main packages needed to run the project code.
The `requirement.txt` file that refers to the environment of the  project is also generated in the folder.

- **torch** is a popular open-source machine learning framework based on the Lua programming language, commonly used for developing and training deep neural networks.
- **torchvision** is a PyTorch package that provides access to popular datasets, model architectures, and image transformations for computer vision tasks.
- **numpy** is the fundamental package for array computing with Python.
- **pandas** provides fast, flexible, and expressive data structures in Python.
- **matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- **os** provides a portable way of using operating system dependent functionality.
- **d2l** is a collection of Jupyter notebooks and utility functions for learning deep learning.
- **PIL** adds image processing capabilities to Python interpreters.
- **timm** is a PyTorch library for efficient implementation of modern computer vision models.

## Role of each file

- **main.py** is where to run the main task. The DNN classifer is trained and evaluated in this file. The dataset is loaded and processed in this file.
- **modules_evaluation.py** scores a set of pre-trained models on the target dataset with their valid loss to choose the best feature extractor.
- **model.py** defines the DNN model structure and how to train and evaluate the model.
- **B1.py** defines the CNN structure and the parameters set for each layer. This file also provides function `B1` to train, test, and predict on the dataset. Results analysis part is included in the evaluate part.
- **B2.py** defines the CNN structure and the parameters set for each layer. This file also provides function `B2` to train, test, and predict on the dataset. Results analysis part is included in the evaluate part.
- **pre_processing.py** provides functions to the data preparation. `extract_features_labels` pre-processes the original data with a face detector to offer input data for Task A; `ImgDataGenerator_process` pre-processes the original data with the imagedatagenerator method to offer input data for Task B.
- **result_process.py** includes functions to visualize the training process and results of the model.
- **shape_predictor_68_face_landmarks.dat** serves as the pre-trained model for the dlib 68 face detector function.
