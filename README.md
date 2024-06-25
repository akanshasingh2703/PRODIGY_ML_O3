# Cats vs. Dogs Image Classification using SVM

## Introduction

This project demonstrates the use of Support Vector Machines (SVM) for classifying images of cats and dogs. The SVM classifier is trained on a dataset of cat and dog images, with images preprocessed using Principal Component Analysis (PCA) for dimensionality reduction.

## Dataset

The dataset used in this project consists of images of cats and dogs, necessary for training and evaluating the SVM classifier. Due to its size, the dataset is not included in this repository.

To obtain the dataset:

1. Download the dataset from [Kaggle Dogs vs. Cats competition](https://www.kaggle.com/c/dogs-vs-cats/data).
2. Extract the `train.zip` and `test1.zip` files locally.
3. Upload the extracted dataset folders (`train` and `test1`) to your Google Colab environment or local directory as specified in the project code.

The project assumes the following directory structure for the dataset:

- `/content/train/train` : Contains training images of cats and dogs.
- `/content/test1/test1` : Contains test images for evaluation.

Ensure the dataset directories are correctly set up before running the provided code.


## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Scikit-image

## Results

The SVM classifier is trained on the training dataset and evaluated on the test dataset. The accuracy of the model is computed using sklearn's metrics. Additionally, a function is included to randomly select images from the dataset and predict their labels.


## Conclusion

This project successfully demonstrates how SVM can be employed for accurate classification of images of cats and dogs. By leveraging PCA for feature reduction and SVM for classification, the model achieves robust performance in distinguishing between the two classes. Further improvements could involve exploring different preprocessing techniques, tuning SVM hyperparameters, or integrating deep learning approaches for enhanced accuracy.
