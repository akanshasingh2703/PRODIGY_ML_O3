# Cats vs. Dogs Image Classification using SVM

## Introduction

This project demonstrates the use of Support Vector Machines (SVM) for classifying images of cats and dogs. The SVM classifier is trained on a dataset of cat and dog images, with images preprocessed using Principal Component Analysis (PCA) for dimensionality reduction.

## Dataset

The dataset consists of images of cats and dogs, stored in two separate zip files (`train.zip` and `test1.zip`). The images are extracted into directories (`/content/train/train` for training images and `/content/test1/test1` for test images) during execution.

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
