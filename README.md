# Intrusion_Detection_System_using_CNN

## Overview ##
This project implements an Intrusion Detection System (IDS) using Convolutional Neural Networks (CNN). The main objective is to enhance the accuracy and precision of intrusion detection compared to traditional methods. The model demonstrates outstanding performance with an accuracy rate of 98% and a precision rate of 94%.

## Libraries and Dependencies ##
- Numpy, Pandas, Joblib: For data manipulation and handling.
- Scikit-learn: For machine learning tools and metrics.
- Matplotlib, Seaborn: For data visualization.
- Keras: For building and training the CNN model.

## Methodology: ##
The project utilizes the following steps:

#### 1. Data Preprocessing: ####
- Loads the KDD Cup 99 dataset for training and testing.
-  Performs data cleaning tasks such as handling missing values and encoding categorical features using one-hot encoding.
- Normalizes numerical features using min-max scaling.

#### 2. Model Development: ####
- Constructs a CNN architecture with the following layers:
  - Convolutional Layer: Extracts features from the input data using a 1D convolutional kernel.
  - Max Pooling Layer: Reduces the dimensionality of the data and improves generalization.
  - Batch Normalization Layer: Accelerates training and improves model stability.
  - Flatten Layer: Converts the 2D convolutional output into a 1D vector.
  - Dropout Layer: Prevents overfitting by randomly dropping neurons during training.
  - Dense Layer: Performs classification and outputs the predicted class probabilities.
- Employs Stratified K-Fold Cross-Validation to evaluate model performance and mitigate overfitting.

## Results: ##
The developed CNN model achieved a validation accuracy of 98%. Additionally, a confusion matrix is provided to analyze the model's performance on different intrusion categories.

## Further Exploration: ##
This project lays the groundwork for further exploration in the domain of intrusion detection using deep learning techniques. Potential areas for future investigation include:
- Optimizing the CNN architecture for improved performance and efficiency.
- Exploring the use of other deep learning architectures such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks.
- Investigating the application of transfer learning techniques for leveraging pre-trained models.


## Disclaimer: ##
The information provided in this document is intended for educational and research purposes only. It should not be used to create or deploy real-world intrusion detection systems without further validation and testing.
