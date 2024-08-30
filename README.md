# Machine-Learning-Model-Vibration-Data-Classification

# Machine Learning Classification of Vibration Data

This repository contains code to classify vibration data from various machines using different machine learning models. The code includes data loading, preprocessing, model training, hyperparameter tuning, and evaluation.

## Project Overview

The objective of this project is to classify time-series vibration data from 12 different machines using several classification models. The data is processed, features are extracted, and various models are trained to achieve classification accuracy. The models evaluated include Support Vector Machine (SVM), k-Nearest Neighbors (k-NN), Naive Bayes, and Decision Tree. An ensemble voting classifier is also used to combine the predictions from these models.

## Directory Structure

- `dataset_dir/`: Directory containing the `.dat` files with the vibration data.

## Requirements

To run this code, ensure you have the following Python packages installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tqdm`

You can install these packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
