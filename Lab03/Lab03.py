# EDIT THIS CELL
# Construct a dictionary of prediction models to compare
# Uncomment the below dictionary and insert as many prediction models as you like.
# You may have used binary classification models in previous exercises
# You may also have to import these modules/libraries to be able to use them

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Construct a dictionary of prediction models to compare
PredictionModels = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-NN Classification': KNeighborsClassifier(n_neighbors=5),
}

models = list(PredictionModels.keys())

print("Fitting models, this may take a while")
for _model in models:
    # Fit model to the training data
    model = PredictionModels[_model]
    model.fit(Train_X, Train_Y)

    # Perform predictions on the Train set and Test set
    Train_Pred_Y = model.predict(Train_X)
    Test_Pred_Y = model.predict(Test_X)

    # Compute f1 score for both Train and Test sets
    train_f1 = f1_score(Train_Y, Train_Pred_Y, average='binary')
    test_f1 = f1_score(Test_Y, Test_Pred_Y, average='binary')

    # Display the F1 scores
    print(f"{_model} - Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")
    
    
    
    # Lab03.py
    
    
    # 4C16 Lab 3 -- classifier comparison

def question_1():
    # This function should return a string.
    # Uncomment the correct answer.
    #
    # Yes, you could try them each in turn..!
    # But don't do that.
    return "Nearest Neighbors"   # Placeholder
    # return "Nearest Neighbors"
    # return "Logistic Regression"
    # return "Linear SVM"
    # return "RBF SVM"
    # return "Decision Tree"

def question_2():
    # This function returns a single number,
    # corresponding to the answer.
    return 1.35 # Replace with your answer.

def question_3():
    # This function returns a single number,
    # corresponding to the answer (0--100)
    return 6.8 # Replace with your answer.

def question_4():
    # This function returns a single number, corresponding to the
    # digit most commonly misclassified.
    return 9 # Replace with your answer.