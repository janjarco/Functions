
#### Evaluation functions for binary classification problems

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score)
from sklearn import metrics
import sklearn.metrics as metrics
from datetime import datetime #  time checking
import warnings
from sklearn import metrics
import sklearn.metrics as metrics

warnings.simplefilter(action='ignore', category=FutureWarning)

# calculate the fpr and tpr for all thresholds of the classification


def plot_roc_auc_curve(y_test, y_proba, y_pred_bin, valid_score):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    """
    Plots the Receiver Operating Characteristic (ROC) curve for binary classification models.

    Parameters:
    y_test (array-like): True binary labels for the test set.
    y_proba (array-like): Predicted probabilities for the positive class.
    y_pred_bin (array-like): Predicted binary labels.
    valid_score (float): Valid ROC AUC score.

    Returns:
    None
    """

    # Calculate false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba)

    # Calculate area under the ROC curve
    roc_auc = metrics.auc(fpr, tpr)

    # Calculate ROC AUC score for y_test and y_pred_bin
    valid_score = metrics.roc_auc_score(y_test, y_pred_bin)

    # Clear current plot
    plt.clf()

    # Set plot size and font size
    plt.figure(figsize=(12, 12))
    font = {'size': 16}
    plt.rc('font', **font)

    # Set plot title and plot ROC curve
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % valid_score)
    # Set plot legend and plot diagonal line
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    # Set plot axis limits and labels
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(y_label, y_proba_pred):
    """
    Plots the Precision-Recall curve for a binary classifier.

    Parameters:
        y_label (pandas Series): True binary labels of the test set
        y_proba_pred (array-like): Predicted probabilities of the test set

    Returns:
        Shows the plot
    """

    # Calculate precision, recall and threshold using the scikit-learn library
    precision, recall, threshold = precision_recall_curve(y_label.values, y_proba_pred)

    # Create a new figure for the plot
    plt.clf()
    plt.figure(figsize=(12,12))

    # Set the title and axis labels for the plot
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Sensitivity', fontsize=16)

    # Set the limits for the axis
    plt.axis([-0.01,1,0,1])
    # Plot the Precision-Recall curve using blue line
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.tight_layout()
    plt.show()
    # Uncomment the below line to save the plot
    # plt.savefig(path +"\\..\\Modelling"+ '\\Precision recall curve.png')


# Creating a function to report confusion metrics
def confusion_metrics(conf_matrix):
# save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TP / float(TP + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')



#%% Model training and evaluation
import pandas as pd
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def model_evaluation(model, X_test, y_test, cutting_point=0.5, show_variables=False, show_params=False, plot_roc=False, plot_pr_curve=False):
    """
    Evaluate a binary classification model.
    Args:
        model: A trained binary classification model.
        X_test: The feature matrix of the test dataset.
        y_test: The target vector of the test dataset.
        cutting_point: The threshold used to binarize the predicted probabilities.
        show_variables: If True, show the variables used in the model.
        show_params: If True, show the parameters used to train the model.
        plot_roc: If True, plot the ROC curve.
        plot_pr_curve: If True, plot the Precision-Recall curve.

    Returns:
        A tuple containing the predicted probabilities, the binarized predictions, and the validation AUC score.
    """

    # Print separator
    print(10 * "_____________")

    # Show variables used in the model if requested
    if show_variables:
        print("Variables used in the model:", X_test.columns)

    # Show parameters used to train the model if requested
    if show_params:
        print("Parameters used to train the model:", model.get_params)

    # Get predicted probabilities and binarize predictions based on the cutting point
    y_proba = model.predict(X_test).tolist()
    y_pred = model.predict(X_test).tolist()
    y_pred_bin = y_pred
    for i in range(len(y_proba)):
        if y_pred[i][0] >= cutting_point:
            y_pred_bin[i] = 1
        else:
            y_pred_bin[i] = 0

    # Get the current time for model evaluation
    modelling_hour = datetime.now()

    # Print model evaluation metrics
    print("Model executed at:\n", modelling_hour)
    print("Cutting point:", cutting_point)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_bin)))
    valid_score = metrics.roc_auc_score(y_test, y_pred_bin)
    print(f"Validation AUC score: {valid_score}")

    # Plot ROC curve if requested
    if plot_roc:
        plot_roc_auc_curve(y_test, y_proba, valid_score)

    # Plot Precision-Recall curve if requested
    if plot_pr_curve:
        plot_pr_curve(y_test, y_proba)

    # Compute and print confusion matrix and other classification metrics
    cm = confusion_matrix(y_test, y_pred_bin)
    cm_df = pd.DataFrame(cm,
                        columns=['Predicted Negative', 'Predicted Positive'],
                        index=['Actual Negative', 'Actual Positive'])
    print(cm_df)
    confusion_metrics(cm)
    report = classification_report(y_test, y_pred_bin, target_names=['is_not_attributed', 'is_attributed'])
    print(report)

    # Return predicted probabilities, binarized predictions, and validation AUC score
    return y_proba, y_pred_bin, valid_score

