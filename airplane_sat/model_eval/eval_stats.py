from sklearn.metrics import classification_report, plot_confusion_matrix, \
    recall_score, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import pandas

def get_model_report(model, x_test, y_test, \
    confusion_matrix_name="confusion_matrix", classification_report_name="confusion_matrix"):
    '''
    Calculates and saves confusion matrix and classification report for model
    model: trained model
    x_test: features for trained model to make predictions on
    y_test: labeled data for trained model to be evaluated against
    confusion_matrix_name: file name for confusion matrix
    classification_report_name: file name for classication report
    '''
    # Get predictions on Test Data
    y_pred = model.predict(x_test)
    
    
    # Get performance report, save as cvs
    report = classification_report(y_test, y_pred)
    df_report = pandas.DataFrame(report).transpose()
    df_report.to_csv('{}.csv'.format(classification_report_name), index= True)

    #Get confustion matrix, save a png
    plot_confusion_matrix(model, x_test, y_test)  
    plt.savefig('{}.png'.format(confusion_matrix_name))


def get_stats(model, x_test, y_test, labels=None):
    '''
    Returns: accuracy,recall, precision and f1 score
    model: trained model
    x_test: features for trained model to make predictions on
    y_test: labeled data for trained model to be evaluated against
    labels: if classification is not binary, provide labels here
    '''
    #test paramaters
    y_pred = model.predict()
    recall = recall_score(y_test, y_pred, labels=labels, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=labels, average='weighted')
    f1_score = f1_score(y_test, y_pred,labels=labels)
    return accuracy,recall, precision, f1_score
