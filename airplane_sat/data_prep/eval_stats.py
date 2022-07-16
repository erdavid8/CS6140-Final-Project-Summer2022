from sklearn.metrics import classification_report, plot_confusion_matrix, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import pandas

def get_model_report(model, x_test, y_test, \
    confusion_matrix_name="confusion_matrix", classification_report_name="confusion_matrix"):
    '''
    Calculates and saves confusion matrix and classification report for model
    params: paramters to use for model
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
