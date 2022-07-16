from sklearn.naive_bayes import GaussianNB
from airplane_sat.model_eval.eval_stats import get_stats, get_model_report

def naive_bayes(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    get_model_report(gnb, x_test, y_test, \
    confusion_matrix_name="naive_bayes_confusion", classification_report_name="naive_bayes_classification_report")
    #accuracy,recall, precision, f1_score = get_stats(gnb, x_test, y_test)
    