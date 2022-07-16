from sklearn.naive_bayes import GaussianNB
from airplane_sat.model_eval.eval_stats import get_stats, get_model_report
import pandas as pd
import numpy as np 
from airplane_sat.data_prep.data_prep import load_seperate_data

def naive_bayes(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    get_model_report(gnb, x_test, y_test, \
    confusion_matrix_name="naive_bayes_confusion", classification_report_name="naive_bayes_classification_report")
    

def naive_bayes_from_scratch():
    category_features = ['Gender', 'Customer Type', 'Type of Travel', \
        'Class', 'Inflight wifi service', 'Departure/Arrival time convenient', \
            'Ease of Online booking', 'Gate location', 'Food and drink', \
                'Online boarding', 'Seat comfort', 'Inflight entertainment', \
                    'On-board service', 'Leg room service', 'Baggage handling', \
                        'Checkin service', 'Inflight service', 'Cleanliness']


    #Use naive bayes to predict is the following customer will buy a computer
    #Dont use Laplace Smoothing
    df_train, df_test = load_seperate_data("train.csv", "test.csv")

    #split df for each class
    class_df = []
    classes = df_train['satisfaction'].unique()
    df_pos = df_train.loc[df_train['satisfaction'] == classes[0]]
    df_neg = df_train.loc[df_train['satisfaction'] == classes[1]]
    class_df = [df_pos, df_neg]
    N = df_pos.shape[0] + df_neg.shape[0]

    #feature probabilities in each class
    probabilities = np.zeros((len(classes), len(category_features)))
    #how likily that class is to be choosen
    class_probabilities = np.asarray([df_pos.shape[0]/N, df_neg.shape[0]/N])
    count = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for test_sample in df_test.rows:
        #for each class
        for class_data in class_df:
            class_size = class_data.shape[0]
            #for each feature
            for feature_index in range(len(category_features)):
                feature_title = category_features[feature_index]
                sample_value = test_sample.iloc[feature_title]
                instances = class_data.loc[class_data[feature_title] == sample_value].count()[0]
                probabilities[count][feature_index] = instances/class_size
            count += 1
            
        #multiply probabilities of each feature together per row/class
        #eg P(Age = youth|Class=Yes) * P(Income = 'medium'| Class=Yes) * P('Student' = yes| Class=Yes)* P('Credit' = fair| Class=Yes)
        feature_prod = np.prod(probabilities, axis=1)

        #Naive Bayes p(y|x) = p(x|y)*p(y)
        class_likelihood = np.multiply(feature_prod, class_probabilities)

        #Which Class Has the Highest Likelihood?
        best_match_index = np.argmax(class_likelihood)
        match = classes[best_match_index]
        if match == test_sample.iloc['satisfaction']:
            if best_match_index == 0:
                TP += 1
            else:
                TN += 1
        else:
            if best_match_index == 0:
                FP += 1
            else:
                FN += 1
                

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("The accurary of the test is {}".format(accuracy))

    precision =  TP / TP + FP
    recall =  TP / TP + FN
    print("The precision of the test is {} and the recall is {}".format(precision, recall))

    #What is the F1 score?
    f1 = (2*precision*recall)/(precision + recall)
    print("The F1 score is {}".format(f1 ))

naive_bayes_from_scratch()