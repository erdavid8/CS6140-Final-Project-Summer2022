from sklearn import svm
import numpy as np 
import matplotlib.pyplot as plt
from  airplane_sat.data_prep.data_prep import load_seperate_data
from airplane_sat.model_eval.eval_stats import get_stats
import itertools

class FeatureCombo:
    def __init__(self, features, accuracy = 0, precision =0, recall =0, f1=0):
        self.features = features
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

def make_meshgrid(x, y, h=.02):
    ''' 
        Make meshgrid for 2D SVM visulaization
    '''
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    ''' 
        Plot contour of 2D SVM 
    '''
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_3D_SVM(model, X, Y):
    ''' 
        Plot contour of 3D SVM 
    '''
    
    z = lambda x,y: (-model.intercept_[0]-model.coef_[0][0]*x -model.coef_[0][1]*y) / model.coef_[0][2]

    tmp = np.linspace(-5,5,30)
    x,y = np.meshgrid(tmp,tmp)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
    ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
    ax.plot_surface(x, y, z(x,y))
    ax.view_init(30, 60)

def svm_linear(x_train, y_train, x_test, y_test, feature_names):
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)

    #get stats
    accuracy,recall, precision, f1 = get_stats(clf, x_test, y_test)
    print('Accuracy {:.4f}, Recall {:.4f}, Precision {:.4f}, F1 {:.4f}'.format(accuracy,recall, precision, f1))

    #plot 2D feature
    if len(feature_names) == 2:
        fig, ax = plt.subplots()
        xx, yy = make_meshgrid(x_train[:, 0], x_train[:, 1], h=.02)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel(feature_names[0])
        ax.set_xlabel(feature_names[1])
        title = "2D Linear SVM with Features: {} and {}".format(feature_names[0], feature_names[1])
        ax.set_title(title)
        plt.suptitle('Accuracy {:.4f}, Recall {:.4f}, Precision {:.4f}, F1 {:.4f}'.format(accuracy,recall, precision, f1), fontsize=8)
        plt.savefig("{}.png".format(title))

    #plot 3D feature
    if len(feature_names) == 3:
        z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
        tmp = np.linspace(-5,5,30)
        x,y = np.meshgrid(tmp,tmp)

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot3D(x_train[y_train==0,0], x_train[y_train==0,1], x_train[y_train==0,2],'ob')
        ax.plot3D(x_train[y_train==1,0], x_train[y_train==1,1], x_train[y_train==1,2],'sr')
        ax.plot_surface(x, y, z(x,y))
        ax.view_init(30, 60)
        title = "3D Linear SVM with Features: {}, {} and {}".format(feature_names[0], feature_names[1], feature_names[2])
        ax.set_title(title, fontsize=8)
        plt.suptitle('Accuracy {:.4f}, Recall {:.4f}, Precision {:.4f}, F1 {:.4f}'.format(accuracy,recall, precision, f1), fontsize=8)
        plt.savefig("{}.png".format(title))

    
        
    return FeatureCombo(feature_names, accuracy, precision, recall, f1)

def svm_linear_2Features():
    ''' 
        Trains and evaluates a linear svm with 2 features
    '''
    #prep data
    df_train, df_test = load_seperate_data("train.csv", "test.csv")
    classes = df_train['satisfaction'].unique()
    sat = {classes[0]: 1,classes[1]: 0}
    df_train['satisfaction'] = [sat[item] for item in df_train['satisfaction']]
    df_test['satisfaction'] = [sat[item] for item in df_test['satisfaction']]
    y_train = df_train['satisfaction'].to_numpy()
    y_test = df_test['satisfaction'].to_numpy()


    #based off of initial distrubtions, these categories naively seem most likely to not be noisy
    #try all combinations, 
    category_to_try = ['Seat comfort', 'Online boarding', 'Leg room service', 'Inflight service', \
        'Inflight entertainment', 'Inflight wifi service']

    combos = itertools.combinations(category_to_try, 2)
    combo_results = []
    for combo in combos:
        print("Training combo {} and {}".format(combo[0] ,combo[1]))
        feature_names = [combo[0] ,combo[1]]
        x_train = df_train[[combo[0] ,combo[1]]].to_numpy()
        x_test = df_test[[combo[0] ,combo[1]]].to_numpy()
        combo_results.append(svm_linear(x_train, y_train, x_test, y_test, feature_names))

    #sort all the combos tested by accuracy            
    combo_results.sort(key=lambda x: x.accuracy)
    #print the top ten                
    for i in range(10):
        combo_features = combo_results[i].features
        print("The {}th best combo is {} and {}".format(i, combo_features[0], combo_features[1]))
    
# svm_linear_2Features()

def svm_linear_3Features():
    ''' 
        Trains and evaluates a linear svm with 3 features
    '''
    # The 1th best combo is Seat comfort , Inflight service and Inflight entertainment
    # The 2th best combo is Seat comfort , Inflight entertainment and Inflight wifi service
    # The 3th best combo is Leg room service , Inflight service and Inflight wifi service
    # The 4th best combo is Inflight service , Inflight entertainment and Inflight wifi service
    # The 5th best combo is Leg room service , Inflight service and Inflight entertainment
    # The 6th best combo is Seat comfort , Leg room service and Inflight service
    # The 7th best combo is Seat comfort , Leg room service and Inflight entertainment
    # The 8th best combo is Leg room service , Inflight entertainment and Inflight wifi service
    # The 9th best combo is Seat comfort , Leg room service and Inflight wifi service

    #prep data
    df_train, df_test = load_seperate_data("train.csv", "test.csv")
    classes = df_train['satisfaction'].unique()
    sat = {classes[0]: 1,classes[1]: 0}
    df_train['satisfaction'] = [sat[item] for item in df_train['satisfaction']]
    df_test['satisfaction'] = [sat[item] for item in df_test['satisfaction']]
    y_train = df_train['satisfaction'].to_numpy()
    y_test = df_test['satisfaction'].to_numpy()


    #based off of initial distrubtions, these categories naively seem most likely to not be noisy
    #try all combinations
    category_to_try = ['Seat comfort', 'Online boarding', 'Leg room service', 'Inflight service', \
        'Inflight entertainment', 'Inflight wifi service']

    combos = itertools.combinations(category_to_try, 3)
    combo_results = []
    for combo in combos:
        print("Training combo {} , {} and {}".format(combo[0] ,combo[1],combo[2]))
        feature_names = [combo[0] ,combo[1],combo[2]]
        x_train = df_train[[combo[0] ,combo[1],combo[2]]].to_numpy()
        x_test = df_test[[combo[0] ,combo[1],combo[2]]].to_numpy()
        combo_results.append(svm_linear(x_train, y_train, x_test, y_test, feature_names))

    #sort all the combos tested by accuracy            
    combo_results.sort(key=lambda x: x.accuracy)
    #print the top ten                
    for i in range(10):
        combo_features = combo_results[i].features
        print("The {}th best combo is {} , {} and {}".format(i, combo_features[0], combo_features[1], combo_features[2]))
    

# svm_linear_3Features()

def svm_linear_AllFeatures():
    ''' 
        Trains and evaluates a linear svm with all features
    '''
    #RESULTS: Accuracy 0.8113, Recall 0.8113, Precision 0.8110, F1 0.8360
    
     #prep data
    df_train, df_test = load_seperate_data("train.csv", "test.csv")
    classes = df_train['satisfaction'].unique()
    sat = {classes[0]: 1,classes[1]: 0}
    df_train['satisfaction'] = [sat[item] for item in df_train['satisfaction']]
    df_test['satisfaction'] = [sat[item] for item in df_test['satisfaction']]
    y_train = df_train['satisfaction'].to_numpy()
    y_test = df_test['satisfaction'].to_numpy()

    #based off of initial distrubtions, these categories naively seem most likely to not be noisy
    category_to_try = ['Seat comfort', 'Online boarding', 'Leg room service', 'Inflight service', \
        'Inflight entertainment', 'Inflight wifi service']
    x_train = df_train[category_to_try ].to_numpy()
    x_test = df_test[category_to_try ].to_numpy()
    svm_linear(x_train, y_train, x_test, y_test, category_to_try )

svm_linear_AllFeatures()