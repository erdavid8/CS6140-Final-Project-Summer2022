from sklearn import svm
import numpy as np 
import matplotlib.pyplot as plt
from  airplane_sat.data_prep.data_prep import load_seperate_data

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
    
    z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

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

    #plot 2D feature
    if len(feature_names) == 2:
        fig, ax = plt.subplots()
        xx, yy = make_meshgrid(x_train[:, 0], x_train[:, 1], h=.02)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel(feature_names[0])
        ax.set_xlabel(feature_names[1])
        title = "2D Linar SVM with Features: {} and {}".format(feature_names[0], feature_names[1])
        ax.set_title(title)
        plt.savefig("{}.png".format(title))
    return FeatureCombo(feature_names, accuracy = 0, precision =0, recall =0, f1=0)

def svm_linear_2Features():
    ''' 
        Trains and evaluates a linear svm with 2 features
    '''
    combos_tested = set()
    combos = []
    df_train, df_test = load_seperate_data("train.csv", "test.csv")
    sat = {'satisfied': 1,'neutral or dissatisfied': 0}
    df_train['satisfaction'] = [sat[item] for item in df_train['satisfaction']]
    df_test['satisfaction'] = [sat[item] for item in df_train['satisfaction']]
    y_train = df_train['satisfaction'].to_numpy()
    y_test = df_test['satisfaction'].to_numpy()


    #based off of initial distrubtions, these categories naively seem most likely to not be noisy
    #try all combinations
    category_to_try = ['Seat comfort', 'Online boarding', 'Leg room service', 'Inflight service', \
        'Inflight entertainment', 'Inflight wifi service','Flight Distance', 'Age']
    for category1 in category_to_try:
        for category2 in category_to_try:
            if category1 != category2 and (category1 , category2) not in combos_tested:
                combos_tested.add((category1 , category2))
                combos_tested.add((category2 , category1))
                feature_names = [category1 , category2]
                x_train = df_train[[category1 , category2]].to_numpy()
                x_test = df_test[[category1 , category2]].to_numpy()
                combos.append(svm_linear(x_train, y_train, x_test, y_test, feature_names))

    #sort all the combos tested by accuracy            
    combos.sort(key=lambda x: x.accuracy)
    #print the top ten                
    for i in range(10):
        combo_features = combos[i].features
        print("The {}th best combo is {} and {}".format(i, combo_features[0], combo_features[1]))
    
def svm_linear_3Features():
    ''' 
        Trains and evaluates a linear svm with 3 features
    '''
    #based off of initial distrubtions, these categories naively seem most likely to not be noisy
    category_to_try = ['Seat comfort', 'Online boarding', 'Leg room service', 'Inflight service', \
        'Inflight entertainment', 'Inflight wifi service','Flight Distance', 'Age']

def svm_linear_AllFeatures():
    ''' 
        Trains and evaluates a linear svm with all features
    '''
    #based off of initial distrubtions, these categories naively seem most likely to not be noisy
    category_to_try = ['Seat comfort', 'Online boarding', 'Leg room service', 'Inflight service', \
        'Inflight entertainment', 'Inflight wifi service','Flight Distance', 'Age']