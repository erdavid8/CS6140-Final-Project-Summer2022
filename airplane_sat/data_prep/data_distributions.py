import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from airplane_sat.data_prep.data_prep import load_data

def feature_stats():
    ''' data: input features
        labels: output features
    '''
    df = load_data("train.csv", "test.csv")
    classes = df['satisfaction'].unique()
    df_pos =df.loc[df['satisfaction'] == classes[0]]
    df_neg =df.loc[df['satisfaction'] == classes[1]]
    feature_names = list(df.columns)
    feature_names.remove('satisfaction')
    category_features = []
    for feature in feature_names:
        feature_categories = df[feature].unique()
        
        x = np.arange(len(feature_categories))  # the label locations
        width = 0.35  # the width of the bars

        #if there are more than 10 inputs, probably not categorical and not work looking into
        #for this kind of inital analysis
        postive_counts = []
        negative_counts = []
        if len(feature_categories) <= 10:

            category_features.append(feature)
            if not isinstance(feature_categories, int):
                feature_categories.sort()
            for feature_cat in feature_categories:
                if feature_cat in df_pos[feature].unique():
                    postive_counts.append(df_pos[feature].value_counts()[feature_cat])
                else:
                    postive_counts.append(0)
                if feature_cat in df_neg[feature].unique():
                    negative_counts.append(df_neg[feature].value_counts()[feature_cat])
                else:
                    negative_counts.append(0)


            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, postive_counts, width, label=classes[0])
            rects2 = ax.bar(x + width/2, negative_counts, width, label=classes[1])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Number of Data Instances')
            ax.set_title('Category Counts of {} Feature by Satisfaction Class'.format(feature))
            ax.set_xticks(x)
            ax.set_xticklabels(feature_categories)
            ax.legend()

            fig.tight_layout()
            title = feature.replace("/", "_")
            plt.savefig("feature_{}_category_distrubtions.png".format(title))
    print(category_features)
feature_stats()