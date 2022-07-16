import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def feature_stats(data_dir):
    ''' data: input features
        labels: output features
    '''
    df = pd.read_csv(data_dir)

    classes = df['satisfaction'].unique()
    feature_names = list(df.columns)

    for feature in feature_names:
        feature_categories = print(df[feature].unique())
        x = np.arange(len(feature_categories))  # the label locations
        width = 0.35  # the width of the bars

        #if there are more than 10 inputs, probably not categorical and not work looking into
        #for this kind of inital analysis
        postive_counts = []
        negative_counts = []
        if len(feature_categories) <10:
            for feature_cat in feature_categories:
                postive_counts.append(df.loc[(df[feature] == feature_cat) & (df['satisfaction'] == classes[0])].count())
                negative_counts.append(df.loc[(df[feature] == feature_cat) & (df['satisfaction'] == classes[1])].count())

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, postive_counts, width, label=classes[0])
            rects2 = ax.bar(x + width/2, negative_counts, width, label=classes[1])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Number of Data Instances')
            ax.set_title('Category Counts of {} Feature by Satisfaction Class'.format(feature))
            ax.set_xticks(x, feature_categories)
            ax.legend()

            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)

            fig.tight_layout()

            plt.savefig("feature_{}_category_distrubtions.png".format(feature))
