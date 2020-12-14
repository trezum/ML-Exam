import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder

# Setting to print without scientific notation
# Source: https://stackoverflow.com/questions/2891790/how-to-pretty-print-a-numpy-array-without-scientific-notation-and-with-given-pre
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

# Options set to print more of a df:
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Source:
# https://stackoverflow.com/a/11711637/2067677


df = pd.read_csv('penguins_size.csv', sep=',', header=0)

# Setting the datatype of the sex collumn to be str so we can more easily work with its content.
df['sex'] = df.sex.astype('str')

# A easy way to check for NaN
# Source: https://datatofish.com/rows-with-nan-pandas-dataframe/
print(df[df.isna().any(axis=1)])

# A easy way to display all rows where sex is not MALE or FEMALE
# Luckily this slection also includes our previous NaN rows in this case, so it can be used to drop the bad data.
print(df[(df['sex'] != 'FEMALE') & (df['sex'] != 'MALE')])

df = df.drop(df[(df['sex'] != 'FEMALE') & (df['sex'] != 'MALE')].index)

# Validating the rows have been dropped
print(df[(df['sex'] != 'FEMALE') & (df['sex'] != 'MALE')])

labels = df["species"].copy()
df.drop("species", axis=1, inplace=True)

# Encoding data
lb = LabelEncoder()
df["island"] = lb.fit_transform(df["island"])
df["sex"] = lb.fit_transform(df["sex"])

# Encoding labels
#labels = lb.fit_transform(labels)

# using manual replace here to more easily evaluate kmeans performance
# random state set so the clusters will be the same
labels = labels.replace(['Adelie'], 0)
labels = labels.replace(['Chinstrap'], 2)
labels = labels.replace(['Gentoo'], 1)

# Using kbest to select the two best features.
selector = SelectKBest(k=2)

print("transformed dimensions of data:")
X_new = selector.fit_transform(df, labels)
print(X_new.shape)

features_selected = selector.get_support(indices=True)

print("scores of features:")
print(selector.scores_)
print("features selected (indexing starts at 0):")
print(features_selected)

print("labels of features selected:")
for i in features_selected:
    print(df.columns[i])

best_feature1 = "culmen_length_mm"
best_feature2 = "flipper_length_mm"

# Creating a plot with the lable categories
colormap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
df.plot(legend=False, kind="scatter", x=best_feature1, c=labels, y=best_feature2, cmap=colormap)

X_best_features = df[[best_feature1, best_feature2]]

# Using kmeans to find labels
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_best_features)
labels_kmeans = kmeans.labels_
clusters = kmeans.cluster_centers_


df.plot(legend=False, kind="scatter", x=best_feature1, c=labels_kmeans, y=best_feature2, cmap=colormap)

# Plotting cluster centers
plt.plot(clusters[0][0], clusters[0][1:], 'ys', markersize=10)
plt.plot(clusters[1][0], clusters[1][1:], 'ys', markersize=10)
plt.plot(clusters[2][0], clusters[2][1:], 'ys', markersize=10)


def correct_observation_calculator(correct, observed):
    same_count = 0
    diff_count = 0

    for c in range(0, correct.size):
        if correct[c] == observed[c]:
            same_count += 1
        else:
            diff_count += 1

    return same_count / (same_count + diff_count)


# Calculating what percentage kmeans got right:
print('kmeans score:' + str(correct_observation_calculator(labels.values, labels_kmeans)))

plt.show()





