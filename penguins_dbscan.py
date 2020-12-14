from os import path
import joblib
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, \
    QuantileTransformer, PowerTransformer, PolynomialFeatures, Normalizer

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

# Using manual replace here to more easily evaluate kmeans performance
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
colormap = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#000000'])


def correct_observation_calculator(correct, observed):
    same_count = 0
    diff_count = 0

    for c in range(0, correct.size):
        if correct[c] == observed[c]:
            same_count += 1
        else:
            diff_count += 1

    return same_count / (same_count + diff_count)

# scaler = MinMaxScaler()
# df_scaled = scaler.fit_transform(df)
# df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
#df = df_scaled

db = DBSCAN(eps=78.75139345732582, min_samples=9).fit(df)
db = DBSCAN(eps=78.75139345732582, min_samples=9).fit(df)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# #assign true to the all the samles indicated as core_points by the algorithm
# core_samples_mask[db.core_sample_indices_] = True

labels_dbscan = db.labels_
n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_ = list(labels).count(-1)

print('dbscan clusters: ' + str(n_clusters_))
print('dbscan noise: ' + str(n_noise_))
df.plot(legend=False, kind="scatter", x=best_feature1, c=labels_dbscan, y=best_feature2, cmap=colormap)

print('dbscan score:' + str(correct_observation_calculator(labels.values, labels_dbscan)))

plt.show()


def first_try_at_optimizing_dbscan():
    for k in range(1, 20):
        print(k)
        eps = 0.001
        while eps <= 20.0:
            db = DBSCAN(eps=eps, min_samples=k).fit(df)
            n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

            if n_clusters_ == 3:
                print(eps)
                print(k)
            eps = eps + 0.001


def select_scaler_by_parmeter(scaler):
    if scaler == 'MinMaxScaler':
        return MinMaxScaler()
    if scaler == 'MaxAbsScaler':
        return MaxAbsScaler()
    if scaler == 'RobustScaler':
        return RobustScaler()
    if scaler == 'StandardScaler':
        return StandardScaler()
    if scaler == 'QuantileTransformer':
        return QuantileTransformer()
    if scaler == 'PowerTransformer':
        return PowerTransformer()
    if scaler == 'PolynomialFeatures':
        return PolynomialFeatures()
    if scaler == 'Normalizer':
        return Normalizer()


def objective(trial):
    min_samples = trial.suggest_int('min_samples', 1, 20)
    eps = trial.suggest_uniform('eps', 0.1, 100)
    scaler = trial.suggest_categorical('scaler',
                                       ['MinMaxScaler',
                                        'MaxAbsScaler',
                                        'RobustScaler',
                                        'StandardScaler',
                                        'QuantileTransformer',
                                        'PowerTransformer',
                                        'PolynomialFeatures'])

    # Try differnt scalers.
    scaled_data = select_scaler_by_parmeter(scaler).fit_transform(df)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_data)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    # The further n_clusters is from 3 the worse the score should be.
    if n_clusters != 3:
        return abs(n_clusters - 3) * -1

    # If n_clusters is 3 the score should be the accuracy calculated by the method, correct_observation_calculator
    # We need to try all combinations for the cluster placement for calculating the score.

    bestscore = 0
    combinations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

    for c in combinations:
        labels_db = db.labels_

        # Replacing the numbers with temp values
        labels_db = np.where(labels_db == 0, 3, labels_db)
        labels_db = np.where(labels_db == 1, 4, labels_db)
        labels_db = np.where(labels_db == 2, 5, labels_db)

        # Replacing with values for the curren test
        labels_db = np.where(labels_db == 3, c[0], labels_db)
        labels_db = np.where(labels_db == 4, c[1], labels_db)
        labels_db = np.where(labels_db == 5, c[2], labels_db)

        # Evaluating how well the labels fit for the current test
        current_value = correct_observation_calculator(labels.values, labels_db)
        if bestscore <= current_value:
            bestscore = current_value

    return bestscore


def optuna_search():

    if path.exists("dbscan_study.pkl"):
        study = joblib.load('dbscan_study.pkl')
    else:
        study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=5000)

    joblib.dump(study, 'dbscan_study.pkl')
    print('Optuna search done...')
    print('Best trial:')
    print(study.best_trial)
    print('Best params:')
    print(study.best_params)
    print('Best value:')
    print(study.best_value)


# Sarting the search
optuna_search()

