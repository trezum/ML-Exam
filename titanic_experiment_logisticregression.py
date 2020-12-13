import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, make_scorer, classification_report, \
    precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

# Options set to print more of a df:
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Source:
# https://stackoverflow.com/a/11711637/2067677

df = pd.read_csv('titanic_800.csv', sep=',', header=0)

# One row is missing the embarked tag.
# it is dropped because it is only one sample
# This is done early to not affect the code below,
# splitting in test and train first complicates dropping somewhat.
# Source: https://stackoverflow.com/a/13413845/2067677
df = df[df['Embarked'].notna()]

# Extracting labels
labels = pd.DataFrame(dict(Survived=[]), dtype=int)
labels["Survived"] = df["Survived"].copy()

# Features thrown away
df.drop('Survived', axis=1, inplace=True)
df.drop('PassengerId', axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Encoding text fields to numbers so we can use them for training.
lb = LabelEncoder()
df["Sex"] = lb.fit_transform(df["Sex"])
lb = LabelEncoder()
df["Embarked"] = lb.fit_transform(df["Embarked"])

# Splitting data in test and train sets
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

# Filling put missing datapoints
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# A easy way to check if any NaNs were missed
# Source: https://datatofish.com/rows-with-nan-pandas-dataframe/
print(X_train[X_train.isna().any(axis=1)])
print(X_test[X_test.isna().any(axis=1)])

# Scaling our data
# Fitting twice to be sure no data is leaked between the datasets.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Setup for printing data after transformation (numpy array)
np.set_printoptions(threshold=np.inf)
# https://stackoverflow.com/a/48603911/2067677

#print(X_train)
#print(y_train)


def print_classifier_stats(classifier, xtest, ytest):

    predictions = classifier.predict(xtest)
    print(classifier)
    print(classification_report(ytest, predictions))

    print("Precision: " + str(precision_score(ytest, predictions)))
    print("Recall: " + str(recall_score(ytest, predictions)))
    print("F1 score: " + str(f1_score(ytest, predictions)))

space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = [0.001, 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.5, 1]


model = LogisticRegression(max_iter=10000)
scorer = make_scorer(f1_score)
grid_obj = GridSearchCV(model, space, scoring=scorer, verbose=2, n_jobs=-1)
grid_fit = grid_obj.fit(X_train, y_train.values.ravel())
best_estimator = grid_fit.best_estimator_

print_classifier_stats(best_estimator, X_test, y_test)
print('Grid search done...')
print(space)

# F1 score: 0.7343749999999999
# The best parameters found using grid search
best = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=10000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)


