import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

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


def test_classifier(classifier_param):

    if classifier_param.__class__.__name__ == 'DecisionTreeClassifier':
        classifier_param.fit(X_train, y_train)
    else:
        classifier_param.fit(X_train, y_train.values.ravel())

    predictions = classifier_param.predict(X_test)

    print(classifier_param.__class__.__name__)
    print('Precision: ' + str(precision_score(y_test, predictions)))
    print('Recall: ' + str(recall_score(y_test, predictions)))
    print('F1: ' + str(f1_score(y_test, predictions)))

    # For moving data to execl easily
    # print(precision_score(y_test, predictions))
    # print(recall_score(y_test, predictions))
    # print(f1_score(y_test, predictions))


classifiers = [
    LogisticRegression(max_iter=10000),
    DecisionTreeClassifier(),
    LinearSVC(max_iter=10000),
    RandomForestClassifier(),
    MLPClassifier(max_iter=10000),
    KNeighborsClassifier(),
]

for classifier in classifiers:
    test_classifier(classifier)























