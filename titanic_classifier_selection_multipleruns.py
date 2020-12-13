import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# Scaling our data
# Fitting twice to be sure no data is leaked between the datasets.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)



class TestResult:
    def __init__(self, p, r, f, c):
        self.precision = p
        self.recall = r
        self.f1score = f
        self.classifier = c

    def __str__(self):
        return self.classifier + " precision:" + str(self.precision) + " recall:" + str(self.recall) + " f1:" + str(self.f1score)


def test_classifier(classifier_param):
    if classifier_param.__class__.__name__ == 'DecisionTreeClassifier':
        classifier_param.fit(X_train, y_train)
    else:
        classifier_param.fit(X_train, y_train.values.ravel())

    predictions = classifier_param.predict(X_test)

    return TestResult(
                precision_score(y_test, predictions),
                recall_score(y_test, predictions),
                f1_score(y_test, predictions),
                classifier_param.__class__.__name__
            )


classifier_collumn = 'Classifier'

collumns = [classifier_collumn, 'Precision', 'Recall', 'f1score']
dataframe = pd.DataFrame(columns=collumns)
dataframe[classifier_collumn] = dataframe.Classifier.astype('str')
iterations_per_classifier = 100
count = 0

#Classifiers with deafult parmeterse
classifiers = [
    # LogisticRegression(max_iter=10000),
    # DecisionTreeClassifier(),
    # LinearSVC(max_iter=10000),
    # RandomForestClassifier(),
    # MLPClassifier(max_iter=10000),
    KNeighborsClassifier(),
]

#Loop for testing the classifiers from the collection above
for classifier in classifiers:
    while count < iterations_per_classifier:
        print(classifier.__class__.__name__ + ' : ' + str(count))
        testResult = test_classifier(classifier)
        df2 = pd.DataFrame([[classifier.__class__.__name__, testResult.precision, testResult.recall, testResult.f1score]], columns=collumns)
        df2[classifier_collumn] = df2.Classifier.astype('str')
        dataframe = dataframe.append(df2)
        count += 1
    count = 0

classifier_names = []

for classifier in classifiers:
    classifier_names.append(classifier.__class__.__name__)

for name in classifier_names:
    print('')
    print(name)
    test = dataframe[dataframe[classifier_collumn] == name]
    print(test.describe())












