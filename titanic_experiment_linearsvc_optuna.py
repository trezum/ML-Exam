import pandas as pd
from os import path
import joblib
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, \
    QuantileTransformer, PowerTransformer, PolynomialFeatures, Normalizer
from sklearn.svm import LinearSVC


def prep_raw_data():
    df = pd.read_csv('titanic_800.csv', sep=',', header=0)

    # Used for printing the value distribution
    #print(df['Embarked'].value_counts())
    # Setting the missing embarked value to the most typical value
    # Maybe this could be improved by using clutering and assigning the value from the most similar individuals
    #df[(df['Embarked'] != 'S') & (df['Embarked'] != 'Q') & (df['Embarked'] != 'C')] = 'S'
    df['Embarked'].fillna('S', inplace=True)
    # One row is missing the embarked tag.
    #df = df[df['Embarked'].notna()]


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

    return df, labels


def prepare_data_drop(X_train, X_test, y_train, y_test):
    pass


def prepare_data_mean(xtrain, xtest, ytrain, ytest):
    xtrain = xtrain.fillna(xtrain.mean())
    xtest = xtest.fillna(xtest.mean())
    return xtrain, xtest, ytrain, ytest


def prepare_data_median(xtrain, xtest, ytrain, ytest):
    xtrain = xtrain.fillna(xtrain.median())
    xtest = xtest.fillna(xtest.median())
    return xtrain, xtest, ytrain, ytest


def prepare_data_zero(xtrain, xtest, ytrain, ytest):
    xtrain = xtrain.fillna(0.0)
    xtest = xtest.fillna(0.0)
    return xtrain, xtest, ytrain, ytest


def handle_missing_data(xtrain, xtest, ytrain, ytest, way):
    # if way == 'drop':
    #     return prepare_data_drop(xtrain, xtest, ytrain, ytest,)
    if way == 'mean':
        return prepare_data_mean(xtrain, xtest, ytrain, ytest,)
    if way == 'median':
        return prepare_data_median(xtrain, xtest, ytrain, ytest,)
    if way == 'zero':
        return prepare_data_zero(xtrain, xtest, ytrain, ytest,)


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


def scale_data_with(scaler, xtrain, xtest, ytrain, ytest):
    scaler.fit(xtrain)
    xtrain = scaler.transform(xtrain)
    scaler.fit(xtest)
    xtest = scaler.transform(xtest)
    return xtrain, xtest, ytrain, ytest


def objective(trial):
    # to work with these parmeters a more elaborate set ups needed with some ifs
    # loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
    # penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    # dual = trial.suggest_int('dual', 0, 1)

    missing_data_way = trial.suggest_categorical('missing_data_way', ['mean', 'median', 'zero'])
    c = trial.suggest_uniform('C', 0.001, 10)
    scaler = trial.suggest_categorical('scaler',
                                       ['MinMaxScaler',
                                        'MaxAbsScaler',
                                        'RobustScaler',
                                        'StandardScaler',
                                        'QuantileTransformer',
                                        'PowerTransformer',
                                        'PolynomialFeatures'])

    df, labels = prep_raw_data()
    x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = handle_missing_data(x_train, x_test, y_train, y_test, missing_data_way)
    x_train, x_test, y_train, y_test = scale_data_with(select_scaler_by_parmeter(scaler), x_train, x_test, y_train, y_test)

    model = LinearSVC(max_iter=10000, C=c, )
    model.fit(x_train, y_train.values.ravel())
    predictions = model.predict(x_test)

    return f1_score(y_test, predictions)


def optuna_search():

    if path.exists("linearsvc_study.pkl"):
        study = joblib.load('linearsvc_study.pkl')
    else:
        study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=1000)

    joblib.dump(study, 'linearsvc_study.pkl')
    print('Optuna search done...')
    print('Best trial:')
    print(study.best_trial)
    print('Best params:')
    print(study.best_params)
    print('Best value:')
    print(study.best_value)


# Sarting the search
optuna_search()


# Best params:
# {'missing_data_way': 'mean', 'C': 9.72911587459442, 'scaler': 'StandardScaler'}
# Best value:
# 0.8067226890756303




















