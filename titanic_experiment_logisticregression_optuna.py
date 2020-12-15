import pandas as pd
from os import path
import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, \
    QuantileTransformer, PowerTransformer, PolynomialFeatures, Normalizer


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

    c = trial.suggest_uniform('C', 0.001, 10)

    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    #penalty = trial.suggest_categorical('penalty', ['none', 'l1', 'l2', 'elasticnet'])

    l1_ratio = trial.suggest_float('l1_ratio', 0, 10)

    missing_data_way = trial.suggest_categorical('missing_data_way', ['mean', 'median', 'zero'])

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

    model = LogisticRegression(max_iter=10000, C=c, solver=solver, l1_ratio=l1_ratio)
    model.fit(x_train, y_train.values.ravel())
    predictions = model.predict(x_test)

    return f1_score(y_test, predictions)


def optuna_search():

    if path.exists("logisticregression_study.pkl"):
        study = joblib.load('logisticregression_study.pkl')
    else:
        study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=2000)

    joblib.dump(study, 'logisticregression_study.pkl')
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
# {'C': 0.12339769333139736, 'solver': 'lbfgs', 'l1_ratio': 9.455409069005613, 'missing_data_way': 'mean', 'scaler': 'RobustScaler'}
# Best value:
# 0.8376068376068376




















