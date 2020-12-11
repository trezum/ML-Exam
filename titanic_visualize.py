import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Options set to print more of a df:
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Source:
# https://stackoverflow.com/a/11711637/2067677

df = pd.read_csv('titanic_800.csv', sep=',', header=0)

print(df.describe())
print(df.corr())

# these do not make much sense to draw in a scatter_matrix before encoding
# "PassengerId", "Name", "Sex", "Ticket", "Cabin", "Embarked"

attributes = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
scatter_matrix(df[attributes])
plt.show()