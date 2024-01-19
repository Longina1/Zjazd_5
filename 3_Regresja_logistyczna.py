import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np

df = pd.read_csv('diabetes.csv')
print(df.head(5).to_string())
print(df.describe().T.to_string())
print(df.isna().sum())   # puste wartości
print(df.outcome.value_counts())

for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age']:
    df[col].replace(0, np.NaN, inplace=True)
    mean_ = df[col].mean()
    df[col].replace(np.NaN, mean_, inplace=True)

print(df.isna().sum())
print(df.describe().T.to_string())

X = df.iloc[: , :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('\nZmiana danych')
df1 = df.query('outcome==0').sample(n=500)
df2 = df.query('outcome==1').sample(n=500)
df3 = pd.concat([df1, df2])

X = df3.iloc[: , :-1]
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))