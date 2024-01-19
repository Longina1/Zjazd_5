import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np




df = pd.read_csv('diabetes (1).csv')
print(df.head(5).to_string())
print(df.describe().T.to_string())

print(df.isna().sum()) #puste wartości
print(df.outcome.value_counts())

for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age']: #po wszystkich koluimach bez outcome
    df[col].replace(0, np.NaN, inplace=True) #wykona, jeśli jesy True, alternatywnie df[col] = ...
    mean_ = df[col].mean() #liczy średnią, zmienna mean
    df[col].replace(np.NaN, mean_, inplace=True) # wrzuci średnią tam, gdzie nie byłożadnej wartości


print(df.isna().sum()) #sprawdza, ile zostało pustych wartości
print(df.describe().T.to_string())

X = df.iloc[: , :-1] #bez ostatniej kolumny
y = df.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # x - wejściowe, y -cena (wyjściowe) - będą pod;zielone 80%:20%
X = X_train + X_test
y = y_train + y_test

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test)))) #y prawdziwy (czy zdrowy czy chory) i y predict (obliczony orzez algortytm, czy zdrowy czy chory); prawdziwy y to y_test, y policzony to X-test

print('\nData change')
df1 = df.query('outcome==0').sample(n=500) #filtrowanie zdrowych 0 i wziąć 500 prób
df2 = df.query('outcome==1').sample(n=500) #filtrowanie chorych i wybranie 500 prób
df3 = pd.concat([df1, df2]) #sumowanie df1 i df2

X = df3.iloc[: , :-1]
y = df3.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # x - wejściowe, y -cena (wyjściowe) - będą pod;zielone 80%:20%
X = X_train + X_test
y = y_train + y_test
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))