import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('otodom.csv')
print(df.head(10).to_string()) #wyświetla wszystko - wszystkie kolumny

print(df.describe().to_string())
print(df.describe().T.to_string()) #zamiana kolum na wiersze, usuwająć ID, którego nie może policzyć/posegregować

#print(df.iloc[2:6, 2:4]) #najpierw wiersze 2 do 5, potem kolumny 2 do 3
#print(df.iloc[2:12:2, 2:4]) # co 2.

#print(df.iloc[:, 1:].corr()) # jaka jest korelacja między kolumnami, kolumny bez ID

#sns.heatmap(df.iloc[:, 1:].corr(), annot=True)
#plt.show()

#bez ceny
#sns.heatmap(df.iloc[:, 2:].corr(), annot=True)
#plt.show()

#sns.displot(df.cena)
#plt.show()

#usunięcie najdroższych
#znależenie minimum (zakres 25%-75%)
_min = df.describe().loc['min', 'cena'] # decribe robi tabelę, w describe kolumny i wiersze są nazwane, ale transponowany po zamianiae kolumn na wiersze
q1 = df.describe().loc['25%', 'cena']
q3 = df.describe().loc['75%', 'cena']
print(_min, q1, q3)

df1 = df[(df.cena >= q1) & (df.cena <= q3)] # tylko cena w przedziale 25%-75%, whcodzi w df i wybiera dane
sns.displot(df1.cena)
plt.show()

X = df1.iloc[:, 2:] #kolumny bez ceny i bez pierwszego indeksu, wszystkie kolumny oprócz ceny i ID
y = df.cena #dane wyjściowe

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # x - wejściowe, y -cena (wyjściowe) - będą pod;zielone 80%:20%
X = X_train + X_test
y = y_train + y_test

model = LinearRegression()
model.fit(X_train, y_train)
print(df1.columns)
print(model.coef_) # wyświetla współczynnik kierunkowy
print(model.score(X_test, y_test)) # pokazuje, jak dobry jest algorytm

