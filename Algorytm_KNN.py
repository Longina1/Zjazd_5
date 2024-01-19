import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris (1).csv') # 4 parametry: sepallenght, sepalwidth, petallength, petalwidth
print(df['class'].value_counts())

species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2

}
df['class_value'] = df['class'].map(species)
print(df['class_value'].value_counts())

sample = np.array([5.6, 3.2, 5.2, 1.45]) #liść



plt.scatter(5.6, 3.2, c='r')
sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class') #długość/szerokość liścia
plt.show()

plt.scatter(5.2, 1.45, c='r')
sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
plt.show()

df['distance'] = (df.sepallength-sample[0]) ** 2 + (df.sepalwidth - sample[1]) ** 2 + (df.petallength - sample[2]) ** 2 + (df.petalwidth - sample[3]) **2

print(df.sort_values('distance').head(10)) # domyślnie powinny być sortowane rosnąco, 10 najbliższych

print(df.head().to_string())

X = df.iloc[: , 0:4] #pierwsze są wiersze
y = df.class_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # x - wejściowe, y wyjściowe - będą podzielone 80%:20%


model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_train, y_train)
print(model.score(X_test, y_test )) # sprawdzenie działania w procentach
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test)))) # ile razy, w którą stronę się pomylił

results = [] # to jest y
for k in range(1, 50):
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model.fit(X_train, y_train)
    results.append(model.score(X_test, y_test))
plt.plot(range(1,50), results) # pierwszy argument to x, a y są w results; x mogą być podane jako zakres
plt.grid() # siatka
plt.show()

#results = [] # to jest y
#for k in range(1, 50):
#    model = KNeighborsClassifier(k)
#    model.fit(X_train, y_train)
#    results.append(model.score(X_test, y_test))
#plt.plot(results)
#plt.show()
