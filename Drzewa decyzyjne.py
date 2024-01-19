import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('iris (1).csv')
print(df['class'].value_counts())

species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2

}
df['class_value'] = df['class'].map(species)
print(df['class_value'].value_counts())

sample = np.array([5.6, 3.2, 5.2, 1.45]) #liść



#plt.scatter(5.6, 3.2, c='r')
#sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class') #długość/szerokość liścia
#plt.show()

#plt.scatter(5.2, 1.45, c='r')
#sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
#plt.show()

X1 = df[['sepallength', 'sepalwidth']] #wycięte 2 kolumny
X2 = df[['petallength', 'petalwidth']]
X3 = df.iloc[:,:4] #X1 + X2 (4 kolumny razem)
y = df.class_value

#X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.2)

model = DecisionTreeClassifier(max_depth=10, min_samples_split=6)
model.fit(X1, y) # potem to samo dla X2
#print(model.score(X_test, y_test))
#print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

plot_decision_regions(X1.values, y.values, model)
plt.show()

#model = DecisionTreeClassifier(max_depth=10, min_samples_split=6)
#model.fit(X2, y)

#plot_decision_regions(X2.values, y.values, model)
#plt.show()

model = DecisionTreeClassifier(max_depth=2, min_samples_split=2)
model.fit(X3, y)

print(model.feature_importances_)
print(model.feature_importances_, X3.columns) #wyświetlenie z kolumnami
print(pd.DataFrame(model.feature_importances_, X3.columns)) # wyświetlenie jako data frame
