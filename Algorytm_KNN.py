import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Iris.csv')
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