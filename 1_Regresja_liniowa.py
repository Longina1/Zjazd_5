import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
print(df.head(3))    #wyświetl 3 pierwsze wiersze
print(df.Gender.value_counts())   #policz ile jest kobiet, ile mężczyzn
df.Height *= 2.54
df.Weight /= 2.2
print(f'po zmianie jednostek\n {df.head(3)}')

# wyświetlanie histogramu - waga wszystkich ludzi
sns.histplot(df.Weight)

# wyświetlanie histogramu - waga mężczyzn
sns.histplot(df.query("Gender=='Male'").Weight)

# wyświetlanie histogramu - waga kobiet
sns.histplot(df.query("Gender=='Female'").Weight)

plt.show()

#zmiana 'male' i 'female' na true i false
df = pd.get_dummies(df)
print(df.head())
#usunięcie kolumny 'male'
del (df["Gender_Male"])
print(df.head())
# dane wejściowe (niezależne) - height, gender,   dane wyjsciowe (zależne) - waga

model = LinearRegression()
model.fit(df[['Height', 'Gender_Female']], df['Weight'])
print(f'wspolczynnki kierunkowy: {model.coef_},\nwyraz wolny: {model.intercept_}')

print(f'wzór: Height * {model.coef_[0]} + Gender * {model.coef_[1]} + {model.intercept_}')