import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#pd.read_csv('weight-height (1).csv')
df = pd.read_csv('weight-height (1).csv')
print(df.head(3)) # 3 pierwsze wiersze
print(df.Gender.value_counts()) #ile jest kobiet/mężczyzn
df.Height *= 2.54
df.Weight /= 2.2
print(f'Recalibrated\n {df.head(3)}')

sns.histplot(df.Weight) #waga wszyskich
#98plt.show()

sns.histplot(df.query("Gender=='Male'").Weight) # waga tylko mężczyzn
#plt.show()

sns.histplot(df.query("Gender=='Female'").Weight) # waga tylko kobiet
#plt.show()

#plt.show()

#zmiana Male i Female na liczby
df = pd.get_dummies(df)#wejdźw pandas, wykonaj metodę get dummies, zamień nazwę na True/False
print(df.head())

#usunięcie kolumny MAle
del (df['Gender_Male'])
print(df.head())

#dane wejściowe (niezależne): height, gender; dane zależne (wyjściowe) - weight

model = LinearRegression()
model.fit(df[['Height', 'Gender_Female']], df['Weight']) #X - height i gender; y - weight
print(f'Współczynnik kierunkowy to: {model.coef_},\n a wyraz wolny to: {model.intercept_}')

print(f'Wzór na wagę: Height * {model.coef_[0]} + Gender * {model.coef_[1]} + {model.intercept_}')