import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('heart.csv', comment='#')
print(df.head(10).to_string())
print(df.target.value_counts())

X = df.iloc[:, :-1]
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def drzewo(max_depth):
    print(f'\nmax_depth wynosi {max_depth}')
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

drzewo(2)
drzewo(3)
drzewo(4)
drzewo(5)
drzewo(10)
drzewo(20)


# siatka parametrów
model = DecisionTreeClassifier()

params = {
    'max_depth': range(2, 12),
    'min_samples_split': [2, 3, 4, 5],
    'max_features': range(2, 8),
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid = GridSearchCV(model, params, scoring='accuracy', cv=10, verbose=2)
grid.fit(X_train, y_train)

print(f'najlepsze parametry: {grid.best_params_}')
print(f'najlepszy wynik: {grid.best_score_}')
print(f'najlepszy estymator {grid.best_estimator_}')
print(pd.DataFrame(grid.best_estimator_.feature_importances_,X.columns).sort_values(0, ascending=False))
y_pred = grid.best_estimator_.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, y_pred)))