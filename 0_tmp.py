# wykresy
# matplotlib.pyplot

# ładne wykresy
# seabourn

# czytanie danych, praca z danymi
# pandas

# sztuczna integencja
# tensorflow, keras

# testy web
# selenium

# matematyka, obliczenia
# numpy, scipy

# grafika
# tkinter, pyqt6

# gry
# pygame


import seaborn as sns
import matplotlib.pyplot as plt

Xy = [2, 4, 12, 14, 3]
Yki = [4, 2, 5, 7, 8]

plt.plot(Xy, Yki, 'r')
plt.show()


#model zapisany
import  joblib
model_Kamila = joblib.load('Tree_v1.1.model')
print(dir(model_Kamila))
print(model_Kamila._estimator_type, model_Kamila.get_params)