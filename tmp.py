#matplotlib.pyplot #wykres
#seaborn # ładny wykres
#pandas # do czytania danych
#tensorflow, keras #sztuczna inteligencja
#selenium #testy webowe
#numpy, scipy #matematyka, obliczenia
#tkinter, pyqt6 #grafika
#pygame #gry

import seaborn as sns
import matplotlib.pyplot as plt

Xy = [1, 2, 3, 4, 5]
Yki = [4, 2, 5, 7, 4]
plt.scatter(Xy, Yki) # pokazuje same punkty
plt.show()

Xy = [1, 2, 3, 4, 5]
Yki = [4, 2, 5, 7, 4]
plt.plot(Xy, Yki, 'r') #x można pominąć, ale jeśli liczba x-ów i y-ów jest różna nie można; 'r' - red, r^ trójkątty
plt.show() # plot łączy punkty, można podać tylko y
