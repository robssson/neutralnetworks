import numpy as np
import xlrd
import pandas as pd
import Tkinter as Tk
import os
from functools import partial

workbook = xlrd.open_workbook('daneOK.xlsx')
worksheet = workbook.sheet_by_index(0)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

def sinus(x):
    return np.sin(x)

def bipolarna(x):
    return 2/(1 + np.exp(-x)) - 1

def derivatives_bipolarna(x):
    return (2 * np.exp(x)) / ((np.exp(x) + 1)**2)

#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#     print output - y

def simulation(entryHiddenLayer, listBoxValue, valueOfScale):
    #print listBoxValue
    suma = 0
    value = 0
    valueTest = 0
    valueTest1 = 0

    value = valueOfScale * 0.8
    valueTest = round(valueOfScale - value)
    valueTest1 = (valueTest) + valueOfScale
    value = round(value)

    popup = Tk.Toplevel(root)
    popup.grab_set()
    w = 220 # width for the Tk root
    h = 410 # height for the Tk root
    ws = popup.winfo_screenwidth() 
    hs = popup.winfo_screenheight() 
    x = (ws/2) - (w/2)
    hs = popup.winfo_screenheight() 
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    popup.geometry('%dx%d+%d+%d' % (w, h, x, y))
    popup.resizable(width=False, height=False)
    popup.title("Simulation")
    X = np.array([[]])
    y = np.array([])
    for i in range(1,int(value)):
        listTwo = []
        for j in range(0,18):
            if (j==0):
                #y = np.append(y, np.array([worksheet.cell(i,j).value]))
                d = np.array([[worksheet.cell(i,j).value]])
                y = np.append(y, d)
            else:
                listTwo.append(worksheet.cell(i,j).value)
        X = np.append(X, listTwo)
        del listTwo[:]
    y = np.array(y).reshape(int(value)-1,1)
    X = np.array(X).reshape(int(value)-1,17)

    epoch = 3500 #Setting training iterations # Jedna iteracji w ktorej zawiera sie propagacja przednia i wsteczna
    lr=0.1 #Setting learning rate # Wielkosc wagi jest kontrolowana przez ten parametr 
    #inputlayer_neurons = 9
    inputlayer_neurons = X.shape[1] #number of features in data set #zwraca rozmiar elementu w tablicy X
    hiddenlayer_neurons = int(entryHiddenLayer)
    output_neurons = 1 #number of neurons at output layer

    wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons)) # waga dla ukrytych warstw
    bh=np.random.uniform(size=(1,hiddenlayer_neurons)) # bias dla ukrytych warstw
    wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons)) # waga dla wyjsciowej warstwy
    bout=np.random.uniform(size=(1,output_neurons)) # bias dla wyjsciowej warstwy


    for i in range(epoch):

    #Forward Propogation
        hidden_layer_input1=np.dot(X,wh) # Iloczyn macierzy wejsciowej i wag przypisanych do krawedzi pomiedzy wartstwami input i ukrytymi
        hidden_layer_input=hidden_layer_input1 + bh # Nastepnie dodaje sie odchylenia neuronow ukrytych warstw do odpowiednich warstw (transformacja liniowa)
        if (listBoxValue == "Sigmoidalna"):
            hiddenlayer_activations = sigmoid(hidden_layer_input) # Transformacja nieliniowa za pomoca funkcji aktywacji (Sigmoid) - zwraca wynik jako 1/(1 + exp(-x)).
        elif (listBoxValue == "Sinus"):
            hiddenlayer_activations = sinus(hidden_layer_input) # Transformacja nieliniowa za pomoca funkcji aktywacji (Sigmoid) - zwraca wynik jako 1/(1 + exp(-x)).
        elif (listBoxValue == "Bipolarna"):
            hiddenlayer_activations = bipolarna(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,wout) # Transformacja liniowa aktywacji ukrytej warstwy
        output_layer_input= output_layer_input1 + bout # Wez macierz z wagami i dodaj bias neuronu warstwy wyjsciowej
        if (listBoxValue == "Sigmoidalna"):
            output = sigmoid(output_layer_input) # Zastosuj funkcje aktywacyjna (Moze byc inna niz sigmoid)
        elif (listBoxValue == "Sinus"):
            output = sinus(output_layer_input)
        elif (listBoxValue == "Bipolarna"):
            output = bipolarna(output_layer_input)
    #Backpropagation
        E = y - output # Porownuje przewidywanie z faktycznym wynikiem # Blad jest srednia strata kwadratowa ((Y-t)^2)/2

        if (listBoxValue == "Sigmoidalna"):
            slope_output_layer = derivatives_sigmoid(output) # Obliczam gradient neuronow warstwy wyjsciowej # Obliczam jako pochodne nielioniowych aktywacji x przy kazdej warstwie
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations) # Obliczam gradient neuronow warstwy ukrytej
        elif (listBoxValue == "Sinus"):
            slope_output_layer = sinus(output) # Obliczam gradient neuronow warstwy wyjsciowej # Obliczam jako pochodne nielioniowych aktywacji x przy kazdej warstwie
            slope_hidden_layer = sinus(hiddenlayer_activations) # Obliczam gradient neuronow warstwy ukrytej
        elif (listBoxValue == "Bipolarna"):
            slope_output_layer = derivatives_bipolarna(output) # Obliczam gradient neuronow warstwy wyjsciowej # Obliczam jako pochodne nielioniowych aktywacji x przy kazdej warstwie
            slope_hidden_layer = derivatives_bipolarna(hiddenlayer_activations)
        d_output = E * slope_output_layer # obliczam zmiane (delte) na warstwie wyjsciowej w zaleznosci od gradientu bledu * gradient warstwy wyjsciowej aktywacji

        Error_at_hidden_layer = d_output.dot(wout.T) # Na tym etapie, blad z powrotem propaguje do sieci, co oznacza blad na warstwie ukrytej. Bierzemy iloczyn punktowy delty warstwy wyjsciowej i wage parametrow na krawedziach miedzy ukryta warstwa a wyjsciowa

        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer # obliczam zmiane (delte) na ukrytej warstwie, mnozac blad na ukrytej warstwie ze spadkiem aktywacji na ukrytej warstwie

        wout += hiddenlayer_activations.T.dot(d_output) *lr # aktualizuje wagi na wyjsciu i w ukrytej warstwie. Wagi w sieci mozna aktualizowac na podstawie bledow obliczonych na podstawie przykladowych szkolen

        # wout = wout + matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate
        # wh =  wh + matrix_dot_product(X.Transpose,d_hiddenlayer)*learning_rate
        # bias at output_layer =bias at output_layer + sum of delta of output_layer at row-wise * learning_rate
        # bias at hidden_layer =bias at hidden_layer + sum of delta of output_layer at row-wise * learning_rate 

        bout += np.sum(d_output, axis=0,keepdims=True) *lr # aktualizacja biasow na warstwie wyjsciowej
        wh += X.T.dot(d_hiddenlayer) *lr
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr # aktualizacja biasow na warstwie 
        

    I = np.array([[]])
    J = np.array([])
    for i in range(int(value+1),int(valueOfScale+1)):
        listTwo = []
        for j in range(0,18):
            if (j==0):
                #y = np.append(y, np.array([worksheet.cell(i,j).value]))
                d = np.array([[worksheet.cell(i,j).value]])
                J = np.append(J, d)
            else:
                listTwo.append(worksheet.cell(i,j).value)
        I = np.append(I, listTwo)
        del listTwo[:]
    J = np.array(J).reshape(int(valueTest),1) # Y
    I = np.array(I).reshape(int(valueTest),17) # X

    for i in range(0,len(J)):
        out = 0
        a = np.dot(I[i], wh)
        b = a + bh
        if (listBoxValue == "Sinus"):
            c = sinus(b)
        elif (listBoxValue == "Sigmoidalna"):
            c = sigmoid(b)
        elif (listBoxValue == "Bipolarna"):
            c = bipolarna(b)
        d = np.dot(c, wout)
        e = d + bout
        if (listBoxValue == "Sigmoidalna"):
            out = sigmoid(e)
        elif (listBoxValue == "Sinus"):
            out = sinus(e)
        elif (listBoxValue == "Bipolarna"):
            out = bipolarna(e)
        if (J[i] - out > 0.40 or J[i] - out < -0.40):
            suma += 1
    suma1 = 0
    suma1 = 1 - (suma / valueTest)
    drawEntries(popup, wh, bh, bout, wout, listBoxValue, suma1)

def drawEntries(self, wh, bh, bout, wout, listBoxValue, suma1):
    
    var = "Prawidlowe: " + str(float(format(suma1, '.2f'))) + " %"
    label0 = Tk.Label(self, text=var)
    label0.grid(row=0)
    label1 = Tk.Label(self, text="Parametr 1")
    label1.grid(row=1)
    entry1 = Tk.Entry(self)
    entry1.grid(row=1, column=1)
    label2 = Tk.Label(self, text="Parametr 2")
    label2.grid(row=2)
    entry2 = Tk.Entry(self)
    entry2.grid(row=2, column=1)
    label3 = Tk.Label(self, text="Parametr 3")
    label3.grid(row=3)
    entry3 = Tk.Entry(self)
    entry3.grid(row=3, column=1)
    label4 = Tk.Label(self, text="Parametr 4")
    label4.grid(row=4)
    entry4 = Tk.Entry(self)
    entry4.grid(row=4, column=1)
    label5 = Tk.Label(self, text="Parametr 5")
    label5.grid(row=5)
    entry5 = Tk.Entry(self)
    entry5.grid(row=5, column=1)
    label6 = Tk.Label(self, text="Parametr 6")
    label6.grid(row=6)
    entry6 = Tk.Entry(self)
    entry6.grid(row=6, column=1)
    label7 = Tk.Label(self, text="Parametr 7")
    label7.grid(row=7)
    entry7 = Tk.Entry(self)
    entry7.grid(row=7, column=1)
    label8 = Tk.Label(self, text="Parametr 8")
    label8.grid(row=8)
    entry8 = Tk.Entry(self)
    entry8.grid(row=8, column=1)
    label9 = Tk.Label(self, text="Parametr 9")
    label9.grid(row=9)
    entry9 = Tk.Entry(self)
    entry9.grid(row=9, column=1)
    label10 = Tk.Label(self, text="Parametr 10")
    label10.grid(row=10)
    entry10 = Tk.Entry(self)
    entry10.grid(row=10, column=1)
    label11 = Tk.Label(self, text="Parametr 11")
    label11.grid(row=11)
    entry11 = Tk.Entry(self)
    entry11.grid(row=11, column=1)
    label12 = Tk.Label(self, text="Parametr 12")
    label12.grid(row=12)
    entry12 = Tk.Entry(self)
    entry12.grid(row=12, column=1)
    label13 = Tk.Label(self, text="Parametr 13")
    label13.grid(row=13)
    entry13 = Tk.Entry(self)
    entry13.grid(row=13, column=1)
    label14 = Tk.Label(self, text="Parametr 14")
    label14.grid(row=14)
    entry14 = Tk.Entry(self)
    entry14.grid(row=14, column=1)
    label15 = Tk.Label(self, text="Parametr 15")
    label15.grid(row=15)
    entry15 = Tk.Entry(self)
    entry15.grid(row=15, column=1)
    label16 = Tk.Label(self, text="Parametr 16")
    label16.grid(row=16)
    entry16 = Tk.Entry(self)
    entry16.grid(row=16, column=1)
    label17 = Tk.Label(self, text="Parametr 17")
    label17.grid(row=17)
    entry17 = Tk.Entry(self)
    entry17.grid(row=17, column=1)
    buttonSimulate = Tk.Button(self, text="Simulate",command= lambda: makeSimulation(entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry8, entry9, entry10, entry11, entry12, entry13, entry14, entry15, entry16, entry17, wh, bh, bout, wout, listBoxValue))
    buttonSimulate.grid(column=1)
    
def makeSimulation(entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry8, entry9, entry10, entry11, entry12, entry13, entry14, entry15, entry16, entry17, wh, bh, bout, wout, listBoxValue):
    a = np.dot([float(entry1.get()),float(entry2.get()),float(entry3.get()),float(entry4.get()),float(entry5.get()),float(entry6.get()),float(entry7.get()),float(entry8.get()),float(entry9.get()),float(entry10.get()),float(entry11.get()),float(entry12.get()),float(entry13.get()),float(entry14.get()),float(entry15.get()),float(entry16.get()),float(entry17.get())] ,wh)
    b = a + bh
    if (listBoxValue == "Sinus"):
        c = sinus(b)
    elif (listBoxValue == "Sigmoidalna"):
        c = sigmoid(b)
    elif (listBoxValue == "Bipolarna"):
        c = bipolarna(b)
    d = np.dot(c, wout)
    e = d + bout
    if (listBoxValue == "Sigmoidalna"):
        out = sigmoid(e)
    elif (listBoxValue == "Sinus"):
        out = sinus(e)
    elif (listBoxValue == "Bipolarna"):
        out = bipolarna(e)
    print out
    
    new = Tk.Toplevel(root)
    new.grab_set()
    w = 120 # width for the Tk root
    h = 22 # height for the Tk root
    ws = new.winfo_screenwidth() 
    hs = new.winfo_screenheight() 
    x = (ws/2) - (w/2)
    hs = new.winfo_screenheight() 
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    new.geometry('%dx%d+%d+%d' % (w, h, x, y))
    new.resizable(width=False, height=False)
    new.title("Result")
    var = ""

    if (out < 0.20):
        var = "Przewidywanie: 1"
    elif (out > 0.20 and out < 0.45):
        var = "Przewidywanie: 1X"
    elif (out > 0.45 and out < 0.55):
        var = "Przewidywanie: X"
    elif (out > 0.55 and out < 0.80):
        var = "Przewidywanie: 2X"
    elif (out > 0.80 and out <= 1):
        var = "Przewidywanie: 2"
    labels = Tk.Label(new, text=var)
    labels.pack()

root = Tk.Tk()
w = 220 # width for the Tk root
h = 300 # height for the Tk root
ws = root.winfo_screenwidth() 
hs = root.winfo_screenheight() 
x = (ws/2) - (w/2)
hs = root.winfo_screenheight() 
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.resizable(width=False, height=False)
root.title("Neutral networks")

labelHiddenLayer = Tk.Label(root, text="Ilosc ukrytych warstw")
chooseFunction = Tk.Label(root, text="Wybierz funkcje aktywacji")
numberOfMatches = Tk.Label(root, text="Ilosc meczy do nauki")
entryHiddenLayer = Tk.Entry(root)
skala = Tk.Scale(from_=0, to=700, length=200, orient=Tk.HORIZONTAL)
path = r"lion2.gif"
photo = Tk.PhotoImage(file=path)
image = Tk.Label(root, image=photo)

labelHiddenLayer.grid(row=0)
entryHiddenLayer.grid(row=1)
chooseFunction.grid(row=2)

listBox = Tk.Listbox(root, width=12, height=3)
listBox.insert(1, "Sigmoidalna")
listBox.insert(2, "Sinus")
listBox.insert(3, "Bipolarna")

listBox.grid(row=3,pady=3)
#HiddenLayer = entryHiddenLayer.get()
#listBoxValue = listBox.get(listBox.curselection())
numberOfMatches.grid(row=4)
skala.grid(row=5)
#valueOfScale = skala.get()
button = Tk.Button(root, text="Ucz siec", command= lambda: simulation(entryHiddenLayer.get(), listBox.get(listBox.curselection()), skala.get()), image=photo)
button.grid(row=6, pady=3)

root.mainloop()  

'''
a = np.dot([0.39,0.62,0.5,0,0,0.5,0,0,0,0.5,0.5,1,1,0.5,1,1,1] ,wh)
b = a + bh
c = sigmoid(b)
d = np.dot(c, wout)
e = d + bout
out = sigmoid(e)
'''

