#!python

'''
logistic classifier dual
'''

import csv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math



# funtion to read the dataset (first function called of the script)
# primeira linha do dataset tem o nr de imagens e os seus pixeis
# 600 8 8 -> significa que tem 600 imagens de 8x8 pixeis
# data -> lista de listas com os valores de cada linha
def read_asc_data(filename):
    f = open(filename, 'r')
    #linha do documento
    tmp_str = f.readline()
    #elementos da linha acima iniciada
    tmp_arr = tmp_str[:-1].split(' ')
    N = int(tmp_arr[0])
    n_row = int(tmp_arr[1])
    n_col = int(tmp_arr[2])
    print("N=%d, row=%d, col=%d" %(N,n_row,n_col))
    # inicializar os dados a zero
    data = np.zeros([N,n_row*n_col+1])
    #ciclo que preenche a matriz data com os dados
    for n in range(N):
        tmp_str = f.readline()
        tmp_arr = tmp_str[:-1].split(' ')
        for i in range(n_col*n_row+1):
            data[n][i] = int(tmp_arr[i])
    f.close()
    print(data)
    return N, n_row, n_col, data 
    

# funçao para visualizar os dados num grafico 
def plot_data(row,col,n_row,n_col,data):
    fig=plt.figure(figsize=(row,col))
    for n in range(1, row*col +1):
        img=np.reshape(data[n-1][:-1],(n_row,n_col))
        fig.add_subplot(row, col, n)
        plt.imshow(img,interpolation='none',cmap='binary')
    plt.show()





# -------------- Classifier Code -----------------

# finçao aux que calcula a funçao sigmoidal
def sigmoid(s):
    large = 30
    if s < -large: s = -large
    if s > large: s = large
    return (1/(1 + np.exp(-s)))

# funçao que calcula a previsao
# no dual tenho que mudar isto para ewT (transposta) -> predictor(ewT, x)
# lambda substitui o ew
def predictor (x, X, lambd):
    sum = np.zeros([I+1])
    #componente nr 0 // tratada a parte
    bias = 0
    for n in X:
        sum = sum + lambd[n] * X[n]
        bias = bias + lambd[n]
    sum = np.dot(sum,x) + bias
    

    sigma = sigmoid (sum)
    return sigma


# Funçao que calcula o custo ou erro, compara o valor previsto com o real e avalia a precisao da previsao
# A funçao de custo é igual ao do primal, o que muda á a de previsao
def cost(X, Y, N, lambd):
    # Custo
    En = 0
    # nao sei o que significa
    epsi = 1.e-12 # valor de segurança
    for n in range(N):
        y = predictor(X[n], X, lambd)
        if y < epsi: y = epsi
        if y > 1-epsi: y = 1-epsi
        En = En + Y[n] * np.log(y) + (1 + Y[n]) * np.log(1 - y)
    En = -En / N
    return En










# -----------------  Main Code ---------------------

# read the data file
#N,n_row,n_col,data=read_asc_data('./dataset/AND.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/XOR.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle60.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle600.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/line600.txt')
N,n_row,n_col,data=read_asc_data('./dataset/square_circle.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/line1500.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/my_digit.txt');np.place(data[:,-1], data[:,-1]!=1, [-1])
print('find %d images of %d X %d pixels' % (N,n_row,n_col))

#plot_data(10,6,n_row,n_col,data)

#Porque esta definição?
#o N ja é um int
# 0.8 -> percentagem de valores de teste
Nt = int(N*0.8)
#numero total de elementos
I = n_row * n_col
#definir os vetores de dados de teste
Xt = data[:Nt, :-1]
Yt = data[:Nt, -1]
#vetor preenchido pelo valor 1
lambd = np.ones([Nt])
err = []
print(lambd)
#in sample error
err.append(cost(Xt,Yt,Nt,lambd))

#validation data
Nv = N-Nt
Xv = data[Nt + 1 : N, :-1]
Yv = data[Nt + 1 : N, -1]
#out sample error
# avaliaçao da capacidade de generalização a outros dados que nao os de treino
print(cost(Xv,Yv,Nv,lambd))
