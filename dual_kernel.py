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


def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    #plt.ylim([0,5])
    plt.show()
    return 

def confusion(Xeval,Yeval,N,lambd):
    C=np.zeros([2,2])
    for n in range(N):
        y=predictor(Xeval[n], Xeval,lambd)
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1 
    return C


def plot_tagged_data(row,col,n_row,n_col,X,Y,lambd): 
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        if(predictor(X[n],X,lambd)>0.5):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')               
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
# length X -> 1500 // length X [1] -> 144
# length x -> 144
# length lambd -> 1500
def predictor (x, X, lambd):
    # DUVIDA -> tinhamos I + 1 mas so pode ser I senao os comprimentos nao coencidem
    #sum = np.zeros([I])
    #k define o grau do kernel
    k = 4
    #componente nr 0 // tratada a parte
    h = 0
    for n in range(len(X)):
        # print("teste de lambd")
        # print(len(lambd))
        # nao esta a dar para multiplicar o array por um int
        #aux_list = [i * lambd[n] for i in X[n]]
        # print("teste de lista aux")
        # print((aux_list))
        if lambd[n] == 0:
            continue

        dotProd = np.dot(x,X[n])
        
        h = h + lambd[n] * dotProd**k + lambd[n]
        # sum = sum + lambd[n] * X[n]
        # bias = bias + lambd[n]
    

    #h = np.dot(sum,x) + bias
    

    sigma = sigmoid (h)
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
        En = En + Y[n] * np.log(y) + (1 - Y[n]) * np.log(1 - y)
    En = -En / N
    return En


# m -> index aleatorio passado na funçao run_Stocastic
def update(m,X,y,eta,lambd):
    x = X[m,:]
    r = predictor(x, X, lambd)
    s = (y - r)
    s = eta * s
    # print ("m: "+str(m)+"          ")
    # print("S: "+str(s))
    #lambd[0] = lambd[0] + s
    lambd[m] = lambd[m] + s # * x
    return lambd



def run_stocastic(X, Y, N, eta, MAX_ITER, lambd, err):
    epsi = 0
    it = 0
    while(err[-1] > epsi):
        n = int(np.random.rand()*N)
        #print("Valor N: "+ str(n))
        new_eta = eta
        lambd = update(n, X, Y[n], new_eta, lambd)
        if it%10 == 0:
            err.append(cost(X, Y, N, lambd))
            print('\niter %d, cost=%f, eta=%e     \r' %(it,err[-1],new_eta),end='')
        it = it + 1
        if(it>MAX_ITER): break
    return lambd, err


# -----------------  Main Code ---------------------

# read the data file
#N,n_row,n_col,data=read_asc_data('./dataset/AND.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/XOR.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/rectangle60.txt')
N,n_row,n_col,data=read_asc_data('./dataset/rectangle600.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/line600.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/square_circle.txt')
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
#script para substituir os valores negativos por zero
for n in range(len(Yt)):
    if Yt[n] < 0:
        Yt[n] = 0
print("####################################")
print(Yt)
#vetor preenchido pelo valor 1
lambd = np.zeros([Nt])
err = []

print(Xt)
#in sample error
err.append(cost(Xt,Yt,Nt,lambd))


#0.1 -> training rate
#500 -> nr de iteraçoes
lambd,err=run_stocastic(Xt,Yt,Nt,0.5,1000,lambd,err);print("\n")
lambd,err=run_stocastic(Xt,Yt,Nt,0.1,200,lambd,err);print("\n")
#lambd,err=run_stocastic(Xt,Yt,Nt,0.1,200,lambd,err);print("\n")
plot_error(err)

print('in-samples error=%f ' % (cost(Xt,Yt,Nt,lambd)))
C =confusion(Xt,Yt,Nt,lambd)
TP = C[0,0]
TN = C[1,1]
acc = ((TP+TN)/Nt)*100
print ("Training accuracy: "+ str(acc))
print(C)



# #validation data
# Nv = N-Nt
# Xv = data[Nt + 1 : N, :-1]
# Yv = data[Nt + 1 : N, -1]
# #out sample error
# # avaliaçao da capacidade de generalização a outros dados que nao os de treino
# print(cost(Xv,Yv,Nv,lambd))





Ne=N-Nt
Xe=data[Nt:N,:-1]
Ye=data[Nt:N,-1]
print(Ne)
print(Xe)
print(Ye)
print('out-samples error=%f' % (cost(Xe,Ye,Ne,lambd)))
C =confusion(Xe,Ye,Ne,lambd)
print(C)
TP = C[0,0]
TN = C[1,1]
FP = C[0,1]
FN = C[1,0]
#TP,TN,FP,FN = confusion(Xe,Ye,Ne,lambd)
print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))
plot_tagged_data(10,6,n_row,n_col,Xe,Ye,lambd)

print('bye')