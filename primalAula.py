#!python
'''
logistic_classifier
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# ============ FILE load and write stuff ===========================
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def read_asc_data(filename):    
    f= open(filename,'r') 
    tmp_str=f.readline()
    tmp_arr=tmp_str[:-1].split(' ')
    N=int(tmp_arr[0]);n_row=int(tmp_arr[1]);n_col=int(tmp_arr[2])
    print("N=%d, row=%d, col=%d" %(N,n_row,n_col))
    data=np.zeros([N,n_row*n_col+1])
    for n in range(N):
        tmp_str=f.readline()
        tmp_arr=tmp_str[:-1].split(' ')       
        for i in range(n_row*n_col+1):
            data[n][i]=int(tmp_arr[i])
    f.close() 
    return N,n_row,n_col,data

def plot_data(row,col,n_row,n_col,data):
    fig=plt.figure(figsize=(row,col))
    for n in range(1, row*col +1):
        img=np.reshape(data[n-1][:-1],(n_row,n_col))
        fig.add_subplot(row, col, n)
        plt.imshow(img,interpolation='none',cmap='binary')
    plt.show()
    
def plot_tagged_data(row,col,n_row,n_col,X,Y,ew): 
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        if(predictor(X[n],ew)>0.5):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')               
    plt.show()
    
def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    #plt.ylim([0,5])
    plt.show()
    return 

def confusion(Xeval,Yeval,N,ew):
    C=np.zeros([2,2]);
    for n in range(N):
        y=predictor(Xeval[n],ew);
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1;
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1;
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1;
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1;  
    return C

def sigmoid(s):  
    large=30
    if s<-large: s=-large
    if s>large: s=large
    return (1 / (1 + np.exp(-s)))

     
        
#============== Logistic classifier Stuff ==================

def predictor(x,ew):
    s=ew[0];
    s=s+np.dot(x,ew[1:])
    sigma=sigmoid(s)
    return sigma

def cost(X,Y,N,ew):
    En=0;epsi=1.e-12
    for n in range(N):
        y=predictor(X[n],ew);
        if y<epsi: y=epsi;
        if y>1-epsi:y=1-epsi;
        En=En+Y[n]*np.log(y)+(1-Y[n])*np.log(1-y)
    En=-En/N
    return En

def update(x,y,eta,ew):
    r=predictor(x,ew);
    s=(y-r);
    #r=2*(r-0.5);
    #s=s*eta/(1+3.7*r*r)
    s = eta * s
    ew[0]=ew[0]+s
    ew[1:]=ew[1:]+s*x
    return ew

def run_epoch(X,Y,N,eta,MAX_EPOCH,ew,err):
    epsi=0;
    nb_epoch=0;
    while(err[-1]>epsi):
        nb_epoch=nb_epoch+1
        if(nb_epoch>MAX_EPOCH): break
        for n in range(N):
            ew=update(X[n],Y[n],eta,ew)  
            err.append(cost(X,Y,N,ew))
    return ew, err

def run_stocastic(X,Y,N,eta,MAX_ITER,ew,err):
    epsi=0;
    it=0
    while(err[-1]>epsi):
        n=int(np.random.rand()*N)
        #new_eta=eta*math.exp(-it/850) 
        new_eta=eta
        ew=update(X[n],Y[n],new_eta,ew)  
        err.append(cost(X,Y,N,ew))
        print('iter %d, cost=%f, eta=%e     \r' %(it,err[-1],new_eta),end='')
        it=it+1    
        if(it>MAX_ITER): break
    return ew, err


#=========== MAIN CODE ===============
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

Nt=int(N*0.8);
I=n_row*n_col;
Xt=data[:Nt,:-1];Yt=data[:Nt,-1]#;Yt = (Yt + 1) / 2
#script para substituir os valores negativos por zero
for n in range(len(Yt)):
    if Yt[n] < 0:
        Yt[n] = 0
print(Yt)
ew=np.ones([I+1])

err=[];err.append(cost(Xt,Yt,Nt,ew));


#0.1 -> training rate
#500 -> nr de iteraçoes
ew,err=run_stocastic(Xt,Yt,Nt,1,2500,ew,err);print("\n")
ew,err=run_stocastic(Xt,Yt,Nt,0.5,2000,ew,err);print("\n")
ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
ew,err=run_stocastic(Xt,Yt,Nt,0.05,1000,ew,err);print("\n")
ew,err=run_stocastic(Xt,Yt,Nt,0.01,1000,ew,err);print("\n")

plot_error(err)


print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))



C =confusion(Xt,Yt,Nt,ew)
TP = C[0,0]
TN = C[1,1]
acc = ((TP+TN)/Nt)*100
print ("Training accuracy: "+ str(acc))
print(C)
#print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))

Ne=N-Nt
Xe=data[Nt:N,:-1]
Ye=data[Nt:N,-1]
print(Ne)
print(Xe)
print(Ye)
print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
C =confusion(Xe,Ye,Ne,ew)
print(C)
TP = C[0,0]
TN = C[1,1]
FP = C[0,1]
FN = C[1,0]
#TP,TN,FP,FN = confusion(Xe,Ye,Ne,ew)
print('True positive=%i, True Negative=%i, False positive=%i, False negative=%i, ' % (TP,TN,FP,FN))
plot_tagged_data(10,6,n_row,n_col,Xe,Ye,ew)

print('bye')










