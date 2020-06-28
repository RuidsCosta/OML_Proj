
# funçao que calcula a previsao
# no dual tenho que mudar isto para ewT (transposta) -> predictor(ewT, x)
# lambda substitui o ew
# length X -> 1500 // length X [1] -> 144
# length x -> 144
# length lambd -> 1500
def predictor (x, X, lambd):
    # DUVIDA -> tinhamos I + 1 mas so pode ser I senao os comprimentos nao coencidem
    #sum = np.zeros([I])

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

        h = h + lambd[n] * np.dot(x,X[n]) + lambd[n]
        # sum = sum + lambd[n] * X[n]
        # bias = bias + lambd[n]
    

    #h = np.dot(sum,x) + bias
    

    sigma = sigmoid (h)
    return sigma