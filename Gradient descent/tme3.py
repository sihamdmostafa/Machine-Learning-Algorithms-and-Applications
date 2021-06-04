import numpy as np
import matplotlib.pyplot as plt

from mltools import plot_data, plot_frontiere, make_grid, gen_arti




def mse(w,x,y):
    
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])


    return ((x.dot(w) - y) ** 2)

def mse_grad(w,x,y):
        
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])

    return 2 * x * (x.dot(w) - y)



def reglog(w,x,y):
    
    x, y = x.reshape(len(y), w.shape[0]), y.reshape(-1, 1)
    w = w.reshape(-1, 1)  
    
    return np.log(1+np.exp(-1*y*np.dot(x,w)))



def reglog_grad(w,x,y):
    x, y = x.reshape(y.shape[0], w.shape[0]), y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    return -1*y*x*1/(1+np.exp(y*np.dot(x,w)))


    
def descente_gradient(datax,datay,testx,testy,f_loss,f_grad,eps,iter):
    w = np.ones((datax.shape[1],1)) 
    allw = [w]
    couts = []
    couts_test=[]
    for i in range(iter):
        w = w - eps * np.mean(f_grad(w,datax,datay),0).reshape(-1, 1)
        couts.append(f_loss(w,datax,datay).mean())
        couts_test.append(f_loss(w,testx,testy).mean())
        allw.append(w)
        if(f_loss(w,datax,datay).mean()<eps):
            break
    allw = np.array(allw)
    return w, np.array(allw) , np.array(couts),np.array(couts_test)



def grad_check(f,f_grad,d=1,N=100,eps=1e-5):
    x = np.random.rand(N,1)
    y = np.random.randint(0,2,N)
    ws = np.random.rand(N,1)
    for w in ws:
        v1 = f(w,x,y)
        v2 = f(w+eps,x,y)
        grad = f_grad(w,x,y)
        r1,r2 = (v2-v1)/eps , grad
        if (np.max(np.abs(r1 - r2)) > eps):
            return False
    return True

           



if __name__=="__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    trainx,trainy =  gen_arti(nbex=700,data_type=0 ,epsilon=0.5)
    testx,testy =  gen_arti(nbex=700,data_type=0 ,epsilon=0.5)
    plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    w,allw,couts,couttest=descente_gradient(trainx,trainy,testx,testy,reglog,reglog_grad,0.001,grid.shape[0])
    plot_frontiere(trainx,lambda x : np.sign(x.dot(w)),step=100)
    plot_data(trainx,trainy)
    
    ## Visualisation de la fonction de coût en 2D
    
    fig, ax = plt.subplots()
    ax.grid(True)
    x = np.arange(0,grid.shape[0], 1)
    ax.plot(x,couts, label = "Train", color = "C1")  
    leg = ax.legend()
    plt.show()
