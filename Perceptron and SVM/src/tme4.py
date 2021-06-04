import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti

def perceptron_loss(w,x,y):
    x, y = x.reshape(len(y), -1), y.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return np.maximum(0, -y * x.dot(w))

def perceptron_gradient(w,datax,datay):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    val=np.where(perceptron_loss(w,datax,datay)>0,1,0)
    return -val*datay*datax

def hinge_loss(w,datax,datay,alpha,lamb):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return np.maximum(0, alpha - datay * datax.dot(w)).mean() + lamb * (w**2)

def hinge_loss_grad(w,datax,datay,alpha,lamb):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    val=np.where(perceptron_loss(w,datax,datay)>0,1,0)
    return -val*datay*datax +2 * lamb * w




class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_gradient,max_iter=1000,eps=0.01,proj=None):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        self.proj=proj
        
    def fit(self,datax,datay,dataxtest,dataytest,batch=None,projection=None):
        """
        batch=1 -> stochastique sinon minibatch 
        """
        self.w = np.ones((datax.shape[1],1))
        self.couts = []
        self.couts_test=[]
        allw=[]
        N=len(datay)
        grid,x,y=make_grid(step=100)
        if projection is not None : 
            datax=self.proj(datax)
            dataxtest=self.proj(dataxtest)
            self.w = np.ones((np.array(datax).shape[1],1))
        for i in range(self.max_iter): 
            list_batch = np.random.choice(N, batch, False)
            grad = self.loss_g(self.w,datax[list_batch], datay[list_batch])
            self.w = self.w - self.eps * np.mean(grad,0).reshape(-1, 1)
            self.couts.append(self.loss(self.w,datax[list_batch],datay[list_batch]).mean())
            self.couts_test.append(self.loss(self.w,dataxtest,dataytest).mean())
            if(self.loss(self.w,datax,datay).mean()<self.eps):
                break
        return self.w,np.array(allw),np.array(self.couts),np.array(self.couts_test)
    def predict(self,datax):
        grid,x,y=make_grid(step=100)
        if self.proj is not None : 
            datax=self.proj(datax)
        return np.where(datax.dot(self.w)<0,-1,1)
    def score(self,datax,datay):
        pred=self.predict(datax)
        res=np.array([pred[i][0]*datay[i] for i in range(len(datay))])
        res=np.where(res>0,1,0)
        return sum(res)/len(datay)

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy


def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")


def proj_poly(datax):
    tmp=np.copy(datax)
    tmp=proj_biais(tmp)
    for i in range(datax.shape[1]):
        for j in range(datax.shape[1]):
            tmp=np.hstack((tmp,(datax[:,i]*datax[:,j]).reshape(-1,1)))      
    return tmp


def proj_biais(datax):
    return np.hstack((np.ones(datax.shape[0]).reshape(-1,1),datax))


def proj_gauss(datax,base,sigma): 
    tmp=np.ones((datax.shape[0],1)) 
    for i in range(len(list(base))): 
        tmp=np.hstack((tmp,np.linalg.norm(datax-base[i]*np.ones((1,datax.shape[1])),axis=1).reshape(-1,1)**2/2*sigma)) 
    return np.exp(-tmp[:,1:])




if __name__ =="__main__":
    
    plt.figure()
    trainx,trainy=gen_arti(nbex=500,data_type=1,epsilon=0.5)
    testx,testy=gen_arti(nbex=500,data_type=1,epsilon=0.5)
    perceptron=Lineaire(perceptron_loss,perceptron_gradient,1000,0.01,proj_poly)
    w,allw,cout,couts_test=perceptron.fit(trainx,trainy,testx,testy,len(trainy),1)
    plot_frontiere(trainx,perceptron.predict,step=100)
    plot_data(trainx,trainy)

    
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos =6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
    perceptron=Lineaire(perceptron_loss,perceptron_gradient,1000,0.01,None)
    w,allw,cout,couts_test=perceptron.fit(datax,np.where(datay==5,-1,1),testx,np.where(testy==5,-1,1),len(datax))
    plt.figure()
    fig, ax = plt.subplots()
    x = np.arange(0,1000, 1)
    ax.grid("true")
    ax.plot(x,cout,'-b',label ="train",color = "C1")    
    ax.plot(x,couts_test,'-b',label ="test",color = "C2")    
    leg = ax.legend();
    
    