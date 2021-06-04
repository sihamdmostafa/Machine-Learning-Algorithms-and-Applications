import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.tree as tree
import sklearn.svm as svm
import numpy as np
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from numpy import linalg as LA
from sklearn.linear_model import Perceptron
from sklearn import multiclass, model_selection
import warnings
import pandas as pd
import re
import random
import sklearn
from sklearn import svm,multiclass



def perceptron_loss(w,x,y):
    x, y = x.reshape(len(y), -1), y.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return np.maximum(0, 0.7-y * x.dot(w))

def perceptron_gradient(w,datax,datay):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    val=np.where(perceptron_loss(w,datax,datay)>0,1,0)
    return -val*datay*datax

def tikhonov(w,datax, datay, alpha=1, lamb=1):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return np.maximum(0, alpha - datay * datax.dot(w)).mean() + lamb * (w**2).sum()

def tikhonov_g(datax, datay, w, alpha=1, lamb=1):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    val=np.where(perceptron_loss(w,datax,datay)>0,1,0)
    return -val*datay*datax +2 * lamb * w

class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_gradient,max_iter=1000,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
       
    def fit(self,datax,datay,dataxtest=None,dataytest=None,lamda=None,batch=None):
        self.w = np.ones((datax.shape[1],1))
        self.couts = []
        self.couts_test=[]
        allw=[]
        N=len(datay)
        for i in range(self.max_iter): 
            if batch is not None:
                list_batch = np.random.choice(N, batch, False)
                grad = self.loss_g(self.w,datax[list_batch], datay[list_batch])
            else:
                grad = self.loss_g(self.w,datax, datay)
            self.w = self.w - self.eps * np.mean(grad,0).reshape(-1, 1)
            self.couts.append(self.loss(self.w,datax,datay).mean())
            #self.couts_test.append(self.loss(self.w,dataxtest,dataytest).mean())
            if(self.loss(self.w,datax,datay).mean()<self.eps):
                break
        return self.w,np.array(allw),np.array(self.couts),np.array(self.couts_test)

    def predict(self,datax):
        return np.where(datax.dot(self.w)<0,-1,1)
    def score1(self,datax,datay):
        pred=self.predict(datax)
        res=np.array([pred[i][0]*datay[i] for i in range(len(datay))])
        res=np.where(res>0,1,0)
        return sum(res)/len(datay)

    def score(self,datax,datay):
        pred=self.predict(datax)
        res=np.maximum(np.array(pred*datay),np.zeros(len(datay)))
        return (1-(sum(res)/len(datay)))[0]

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

def plot_frontiere_proba(data,f,step=20):
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),255)    

def SVM( C=10, degree=3,gamma='scale', kernel='linear',  max_iter=100,data_USPS=False,data_type=0, epsilon=0.3,probability=True):  # idk si j'ai pris tt les parametres possible 
    trainx, trainy = gen_arti(nbex=1000, data_type=data_type, epsilon=epsilon)
    testx, testy = gen_arti(nbex=1000, data_type=data_type, epsilon=epsilon)
    s = svm.SVC(C=C, kernel=kernel,degree=degree,gamma=gamma, probability=probability, max_iter=max_iter)

    s.fit(trainx, trainy)

    err_train = 1 - s.score(trainx, trainy)
    err_test = 1 - s.score(testx, testy)

    print("Erreur : train %f, test %f\n" % (err_train, err_test))
    if (not data_USPS):
        if probability:
            def f(x): return s.predict_proba(x)[:, 0]
        else:
            def f(x): return s.decision_function(x)

        plot_frontiere_proba(testx,lambda x : s.predict_proba( x )[ : , 0 ],step =50)
        plot_data(testx,testy)
        plt.title("using the kernel: "+kernel)
    return s






def svm_gridSearch(trainx, trainy,kernel):
    grid = {'C': [1, 5, 10, 15, 20, 50, 100],
            'max_iter': [4000,8000,10000],
            'kernel': [kernel],
            'gamma': [0.0001, 0.001, 0.01, 0.1],
            'degree':[1,3,5,7],
            'shrinking':[True,False]
           }

    clf = svm.SVC()
    clf = model_selection.GridSearchCV(clf, grid,cv=5)
    clf.fit(trainx, trainy)   
    return clf.best_params_


class OvsO(object):
    "One-versus-one implémenté"
    def __init__(self, classifier, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.classifiers = dict()
        for i in range(self.nb_classes):
            for j in range(i + 1, self.nb_classes):
                self.classifiers[(i, j)] = classifier(**kwargs)
                
    def fit(self, datax, datay):
        self.classes = np.unique(datay)
        assert self.classes.size == self.nb_classes
        for i in range(self.nb_classes):
            for j in range(i + 1, self.nb_classes):
                valplus = i
                valminus = j
                train_x = datax[np.logical_or(datay == valplus, datay == valminus), :]    
                train_y = datay[np.logical_or(datay == valplus, datay == valminus)]
                train_y = np.where(train_y == valplus, 1, -1)
                self.classifiers[(i, j)].fit(train_x, train_y,0)
                
    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1,-1)
        score = np.zeros((datax.shape[0], self.nb_classes), dtype = int)
        for i in range(self.nb_classes):
            for j in range(i + 1, self.nb_classes):
                x = (self.classifiers[(i, j)].predict(datax) >= 0)
                score[:, i] += x[:, 0]
                score[:, j] += np.logical_not(x)[:, 0]
        return np.array([self.classes[k] for k in np.argmax(score, 1)])   
    
    def accuracy(self, datax, datay):
        datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
        y_pred = self.predict(datax).reshape(-1, 1)
        return (y_pred == datay).mean()
    
    
class oVsA(object):
    "One-versus-all implémenté"
    def __init__(self, classifier, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.classifiers = np.empty(self.nb_classes, dtype = object)
        for i in range(self.nb_classes):
            self.classifiers[i] = classifier(**kwargs)
                
    def fit(self, datax, datay):
        self.classes = np.unique(datay)
        assert self.classes.size == self.nb_classes
        for i in range(self.nb_classes):
            valplus = i
            train_x = datax 
            train_y = np.where(datay == valplus, -1, 1)
            self.classifiers[i].fit(train_x, train_y,0)
                
    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1,-1)
        score = np.zeros((datax.shape[0], self.nb_classes))
        for i in range(self.nb_classes):
            x = self.classifiers[i].predict(datax)
            score[:, i] = x[:, 0]
        return np.array([self.classes[k] for k in np.argmin(score, 1)])   
    
    def accuracy(self, datax, datay):
        datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
        y_pred = self.predict(datax).reshape(-1, 1)
        return (y_pred == datay).mean()





def sub_string(x, u):
    #retourne tout les sous sequences u dans x
    occ = []   
    list1 = list(x)
    list2 = list(u)        
    for i in range(len(list1)-len(list2)):
        subsequence = list1[i:i+len(list2)]
        if subsequence == list2:
            occ.append(list(range(i,i+len(list2))))  
    return occ


def get_all_subsuq(data,n):
    #retourne tous les sous chaine de taille n dans data
    result = set()
    for text in data:
        text_l = list(text)
        for i in range(len(text_l)-n):
            result.add(''.join(text_l[i:i+n]))
    return list(result)




def stringKernel(first_w,seconde_w,n,alpha = 0.5):
    """returns the dot product resulting from the string kernel"""
    all_word = get_all_subsuq([first_w],n)
    res = 0 #contient la formule du string kernel 
    for word in all_word:# for each subsequence in all_word
        s_seq = sub_string(first_w, word)
        t_seq = sub_string(seconde_w, word)
        # si cette sous-séquence apparaît dans t, sinon le terme est égal à 0 dans le produit scalaire
        for j in t_seq:
            for i in s_seq:
                res = res + pow(alpha, (j[len(j)-1]-j[0]+1)+(i[len(i)-1]-i[0]+1))
    return res












def get_similarity_matrix(train_x,n,alpha=0.1):
    #retorune la matrice de similarité du string kernel
    result = np.zeros((train_x.shape[0],train_x.shape[0]))
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[0]):
            result[i,j] = stringKernel(train_x[i],train_x[j],n,alpha)
    return result


warnings.filterwarnings("ignore")

if __name__ =="__main__":
    
    
    
    trainx,trainy=gen_arti(nbex=500,data_type=0,epsilon=0.5)
    testx,testy=gen_arti(nbex=500,data_type=0,epsilon=0.5)
    
    trainx1,trainy1=gen_arti(nbex=500,data_type=1,epsilon=0.5)
    testx1,testy1=gen_arti(nbex=500,data_type=1,epsilon=0.5)
    
    clf = Perceptron(random_state=0)
    perceptron=Lineaire()
    perceptron.fit(trainx,trainy)
    clf.fit(trainx,trainy)
    print("Perceptron implémenté,2 gaussiane, l'erreur : train %f, test %f"% (1-perceptron.score1(trainx,trainy),1-perceptron.score1(testx,testy)))
    print("le Perceptron de scikit-learn,2 gaussiane, l'erreur: train %f, test %f"% (1-clf.score(trainx,trainy),1-clf.score(testx,testy)))
    
    plt.figure()
    plt.subplot(3, 2, 1)
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plt.title('Implemented Perceptron')
    
    plt.subplot(3, 2, 2)
    plot_frontiere(trainx,clf.predict,200)
    plot_data(trainx,trainy)
    plt.title('Perceptron sktlearn')
    
    clf = Perceptron(random_state=0)
    perceptron = Lineaire()
    perceptron.fit(trainx1,trainy1)
    clf.fit(trainx1,trainy1)
    print("Perceptron implémenté,4 gaussiane, l'erreur : train %f, test %f"% (1-perceptron.score1(trainx1,trainy1),1-perceptron.score1(testx1,testy1)))
    print("Perceptron de scikit-learn,4 gaussiane,l'erreur: train %f, test %f"% (1-clf.score(trainx1,trainy1),1-clf.score(testx1,testy1)))
   
    plt.subplot(3, 2, 3)
    plot_frontiere(trainx1,perceptron.predict,200)
    plot_data(trainx1,trainy1)
    plt.subplot(3, 2, 4)
    plot_frontiere(trainx1,clf.predict,200)
    plot_data(trainx1,trainy1)
    
    """    
    plt.figure(1,figsize=(10,10))
    plt.subplot(2,2,1)
    m1 = SVM( C=10, kernel='linear',  max_iter=100,data_USPS=False,data_type=0, epsilon=0,probability=True)
    plt.subplot(2,2,2)
    print('Sur données artificielles ,separables linéairement,  bruité') 
    m1 = SVM( C=10, kernel='linear',  max_iter=100,data_USPS=False,data_type=0, epsilon=0.9,probability=True)
    plt.show()    
    m2 = SVM( C=5, kernel='linear',  max_iter=100,data_USPS=False,data_type=1, epsilon=0,probability=True)
    plt.show()
    
    
    #le kernel polynomial
    plt.figure(1,figsize=(10,10))
    i = 1
    for d in [1, 2, 3, 4]:
        plt.subplot(2,2,i)
        i +=1
        m = SVM( C=10, kernel='poly',degree=d,  max_iter=100,data_USPS=False,data_type=1, epsilon=0,probability=True)
        plt.title("d = "+str(d)+"\n"+"nbre vecteurs supports = "+str(len(m.support_vectors_)))
    plt.show()
     
    
    m = SVM( C=10, kernel='rbf',gamma=10,  max_iter=100,data_USPS=False,data_type=2, epsilon=0,probability=True)
    plt.show()
    plt.figure(1,figsize=(15,15))
    i = 1
    for (gamma, c) in [(0.1,0.01), (0.1,1), (0.1,10), (1,0.01), (1,1), (1,10), (10,0.01), (10,1), (10,10)]:
        plt.subplot(3,3,i)
        i +=1
        m = SVM( C=c, kernel='rbf',gamma=gamma,  max_iter=100,data_USPS=False,data_type=0, epsilon=1,probability=True)
        plt.title("gamma = "+str(gamma)+" C = "+str(c)+"\n"+"nbre vecteurs supports = "+str(len(m.support_vectors_)))
    plt.show()
        
    
    trainx0,trainy0 =  gen_arti(nbex=2000,data_type=0 ,epsilon=0.5)
    trainx1,trainy1 =  gen_arti(nbex=2000,data_type=1 ,epsilon=0.5)
    trainx2,trainy2 =  gen_arti(nbex=2000,data_type=2 ,epsilon=0.5)
    testx0,testy0 =  gen_arti(nbex=700,data_type=0 ,epsilon=0.5)
    testx1,testy1 =  gen_arti(nbex=700,data_type=1 ,epsilon=0.5)
    testx2,testy2 =  gen_arti(nbex=700,data_type=2 ,epsilon=0.5)
    
    result=""
    for kernel in ['linear', 'rbf', 'poly']:
        result = result+'for the kernel '+kernel+"\n"
        print("Niiiiiiiiiik Mooooooook")
        result = result+"   best parameter using data type 0 are: "+str(svm_gridSearch(trainx0, np.ravel(trainy0,order='C'),kernel))+"\n"
        result = result+"   best parameter using data type 1 are: "+str(svm_gridSearch(trainx1, np.ravel(trainy1,order='C'),kernel))+"\n"
        result = result+"   best parameter using data type 2 are: "+str(svm_gridSearch(trainx2, np.ravel(trainy2,order='C'),kernel))+"\n"
    print("------------ optimization finished , the best parameters are : \n"+result)
   
    
    trainx0,trainy0 =  gen_arti(nbex=2000,data_type=0 ,epsilon=0.5)
    trainx1,trainy1 =  gen_arti(nbex=2000,data_type=1 ,epsilon=0.5)
    trainx2,trainy2 =  gen_arti(nbex=2000,data_type=2 ,epsilon=0.5)
    testx0,testy0 =  gen_arti(nbex=700,data_type=0 ,epsilon=0.5)
    testx1,testy1 =  gen_arti(nbex=700,data_type=1 ,epsilon=0.5)
    testx2,testy2 =  gen_arti(nbex=700,data_type=2 ,epsilon=0.5)
    nb_obs=[]
    
    err_train0_linear=[]
    err_test0_linear=[]
    err_train1_linear=[]
    err_test1_linear=[]
    err_train2_linear=[]
    err_test2_linear=[]
    
    
    for i in range (200,2000,100):
        nb_obs.append(i)
        s = svm.SVC(C=1, degree= 1, gamma= 0.0001, kernel='linear', max_iter= 4000, shrinking= True)
        s.fit(trainx0[:i], trainy0[:i])
        err_train0_linear.append( 1 - s.score(trainx0[:i], trainy0[:i]))
        err_test0_linear.append( 1 - s.score(testx0, testy0))
        s = svm.SVC(C=20, degree= 1, gamma= 0.0001, kernel='linear', max_iter= 10000, shrinking= True)     
        s.fit(trainx1[:i], trainy1[:i])
        err_train1_linear.append( 1 - s.score(trainx1[:i], trainy1[:i]))
        err_test1_linear.append( 1 - s.score(testx1, testy1))
        s = svm.SVC(C=15, degree= 1, gamma= 0.0001, kernel='linear', max_iter= 8000, shrinking= True)
        s.fit(trainx2[:i], trainy2[:i])
        err_train2_linear.append( 1 - s.score(trainx2[:i], trainy2[:i]))
        err_test2_linear.append( 1 - s.score(testx2, testy2))
    plt.subplot(1, 2, 1)
    plt.plot(nb_obs,err_train0_linear,label="type 0")
    plt.plot(nb_obs,err_train1_linear,label="type 1")
    plt.plot(nb_obs,err_train2_linear,label="type 2")
    plt.ylabel("l'erreur")
    plt.xlabel("nombre d'exemples pour l'apprentissage")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(nb_obs,err_test0_linear,label="type 0")
    plt.plot(nb_obs,err_test1_linear,label="type 1")
    plt.plot(nb_obs,err_test2_linear,label="type 2")

    plt.xlabel("nombre d'exemples pour l'apprentissage")
    plt.legend()
    plt.show()
  
    err_train0_linear=[]
    err_test0_linear=[]
    err_train1_linear=[]
    err_test1_linear=[]
    err_train2_linear=[]
    err_test2_linear=[]

    
    for i in range (200,2000,100):
        nb_obs.append(i)
        s = svm.SVC(C=1, degree= 1, gamma= 0.0001, kernel='poly', max_iter= 4000, shrinking= True)
        s.fit(trainx0[:i], trainy0[:i])
        err_train0_linear.append( 1 - s.score(trainx0[:i], trainy0[:i]))
        err_test0_linear.append( 1 - s.score(testx0, testy0))
        s = svm.SVC(C=10, degree= 2, gamma= 0.1, kernel='poly', max_iter=4000, shrinking= True)     
        s.fit(trainx1[:i], trainy1[:i])
        err_train1_linear.append( 1 - s.score(trainx1[:i], trainy1[:i]))
        err_test1_linear.append( 1 - s.score(testx1, testy1))
        s = svm.SVC(C=20, degree= 4, gamma= 0.1, kernel='poly', max_iter= 10000, shrinking= True)
        s.fit(trainx2[:i], trainy2[:i])
        err_train2_linear.append( 1 - s.score(trainx2[:i], trainy2[:i]))
        err_test2_linear.append( 1 - s.score(testx2, testy2))
    plt.subplot(1, 2, 1)
    plt.plot(nb_obs,err_train0_linear,label="type 0")
    plt.plot(nb_obs,err_train1_linear,label="type 1")
    plt.plot(nb_obs,err_train2_linear,label="type 2")
    plt.ylabel("l'erreur")
    plt.xlabel("nombre d'exemples pour l'apprentissage")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(nb_obs,err_test0_linear,label="type 0")
    plt.plot(nb_obs,err_test1_linear,label="type 1")
    plt.plot(nb_obs,err_test2_linear,label="type 2")

   
    plt.xlabel("nombre d'exemples pour l'apprentissage")
    plt.legend()
    plt.show()
    
    
    for i in range (200,2000,100):
        nb_obs.append(i)
        s = svm.SVC(C=1, degree= 1, gamma= 0.0001, kernel='rbf', max_iter= 4000, shrinking= True)
        s.fit(trainx0[:i], trainy0[:i])
        err_train0_linear.append( 1 - s.score(trainx0[:i], trainy0[:i]))
        err_test0_linear.append( 1 - s.score(testx0, testy0))
        s = svm.SVC(C=1, degree= 1, gamma= 0.1, kernel='rbf', max_iter=4000, shrinking= True)     
        s.fit(trainx1[:i], trainy1[:i])
        err_train1_linear.append( 1 - s.score(trainx1[:i], trainy1[:i]))
        err_test1_linear.append( 1 - s.score(testx1, testy1))
        s = svm.SVC(C=20, degree= 5,gamma=1, kernel='rbf', max_iter= 10000, shrinking= True)
        s.fit(trainx2[:i], trainy2[:i])
        err_train2_linear.append( 1 - s.score(trainx2[:i], trainy2[:i]))
        err_test2_linear.append( 1 - s.score(testx2, testy2))
    
    plt.subplot(1, 2, 1)
    plt.plot(nb_obs,err_train0_linear,label="type 0")
    plt.plot(nb_obs,err_train1_linear,label="type 1")
    plt.plot(nb_obs,err_train2_linear,label="type 2")
    plt.ylabel("l'erreur")
    plt.xlabel("nombre d'exemples pour l'apprentissage")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(nb_obs,err_test0_linear,label="type 0")
    plt.plot(nb_obs,err_test1_linear,label="type 1")
    plt.plot(nb_obs,err_test2_linear,label="type 2")
    plt.xlabel("nombre d'exemples pour l'apprentissage")
    plt.legend()
    plt.show()
   

     
    
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    train_datax, train_datay = load_usps(uspsdatatrain)
    test_datax, test_datay = load_usps(uspsdatatest)
    
    OvsO_model = OvsO(Lineaire, 10, eps=0.1)
    OvsO_model.fit(train_datax, train_datay)
    print("Implementé,Accuracy : train %f, test %f"% (OvsO_model.accuracy(train_datax, train_datay),OvsO_model.accuracy(test_datax, test_datay)))
    
    
    oVsA_model = oVsA(Lineaire, 10, eps=0.1)
    oVsA_model.fit(train_datax, train_datay)
    print("Implementé,Accuracy : train %f, test0.885401 %f"% (oVsA_model.accuracy(train_datax, train_datay),oVsA_model.accuracy(test_datax, test_datay)))
    
    oneVsOne = multiclass.OneVsOneClassifier(svm.LinearSVC(max_iter=16000))
    oneVsAll = multiclass.OneVsRestClassifier(svm.LinearSVC(max_iter=16000))

    oneVsOne.fit(train_datax, train_datay)
    oneVsOneTrainErr = oneVsOne.score(train_datax, train_datay)
    oneVsOneTestErr = oneVsOne.score(test_datax, test_datay)

    oneVsAll.fit(train_datax, train_datay)
    oneVsAllTrainErr= oneVsAll.score(train_datax, train_datay)
    oneVsAllTestErr = oneVsAll.score(test_datax, test_datay)

    print("Skitlearn,L'erreur en train pour oneVsOne :"+str(oneVsOneTrainErr)+" et pour oneVsAll :"+str(oneVsAllTrainErr))
    print("Skitlearn,L'erreur en test de oneVsOne :"+str(oneVsOneTestErr)+" et pour oneVsAll :"+str(oneVsAllTestErr))
    
    
    # corpus contenant des quotes et leur auteur
    corpus = pd.read_csv("quotes.csv")
    # on ne garde que deux auteurs
    corpus_clean = corpus[corpus["author"].isin(["A. A. Milne","A. J. Jacobs"])]
    # classification, en -1 et 1
    corpus_clean["author"].replace({"Debasish Mridha":1,"Lailah Gifty Akita":-1}, inplace=True)
    # données netoyées
    features = corpus_clean["quote"].to_numpy()
    target = corpus_clean["author"].to_numpy()
    similarity_matrix = get_similarity_matrix(features,4,1)
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(similarity_matrix)
    ax.set_xticks(np.arange(len(target)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_xticklabels(map(str, target))
    ax.set_yticklabels(map(str, target))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    ax.set_title("Matrice de similarité | deux auteurs | string kernel")
    fig.tight_layout()
    plt.show()
    #Apprentisage
    shuffeld_ind = list(range(0,features.shape[0]))
    random.shuffle(shuffeld_ind)
    train_ind = shuffeld_ind[len(shuffeld_ind)//3:]
    test_ind = shuffeld_ind[:len(shuffeld_ind)//3]
    train_x = similarity_matrix[train_ind]
    train_y = target[train_ind]
    test_x = similarity_matrix[test_ind]
    test_y = target[test_ind]
    model = svm.SVC(C=10)
    model.fit(train_x, train_y) 
    print("L'accuracy est de :",sklearn.metrics.accuracy_score(test_y, model.predict(test_x)))
    """
    
    
    