import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pandas as pd
import math 
from numpy import linalg as LA

POI_FILENAME = "poi-paris.pkl"
parismap = mpimg.imread('paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]


class Density(object):
    def fit(self,data):
        pass
    def predict(self,data):
        pass
    def score(self,data):
        return np.sum(np.log(self.predict(data)+10e-10))

class Histogramme(Density):
    def __init__(self,steps=10):
        Density.__init__(self)
        self.steps = steps
    def fit(self,x):
        #A compléter : apprend l'histogramme de la densité sur x
        self.hist,self.axes=np.histogramdd(x,np.array([self.steps,self.steps]))
    def to_bins(self,x):
        indice_x=0
        indice_y=0
        for i in range(10):
            if(self.axes[0][i]<=x[0]<=self.axes[0][i+1]):
                indice_x=i 
            if(self.axes[1][i]<=x[1]<=self.axes[1][i+1]):
                indice_y=i
        return  indice_x,indice_y      
    def predict(self,x):
        liste=[]
        for i in range(x.shape[0]):
             liste.append(self.hist[self.to_bins(x[i])]/((x.shape[0]*((xmax-xmin)/self.steps)*((ymax-ymin)/self.steps))))
        return np.array(liste)
class KernelDensity(Density):
    def __init__(self,kernel=None,sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma
        
    def fit(self,x):
        self.x = x
    def predict(self,data):
        #A compléter : retourne la densité associée à chaque point de data
        prediction = np.zeros(len(data))
        N=len(self.x)
        d=self.x.shape[1]
        for i in range(len(data)):
            prediction[i] = (1 / (N * self.sigma **d)) *(self.kernel((data[i] - self.x) / self.sigma ).sum())
        return prediction
    
class Nadaraya(Density):
    
    def __init__(self, kernel=None, sigma=0.1):
        Density.__init__(self)
        self.kernel=kernel
        self.sigma=sigma
    
    def fit(self,data_train,notes_train):
        self.data_train=data_train
        self.notes_train=notes_train
    

    def predict(self,data_test):
        res=[]
        for x in data_test:
            
            kernel_vect=self.kernel((x - data_train)/self.sigma)
            res_i=self.notes_train*kernel_vect
            for i in range(self.notes_train.shape[0]):
                res_i[i]/=np.sum(kernel_vect[i:]) + 1
            res.append(res_i.sum())
        return res      
    
    
    
    
def kernel_uniform(x):
    return np.array([1 if np.sqrt(np.power(xi, 2).sum()) <= 0.5 else 0 for xi in x])


def kernel_gaussian(x):
    x=np.array(x)
    d = x.shape[1]
    return np.array([(2 * np.pi) ** (-d / 2) * np.exp(-1 / 2 * xi.T @ xi) for xi in x])



def get_density2D(f,data,steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:,0].min(), data[:,0].max()
    ymin, ymax = data[:,1].min(), data[:,1].max()
    xlin,ylin = np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps)
    xx, yy = np.meshgrid(xlin,ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin

def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res+1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan


def load_poi(typepoi,fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])
    
    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1],v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data,note
def separation_data_with_out_label(datax,percnt):
    #séparer les données train et test sans label
    indice=np.arange(len(datax))
    np.random.shuffle(indice)
    data_train=datax[indice[:int(len(datax)*percnt)]]
    data_test=datax[indice[int(len(datax)*percnt):]]
    return data_train,data_test   


def separation_data(datax,datay,percnt):
    #séparer les données train et test
    indice=np.arange(len(datax))
    np.random.shuffle(indice)
    data_train=datax[indice[:int(len(datax)*percnt)]]
    data_test=datax[indice[int(len(datax)*percnt):]]
    lebel_train=datay[indice[:int(len(datax)*percnt)]]
    label_test=datay[indice[int(len(datax)*percnt):]]
    return data_train,data_test,lebel_train,label_test

