# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:52:01 2021

@author: hp
"""

from collections import Counter
import math
import numpy as np
import pickle
from sklearn . tree import export_graphviz
from sklearn . tree import DecisionTreeClassifier as DTree
import pydotplus
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from itertools import chain
#####Exercice 1 – Entropie ####
def entropie(vect): 
    #calcule l’entropie de vecteur vect 
    hist=Counter(vect)
    entropi=0
    for i in hist:
        p=hist[i]/sum(hist.values())
        entropi+=-p*math.log(p)
    return entropi
#------------------------------------------------------------------#
def entropie_cond(list_vect):
    #calcule l’entropie conditionnelle de la liste de listes de labels
    sum1 = 0
    for pi in list_vect:
        Ppi = len(pi)/sum([len(list_vect[j]) for j in range(len(list_vect))])
        Hypi = entropie(pi) 
        sum1+=Ppi * Hypi
    return sum1
#------------------------------------------------------------------#
# data : tableau ( films , features ) , id2titles : dictionnaire id -> titre ,
# fields : id feature -> nom
[ data , id2titles , fields ]= pickle.load(open("imdb_extrait.pkl","rb"))
# la derniere colonne est le vote
datax = data [: ,:32]
datay = np . array ([1 if x [33] >6.5 else -1 for x in data ])

entropi=entropie(datay)
entropie_cond_array=np.zeros(28)

for i in range(28):
    entropie_cond_array[i]=entropie_cond([datay[(datax[:, i] == 1)],datay[(datax[:, i] == 0)]])
    
def_entropi=entropi-entropie_cond_array
print("le meilleur attribut pour la première partition est:",fields[def_entropi.argmax()])


### Quelques expériences préliminaires ###
score=[]

for i in range(3,13):
    id2genre = [ x [1] for x in sorted ( fields . items ())[: -2]]
    dt = DTree ()
    dt . max_depth = i # on fixe la taille max de l ’ arbre a 5
    dt . min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
    dt . fit ( datax , datay )
    dt . predict ( datax [:5 ,:])
    print("le score de bonne classification pour la profondeur " + str(i) + " est :", dt.score(datax,datay)) 
    score.append(dt.score(datax,datay))
    # utiliser http :// www . webgraphviz . com / par exemple ou https :// dreampuf . github . io / GraphvizOnline
    export_graphviz(dt,out_file ="tree.dot",feature_names = id2genre)
    # ou avec pydotplus
    tdot = export_graphviz (dt,feature_names=id2genre)
    pydotplus.graph_from_dot_data(tdot).write_pdf("trees" + str(dt.max_depth) + ".pdf")
x = np.arange(3, 13, 1)
fig, ax = plt.subplots()
ax.grid(True)
ax.plot(x,score, label = "", color = "C0") 

#### Sur et sous apprentissage #### 

def separation_data(datax,datay,percnt):
    #séparer les données train et test
    indice=np.arange(len(datax))
    np.random.shuffle(indice)
    data_train=datax[indice[:int(len(datax)*percnt)]]
    data_test=datax[indice[int(len(datax)*percnt):]]
    lebel_train=datay[indice[:int(len(datax)*percnt)]]
    label_test=datay[indice[int(len(datax)*percnt):]]
    return data_train,data_test,lebel_train,label_test
### partitionnement en (0.2,0.8)
xtrain,xtest,ytrain,ytest=separation_data(datax,datay,0.2)
score_test=np.zeros(10)
score_train=np.zeros(10)
for i in range(1,11):
    id2genre = [ x [1] for x in sorted ( fields . items ())[: -2]]
    dt = DTree ()
    dt . max_depth = i # on fixe la taille max de l ’ arbre a 5
    dt . min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
    dt . fit (xtrain,ytrain)
    score_train[i-1]=dt.score(xtrain,ytrain)
    score_test[i-1]=dt.score(datax,datay)
    
fig, ax = plt.subplots()
ax.grid(True)

x = np.arange(1, 11, 1)
ax.plot(x,score_train, label = "Train 20%", color = "C0")
ax.plot(x,score_test, label = "test 80%", color = "C1", linestyle = "dashed")
ax.set_xticks(x)
ax.set_xlabel("Profondeur de l'arbre")
ax.set_ylabel("Taux de bonne classification")
ax.legend()
plt.show()
### partitionnement en (0.5,0.5)

xtrain,xtest,ytrain,ytest=separation_data(datax,datay,0.5)
score_test=np.zeros(10)
score_train=np.zeros(10)
for i in range(1,11):
    id2genre = [ x [1] for x in sorted ( fields . items ())[: -2]]
    dt = DTree ()
    dt . max_depth = i # on fixe la taille max de l ’ arbre a 5
    dt . min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
    dt . fit (xtrain,ytrain)
    score_train[i-1]=dt.score(xtrain,ytrain)
    score_test[i-1]=dt.score(datax,datay)
    
fig, ax = plt.subplots()
ax.grid(True)

x = np.arange(1, 11, 1)
ax.plot(x,score_train, label = "Train 50%", color = "C0")
ax.plot(x,score_test, label = "test 50%", color = "C1", linestyle = "dashed")
ax.set_xticks(x)
ax.set_xlabel("Profondeur de l'arbre")
ax.set_ylabel("Taux de bonne classification")
ax.legend()
plt.show()
#### partitionnement en (0.8,0.2)

xtrain,xtest,ytrain,ytest=separation_data(datax,datay,0.8)
score_test=np.zeros(10)
score_train=np.zeros(10)
for i in range(1,11):
    id2genre = [ x [1] for x in sorted ( fields . items ())[: -2]]
    dt = DTree ()
    dt . max_depth = i # on fixe la taille max de l ’ arbre a 5
    dt . min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
    dt . fit (xtrain,ytrain)
    score_train[i-1]=dt.score(xtrain,ytrain)
    score_test[i-1]=dt.score(datax,datay)
    
fig, ax = plt.subplots()
ax.grid(True)

x = np.arange(1, 11, 1)
ax.plot(x,score_train, label = "Train 80%", color = "C0")
ax.plot(x,score_test, label = "test 20%", color = "C1", linestyle = "dashed")
ax.set_xticks(x)
ax.set_xlabel("Profondeur de l'arbre")
ax.set_ylabel("Taux de bonne classification")
ax.legend()
plt.show()


#### Validation croisée : sélection de modèle #####


def crossvalidation(C, DS, m=10): 
    data_desc, data_labels = DS[0], DS[1]
    indices = [i for i in range(len(data_desc))]
    train_accs = []
    test_accs = []
    m_indices = []
    
    length = len(data_desc) // m
    
    for i in range(m): # we're not treating the case where m%len(DS) != 0 (if the rest has a size = 1, it'll kill the avg)
        # random tirage
        np.random.shuffle(indices)
        
        m_indices.append([i for i in indices[:length]])

        # remove the first length indices so that we don't take the same description twice
        for j in range(length):
            indices.pop(0)
    
    for test_index in range(m):
        # getting the training indices in each iteration
        train_indices = list(chain.from_iterable([m_indices[i] for i in range(m) if i != test_index]))
        
        # append training accuracy to the training_set
        C.fit(data_desc[train_indices], data_labels[train_indices])   
        train_accs.append(C.score(data_desc[train_indices], data_labels[train_indices]))

        # append test accuracy to the test_set
        test_accs.append(C.score(data_desc[m_indices[test_index]], data_labels[m_indices[test_index]]))    
        
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    return (train_accs.mean(),test_accs.mean())

N=10
error_train=np.zeros(34)
error_test=np.zeros(34)
for i in range(1,35):
    model = DTree()
    model.max_depth = i
    res_apprentissage, res_test = crossvalidation(model, (datax, datay), N)
    error_train[i-1]=res_apprentissage
    error_test[i-1]=res_test
    
fig, ax = plt.subplots()
ax.grid(True)
x = np.arange(1, 35, 1)
ax.plot(x,error_train, label = "cvTrain N=10", color = "C1")    
ax.plot(x,error_test, label = "cvtest N=10", color = "C0") 
ax.set_xticks(x)
ax.set_xlabel("Profondeur de l'arbre")
ax.set_ylabel("Taux de bonne classification")
ax.legend()
plt.show()

N=20
error_train=np.zeros(34)
error_test=np.zeros(34)
for i in range(1,35):
    model = DTree()
    model.max_depth = i
    res_apprentissage, res_test = crossvalidation(model, (datax, datay), N)
    error_train[i-1]=res_apprentissage
    error_test[i-1]=res_test
    
fig, ax = plt.subplots()
ax.grid(True)
x = np.arange(1, 35, 1)
ax.plot(x,error_train, label = "cvTrain N=20", color = "C1")    
ax.plot(x,error_test, label = "cvtest N=20", color = "C0") 
ax.set_xticks(x)
ax.set_xlabel("Profondeur de l'arbre")
ax.set_ylabel("Taux de bonne classification")
ax.legend()
plt.show()




























            
        
     