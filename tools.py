# -*- coding: utf-8 -*-

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

def to_col(x):
    """ convert an vector to column vector if needed """
    if len(x.shape)==1:
        x=x.reshape(x.shape[0],1)
    return x
def to_line(x):
    """ convert an vector to line vector if needed """
    if len(x.shape)==1:
        x=x.reshape(1,x.shape[0])
    return x

#################################################################################
#Données
#################################################################################

def gen_arti(data_type=0,nbex=1000,centerx=1,centery=1,sigma=0.1,epsilon=0.02):
         #center : entre des gaussiennes
         #sigma : ecart type des gaussiennes
         #nbex : nombre d'exemples
         # ex_type : vrai pour gaussiennes, faux pour echiquier
         #epsilon : bruit

         if data_type==0:
             #melange de 2 gaussiennes
             xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex/2)
             xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex/2)
             data=np.vstack((xpos,xneg))
             y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))
         if data_type==1:
             #melange de 4 gaussiennes
             xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex/4)))
             xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex/4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex/4)))
             data=np.vstack((xpos,xneg))
             y=np.hstack((np.ones(nbex/2),-np.ones(nbex/2)))

         if data_type==2:
             #echiquier
             data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
             y=np.ceil(data[:,0])+np.ceil(data[:,1])
             y=2*(y % 2)-1

         # un peu de bruit
         data[:,0]+=np.random.normal(0,epsilon,nbex)
         data[:,1]+=np.random.normal(0,epsilon,nbex)
         # on mélange les données
         idx = np.random.permutation((range(y.size)))
         data=data[idx,:]
         y=y[idx]
         return data,y


#################################################################################
# Affichage
#################################################################################

def plot_data(x,labels):
        plt.scatter(x[labels<0,0],x[labels<0,1],c='red',marker='x')
        plt.scatter(x[labels>0,0],x[labels>0,1],c='green',marker='+')

def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,data=None,step=20):
    if data!=None:
        xmax=np.max(data[:,0])
        xmin=np.min(data[:,0])
        ymax=np.max(data[:,1])
        ymin=np.min(data[:,1])
    x=np.arange(xmin,xmax,(xmax-xmin)*1./step)
    y=np.arange(ymin,ymax,(ymax-ymin)*1./step)
    xx,yy=np.meshgrid(x,y)
    grid=np.c_[xx.ravel(),yy.ravel()]
    return grid,xx,yy

    #Frontiere de decision
def plot_frontiere(x,f,step=20): # script qui engendre une grille sur l'espace des exemples, calcule pour chaque point le label
                                    # et trace la frontiere
        grid,xvec,yvec=make_grid(data=x,step=step)
        res=f(grid)
        res=res.reshape(xvec.shape)
        plt.contourf(xvec,yvec,res,colors=('gray','blue'),levels=[-1,0,1])

#################################################################################
# Apprentissage
#################################################################################

class Classifier(object):
    """ Classe generique d'un classifieur
        Dispose de 3 méthodes :
            fit pour apprendre
            predict pour predire
            score pour evaluer la precision
    """

    def fit(self,x,y):
        raise NotImplementedError("fit non  implemente")
    def predict(self,x):
        raise NotImplementedError("predict non implemente")
    def score(self,x,y):
        return (self.predict(x)==y).mean()
