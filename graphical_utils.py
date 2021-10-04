# -*- coding: utf-8 -*-
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Funciones auxiliares para graficar

def grafica_fromCompleteVects(x,y, activateGrid=True, showLinks=True):

    #https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    #primero va el indice, y luego lo que quieres agregar. si es escalar, lo repite varias veces
    x = np.insert(x, 0,0, axis=1)
    y = np.insert(y, 0,0, axis=1)

    fig = plt.figure()

    for x_i, y_i in zip(x,y):
        s = plt.scatter(x_i,y_i)
        if showLinks: s = plt.plot(x_i,y_i)

    plt.scatter(0,0, s=123, c='k')
    plt.axis('equal')
    plt.grid(activateGrid)

    pass

def grafica_polar_fromCompleteVects(r,theta):

    #https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    #primero va el indice, y luego lo que quieres agregar. si es escalar, lo repite varias veces
    r = np.insert(r, 0,0, axis=1)
    theta = np.insert(theta, 0,0, axis=1)

    fig = plt.figure()

    for r_i, theta_i in zip(r,theta):
        s = plt.polar(theta_i, r_i)
        #s = plt.plot(theta_i, r_i)

    plt.scatter(0,0, s=123, c='k')

    pass
