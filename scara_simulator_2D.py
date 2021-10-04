# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Robot_SCARA():

    def __init__(self, length_vec):

        self.length_vec = np.array(length_vec)
        self.n = len(self.length_vec)
        self.x_0 = 0
        self.y_0 = 0

        self.x = None
        self.y = None

        self.add_noise = False
        self.perc_noise = 0.01

        self.random_state = 20141719

        self.dimension_limit_inf = None
        self.dimension_limit_sup = None

    def DirectKinematics(self, theta_rel_vec):

        """Using an vector of angles relative to each of the robot's joints, calculates the position (X,Y)
        of the ends of all the links of the SCARA robot."""

        if self.add_noise: theta_rel_vec += self.perc_noise*0 # TO DO: Add noise que vaya de 0 a 1. ya tienes la escalada

        grouping_matrix = np.transpose(np.tril( np.ones((self.n,self.n)) ))
        theta_rel_vec = np.array(theta_rel_vec)

        self.theta_abs_vec  =   np.matmul(  theta_rel_vec  ,  grouping_matrix   )
        self.x_vec = self.x_0 + np.matmul(  self.length_vec*np.cos(self.theta_abs_vec)  ,  grouping_matrix   )
        self.y_vec = self.x_0 + np.matmul(  self.length_vec*np.sin(self.theta_abs_vec)  ,  grouping_matrix   )

        #https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
        self.theta_abs_vec  =   np.arctan2(self.y_vec, self.x_vec) #recalculo para que los ?ngulos resultantes sean de -pi a pi

        self.r_vec = np.sqrt(self.x_vec**2 + self.y_vec**2)

        return self.x_vec, self.y_vec, self.theta_abs_vec, self.r_vec

    def InverseKinematics_2Links(self, x_finEff_vec, y_finEff_vec):
        if len(self.length_vec) != 2:
            raise ValueError("This function only works for a robot with 2 links!")

        q1_vec = []
        q2_vec = []

        a1 = self.length_vec[0]
        a2 = self.length_vec[1]

        for x, y in zip(x_finEff_vec, y_finEff_vec):

            q2 = np.arccos( (x**2+y**2-a1**2-a2**2) / (2*a1*a2) )
            q1 = np.arctan2(x,y) - np.arctan2(  ( a1+a2*np.cos(q2) )  ,  ( a2*np.sin(q2) )  )
            #q1 = np.arctan2(  ( np.cos(q2) )  ,  ( np.sin(q2) )  )

            q1_vec.append(q1)
            q2_vec.append(q2)

        return np.array(q1_vec), np.array(q2_vec)

    def DirectKinematics_DataFrame(self, theta_rel_vec):
        x_vec, y_vec, theta_abs_vec, r_vec = self.DirectKinematics(theta_rel_vec)
        column_names = ["x", "y", "theta_abs", "r"]
        self.column_names_allLinks = [col+"_"+str(i) for col in column_names for i in range(self.n)]
        return pd.DataFrame(np.concatenate(  (x_vec, y_vec, theta_abs_vec, r_vec)  , axis=1), columns=self.column_names_allLinks)

    def set_function_exploration_region(self, dimension_limits):

        """Defines the range of the vector's scalar components that will enter to the mathematical function to generate the
        output the network is going to learn to predict. At each dimension, the value goes from `self.dimension_limit_inf`
        to `self.dimension_limit_sup`."""
 
        self.dimension_limit_inf = dimension_limits[0]
        self.dimension_limit_sup = dimension_limits[1]

        pass

    def generateDataset(self, n_samples):

        assert( (self.dimension_limit_inf is not None) and (self.dimension_limit_sup is not None) ), "You need to define the dimension limits"
        
        print("At each dimension, the values belong to the range [",self.dimension_limit_inf,",",self.dimension_limit_sup,"]")

        aleat_from0to1 = np.random.random_sample((n_samples, self.n))

        self.x = (aleat_from0to1*(self.dimension_limit_sup-self.dimension_limit_inf))+self.dimension_limit_inf
        self.x = pd.DataFrame(self.x, columns = ["theta_rel_"+str(i) for i in range(self.n)])
        self.n_samples = n_samples

        self.y = self.DirectKinematics_DataFrame(self.x.values)
        
        # el dataframe generado contiene las columnas correspondientes a las entradas de la cinemática directa (`self.x`) y las salidas (`self.y`)
        self.df_full = pd.concat([self.x, self.y], axis=1)

        print("Dataset generated!")

        #return self.x, self.y

    def train_test_split_dataset(self, test_size):

        assert ((self.x_train is None) and (self.x_test is None) and (self.y_train is None) and (self.y_test is None)), "You first need to use the function `generateDataset`."

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( self.x, self.y, test_size=test_size,
                                                                                 random_state=self.random_state )

    def grafica_fromFinalEffector(self, x_n_desired, y_n_desired, theta_rel_fromIK, showDesiredPoint=True, showLinks=True,
                                    activateGrid=True):

        # calculamos la cinem?tica directa usando los ?ngulos theta relativos
        # descartamos el vector de radios y el vector de ?ngulos absolutos
        x, y, _, _ = self.DirectKinematics(theta_rel_fromIK)

        #https://numpy.org/doc/stable/reference/generated/numpy.insert.html
        #primero va el indice, y luego lo que quieres agregar. si es escalar, lo repite varias veces
        x = np.insert(x, 0,0, axis=1)
        y = np.insert(y, 0,0, axis=1)

        fig = plt.figure()

        for x_i, y_i in zip(x,y): # por cada pareja de i-?simos elementos de los vectores
            s = plt.scatter(x_i,y_i)
            if showLinks: s = plt.plot(x_i,y_i)

        if showDesiredPoint:
            for x_i, y_i in zip(x_n_desired, y_n_desired): # por cada pareja de i-?simos elementos de los vectores
                s = plt.scatter(x_i,y_i, marker='*')

        plt.scatter(0,0, s=123, c='k') # mostramos el origen de coordenadas 2D

        plt.axis('equal')
        plt.grid(activateGrid)

    def saveDataset(self):

        assert( self.x is not None or self.y is not None ), "You first need to generate the dataset using the method `generateDataset(...)`."

        # creamos un string con las longitudes de los eslabones del robot SCARA
        link_sizes_str = "_".join([str(link_lenght) for link_lenght in self.length_vec])
        dataset_filename="scara_robot_"+link_sizes_str+".csv"
        self.df_full.to_csv(dataset_filename, header=True)
        print("Dataset saved in file:", dataset_filename)
