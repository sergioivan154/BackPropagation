# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:06:55 2016

@author: sergio
"""
from numpy import vstack,random, exp,transpose, dot

class Neurona(object):
    
    def __init__(self, dimensionPatron, lNeuronasConecadas):
        self.W = random.rand(dimensionPatron) # generacion de pesos aleatorios  
        self.net = 0.0
        self.y = 0.0
        self.umbral = 0.1
        self.lNeuronasConecadas = lNeuronasConecadas


        
    def getPeso(self):
        return self.W
        
    def getSinapsis(self, entradas):# net entradas netas para las neuronas
        w = self.W
        x = entradas
        
        for i in range(len(w)):
            self.net += w[i]*x[i]
        return self.net

    def getSalida(self): 
        self.y = 1 / (1 + exp(-self.net))
        return self.y
        
    def getValue(self, entradas):
        self.patron = entradas
        self.getSinapsis(entradas)
        return self.getSalida()        
        
    def setValorDeseado(self, valorDeseado):
        self.d = valorDeseado

    def getValorDeseado(self):
        return self.d

    def getNeuronasConectadas(self):
        return self.lNeuronasConecadas
    
    def getPatron(self):
        return self.patron
    
    def ajustarPeso(self, peso, indice):
        self.W[indice] -= peso 
    
    def getNet(self):
        return self.net
            
        
        
        
        