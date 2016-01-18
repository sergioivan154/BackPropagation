# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:16:25 2016

@author: sergio
"""

from numpy import array,random, exp,transpose, dot
from Neurona import Neurona
class BackPropagation(object):
    

    # la estructura debe de respetar el ultimo valor como la clase a la que pertenece y el penultimo valor el bias -1 normalmente
    def __init__(self,X, neuronasPorCapa): #neuronasPorCapa debe estar representado por capas [2,1], 2 neuronas en la capa 1 y 2 neuronas en la capa 2
        self.X = X
        self.lRedNeuronal = []
        self.neuronasPorCapa = neuronasPorCapa
        #topologia de la red
        self.lNeuronasOcultas =[]      
        #capa 2
        neurona1 = Neurona(len(X)-1, [] )
        neurona2 = Neurona(len(X)-1,[] )
        self.lNeuronasOcultas.append(neurona1)
        self.lNeuronasOcultas.append(neurona2)
        
        #capa 3
        neuronaSalida = Neurona(len(X)-1, self.lNeuronasOcultas)
        
        self.lRedNeuronal.append(neurona1)
        self.lRedNeuronal.append(neurona2)
        self.lRedNeuronal.append(neuronaSalida)
                
        
    def calcForward(self): # esta funcion se manda a llamar una vez por capa
    
        lErrores = []

        for caso in range(len(self.X)): #se recorren los casos por ejemplo en el caso de xor   el patron 0,0 es la clase 0 - caso 1, el patron 0,1 es la clase 1 - caso 2, ..., el patron 1,1 es clase 0 - caso 4                         
            index = 0
            entradasALaCapa3 = []
            self.salidasCapa2 = []
            valorEsperado = self.X.item(caso, len(self.X)-1)
            #capa 2
            for i in self.neuronasPorCapa:
                for j in range(i):    
                    
                    if index < 2:
                        x = self.X[caso,0:len(self.X)-1]
                    else:
                        entradasALaCapa3.append(-1)     
                        x = entradasALaCapa3
                    neurona = self.lRedNeuronal[index]
                    neurona.setValorDeseado(valorEsperado)
                    y = neurona.getValue(x)
                    self.salidasCapa2.append(y)
                    if index < 2:
                        entradasALaCapa3.append(y)
                    else:
                        print (str(self.X[caso, 0])+"  xor  "+str(self.X[caso, 1])+" es igual a "+ str(y))
                    
                    index +=1       
            
            #                   y         d   
            lErrores.append(( y - valorEsperado)**2 / 2)
        E = sum(lErrores)
        print("error es igual a " +str(E))
        
        
        if E > 0.4: 
            #calcular Incrementos por capa
            index = len(self.lRedNeuronal)
            for i in range(len(self.lRedNeuronal)):
                neurona = self.lRedNeuronal[index-1]
                nsalida = 0
                if index == len(self.lRedNeuronal):
                    nsalida = 1
                    self.calcularIncremento(E, neurona,nsalida)
                    index -= 1
    
    def calcularIncremento(self, error, neurona, neuronaSalida):        

        
        if neuronaSalida:
            self.deltapk = (neurona.getValorDeseado() - neurona.getSalida())* neurona.getSalida()*(1-neurona.getSalida())
            
            i = 0
            for salida in self.salidasCapa2:
                neurona.ajustarPeso( (salida * self.deltapk )  *0.1, i ) 
                i = i+1
        else:
            neuronaFinal = self.lRedNeuronal[len(self.lRedNeuronal)-1]
            
            w = neuronaFinal.getPeso()
            sumatoria = 0            
            for i in range(len(w)-1):
                sumatoria +=  self.deltapk * w[i]

            self.deltaPj = (neurona.getNet() * (1-neurona.getNet())) * sumatoria
            
            i = 0
            for x in neurona.getPatron():
                neurona.ajustarPeso(self.deltaPj*x*0.1, i)
            i += 1
        
        self.calcForward()