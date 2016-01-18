# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:13:15 2016

@author: sergio
"""
from numpy import array
from BackPropagation import BackPropagation
def main():
    
    #xor
    X = array([[0, 0, -1, 0],
               [0, 1, -1, 1],
               [1, 0, -1, 1],
               [1, 1, -1, 0],])
    
    
    mlp = BackPropagation(X, [2,1])
    mlp.calcForward()

main()