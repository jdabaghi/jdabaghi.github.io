#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 19:54:53 2021

@author: carnot-smiles
"""

## Property for the numerical solution

## Entropy dissipation
import numpy as np

def entropy_property(sol, Dx, number_cells, number_species):
    product = np.zeros(number_cells);
    sol_reshape = np.reshape(sol, (number_cells, number_species));
    log_sol_reshape = np.reshape(np.log(sol), (number_cells, number_species));
    
    for ii in range(0, number_cells):
        product[ii] = Dx * np.dot(sol_reshape[ii, :], log_sol_reshape[ii, :])     
    
    entropy = sum(product); 
    
    
    return entropy