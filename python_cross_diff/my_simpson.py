#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:39:31 2021

@author: carnot-smiles
"""
import numpy as np

### Simpson formula

def my_simpson(mesh_space, value, number_cells):
    
    int_element = np.zeros(len(value) - 1);
    
    for ii in range(0, number_cells - 1):
               
        internal_value = 0.5 * (value[ii] + value[ii + 1]);
        
        int_element[ii] = 1./6 * (mesh_space[ii + 1] - mesh_space[ii]) * (value[ii] + 4 * internal_value + value[ii + 1]);
    
    return int_element