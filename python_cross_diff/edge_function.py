#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:34:09 2021

@author: carnot-smiles
"""

## function that computes the edge function u_{j,ϭ}^n

import numpy as np
import math
import compute_safe_sol
## first def
#def edge_value(a, b):
#    if min(a, b) < 0:
#        result = 0;
#    elif a == b and a >=0:
#        result = a;
#    else:
#        result = (a - b)/(math.log(a) - math.log(b)); 
#    
#    return result

## second def
def edge_value(a, b):
    err = 1e-8;
    if min(a, b) <= 0:
        result = 0;
    elif abs(a - b) > err:
        result = (a - b)/(math.log(a) - math.log(b));
    else:
        result = (a + b)/2; 
    
    return result

def edge_solution(sol, number_cells, number_internal_edges):
    
    edge_sol = np.zeros((number_cells, 2));
    ## first element has one edge contribution
    edge_sol[0,0] = edge_value(sol[0], sol[1]);
    
    for ii in range(1, number_cells - 1):
        edge_sol[ii, 0] = edge_value(sol[ii], sol[ii - 1]);
        edge_sol[ii, 1] = edge_value(sol[ii], sol[ii + 1]);
        
    edge_sol[-1, 0] = edge_value(sol[-1], sol[-2]);    
     
    return  edge_sol

def edge_solution_reduced_safe(sol, number_cells, number_species, number_internal_edges):
    edge_sol = np.zeros((number_cells, 2));
    
    
    ## first element has one edge contribution    
    edge_sol[0,0] = edge_value(sol[0], sol[1]);
    
    for ii in range(1, number_cells - 1):
        edge_sol[ii, 0] = edge_value(sol[ii], sol[ii - 1]);
        edge_sol[ii, 1] = edge_value(sol[ii], sol[ii + 1]);
        
    edge_sol[-1, 0] = edge_value(sol[-1], sol[-2]);    
     
    return  edge_sol
    
    





