#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:40:51 2021

@author: carnot-smiles
"""

## In this function we compute the safe reduced solution \bar(U_mu^n)

import numpy as np
import computational_tricks

def compute_safe_sol(zbar, number_species, number_cells):
    reshape_zbar = np.zeros((number_cells, number_species));
    #reshape_exp_zbar = np.zeros((number_cells, number_species));
    Ubar_bis = np.zeros(number_cells * number_species);
    ## Compute exponential of the projection
    #exp_zbar = np.exp(zbar);
        
    for ii in range(0, number_species):
        #reshape_exp_zbar[:, ii] = exp_zbar[ii * number_cells : (ii + 1) * number_cells];
        reshape_zbar[:, ii] = zbar[ii * number_cells : (ii + 1) * number_cells];
        
    
    
    for ii in range(0, number_species):
        for jj in range(0, number_cells):        
            Ubar_bis[ii * number_cells + jj] = np.exp(zbar[ii * number_cells + jj] - computational_tricks.logsumexp(reshape_zbar[jj, :]));

    #check = Ubar-Ubar_bis;
        
    return Ubar_bis

def compute_safe_sol_bis(zbar, number_species, number_cells):
    reshape_zbar = np.zeros((number_cells, number_species));
    #reshape_exp_zbar = np.zeros((number_cells, number_species));
    Ubar_bis = np.zeros(number_cells * number_species);
    ## Compute exponential of the projection
    #exp_zbar = np.exp(zbar);
        
    for ii in range(0, number_species):
        #reshape_exp_zbar[:, ii] = exp_zbar[ii * number_cells : (ii + 1) * number_cells];
        reshape_zbar[:, ii] = zbar[ii * number_cells : (ii + 1) * number_cells];
        
    replicate_reshape_zbar = np.matlib.repmat(reshape_zbar, number_species, 1);
    
    for ii in range(0, number_species * number_cells):
        Ubar_bis[ii] = computational_tricks.logsumexp(zbar[ii], replicate_reshape_zbar[ii, :]);
    
        
    return Ubar_bis