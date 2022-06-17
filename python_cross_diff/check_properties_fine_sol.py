#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 19:50:10 2022

@author: carnot-smiles
"""

import numpy as np
import matplotlib.pyplot as plt
import Colors
import os

## Check properties of fine solution



def check_positivity(snapshots_sol, number_species, number_cells, index_time_step,
                     number_parameter_mu, my_path, show_plot):

    ## Check that solution is always positive
        
    ## 1) take inf on all x of snapshots_sol
    min_snap_x = np.min(snapshots_sol, 0);
    ## 2) reshape to have several rows: each row contain a solution_mu
    reshape_snapshots = np.reshape(min_snap_x, (number_parameter_mu, len(index_time_step) - 1));    
    min_x_min_mu_sol = np.min(reshape_snapshots, 0);
   

    return min_x_min_mu_sol


## Check that solution is always below than 1

def check_below_one(snapshots_sol, number_species, number_cells, index_time_step,
                     number_parameter_mu, my_path, show_plot):
    
    ## 1) take sup on all x of snapshots_sol
    max_snap_x = np.max(snapshots_sol, 0);
    ## 2) reshape to have several rows: each row contain a solution_mu
    reshape_snapshots = np.reshape(max_snap_x, (number_parameter_mu, len(index_time_step) - 1));    
    max_x_max_mu_sol = np.max(reshape_snapshots, 0);
    
    
    
    return max_x_max_mu_sol


def check_sum_species_equal_one(number_cells, number_species, number_parameter_mu,
                                snapshots_sol, index_time_step, my_path, show_plot):
    
    sum_extract_specie = np.zeros((number_cells, number_parameter_mu * (len(index_time_step) - 1)));
    for ii in range(0, number_species):
        sum_extract_specie = sum_extract_specie + snapshots_sol[ii * number_cells : (ii + 1) * number_cells, :];
       
    min_sum_snap_x = np.min(sum_extract_specie, 0);
    reshape_sum_snap = np.reshape(min_sum_snap_x, (number_parameter_mu, len(index_time_step) - 1));    
    min_x_min_mu_reshape_sum_snap = np.min(reshape_sum_snap, 0);    
    
    
    
    return min_x_min_mu_reshape_sum_snap