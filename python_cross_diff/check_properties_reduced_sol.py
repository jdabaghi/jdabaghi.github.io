#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:18:43 2021

@author: carnot-smiles
"""

## Check properties of reduced solution

import numpy as np
import matplotlib.pyplot as plt
import Colors
import os

def check_positivity(all_reduced_sol, number_species, number_cells, index_time_step,
                     number_parameter_mu, my_path2, show_plot):
    
    
    ## Check that solution is always positive for safe reduced model. For not safe reduced model it could be negative
    
    ## 1) take inf on all x of snapshots_sol
    min_snap_x = np.min(all_reduced_sol, 0);
    ## 2) reshape to have several rows: each row contain a solution_mu
    reshape_snapshots = np.reshape(min_snap_x, (number_parameter_mu, len(index_time_step)));    
    min_x_min_mu_sol = np.min(reshape_snapshots, 0);
    
    
    
    return min_x_min_mu_sol


def check_below_one(all_reduced_sol, number_species, number_cells, index_time_step,
                     number_parameter_mu, my_path2, show_plot):
    
    ## 1) take sup on all x of snapshots_sol
    max_snap_x = np.max(all_reduced_sol, 0);
    ## 2) reshape to have several rows: each row contain a solution_mu
    #reshape_snapshots = np.reshape(max_snap_x, (number_parameter_mu, len(index_time_step) - 1));
    
    
    #TRY 20 APRIL 2022
    reshape_snapshots = np.reshape(max_snap_x, (number_parameter_mu, len(index_time_step)));
    max_x_max_mu_sol = np.max(reshape_snapshots, 0);
    
    
    
    return max_x_max_mu_sol


    

def check_sum_species_equal_one(number_cells, number_species, number_parameter_mu,
                                all_reduced_sol, index_time_step, my_path2, show_plot):

    ## Check that sum species is always equal to 1
    
#    sum_extract_specie = np.zeros((number_cells, number_parameter_mu * (len(index_time_step) - 1)));
#    
#    for ii in range(0, number_species):
#        sum_extract_specie = sum_extract_specie + all_reduced_sol[ii * number_cells : (ii + 1) * number_cells, :];
#       
#    min_sum_snap_x = np.min(sum_extract_specie, 0);
#    reshape_sum_snap = np.reshape(min_sum_snap_x, (number_parameter_mu, len(index_time_step) - 1));    
#    min_x_min_mu_reshape_sum_snap = np.min(reshape_sum_snap, 0);    
    
    
     ## CHANGED 20 APRIL 2022
    sum_extract_specie = np.zeros((number_cells, number_parameter_mu * (len(index_time_step))));
    
    for ii in range(0, number_species):
        sum_extract_specie = sum_extract_specie + all_reduced_sol[ii * number_cells : (ii + 1) * number_cells, :];
       
    min_sum_snap_x = np.min(sum_extract_specie, 0);
    reshape_sum_snap = np.reshape(min_sum_snap_x, (number_parameter_mu, len(index_time_step)));    
    min_x_min_mu_reshape_sum_snap = np.min(reshape_sum_snap, 0);

    return min_x_min_mu_reshape_sum_snap  
    
    