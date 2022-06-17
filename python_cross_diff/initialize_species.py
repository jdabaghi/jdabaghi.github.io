#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:58:26 2021

@author: carnot-smiles
"""
## Parameters for the species
import numpy as np

def indices_species(number_cells, number_species):

    if number_species == 2:
        index_different_specie = np.array([[1, 0]]);
    elif number_species == 3:
        index_different_specie = np.array([[1, 2], [0, 2], [0, 1]]);    
        ## all indices except index i
        indices_different_first_specie = np.zeros((number_cells, number_species - 1));
        indices_different_second_specie = np.zeros((number_cells, number_species - 1));
        indices_different_third_specie = np.zeros((number_cells, number_species - 1));
        indices_different_first_specie[:, 0] = range(number_cells, number_cells * (number_species - 1));
        indices_different_first_specie[:, 1] = range(2 * number_cells, number_cells * number_species);
        indices_different_second_specie[:, 0] = range(0, number_cells);
        indices_different_second_specie[:, 1] = range(2 * number_cells, number_cells * number_species);
        indices_different_third_specie[:, 0] = range(0, number_cells);
        indices_different_third_specie[:, 1] = range(number_cells, 2 * number_cells);    
        ## Matrix for cross diffusion coefficients
    else:
        index_different_specie = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]);

    return index_different_specie

