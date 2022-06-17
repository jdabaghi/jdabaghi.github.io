#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:55:38 2021

@author: carnot-smiles
"""

import numpy as np
import my_simpson
## Check mass conservation
def check_mass_conservation(number_species, center_space_mesh, number_cells, Solution, nn, mass_init_domain, Dx):
    
    mass_domain = np.zeros(number_species);
    
    for ii in range(0, number_species):
        mass = Dx * Solution[ii * number_cells : (ii + 1) * number_cells, nn];
        #mass = my_simpson.my_simpson(center_space_mesh, Solution[ii * number_cells : (ii + 1) * number_cells, nn], number_cells);             
        mass_domain[ii] = np.sum(mass);
        if abs(mass_domain[ii] - mass_init_domain[ii]) > 1e-2:
            print('error : mass conservation is not satisfied for specie {} at time {}'.format(ii, nn));
                    
    return mass_domain