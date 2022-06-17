#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:09:09 2021

@author: carnot-smiles
"""

import numpy as np
from numpy import linalg as LA

## Orthogonal projection

def orthogonal_projection(basis, vector, dim_r):
    
    coeff_projection = np.zeros(dim_r);
    
    for ll in range(0, dim_r):
        normalization = LA.norm(basis[:, ll], 2) * LA.norm(basis[:, ll], 2);
        coeff_projection[ll] = np.dot(np.transpose(basis[:, ll]), vector)/normalization;
        
    projection = np.sum(coeff_projection * basis, 1);
    
    return (coeff_projection, projection)
