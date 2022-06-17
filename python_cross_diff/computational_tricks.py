#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 17:00:27 2021

@author: carnot-smiles
"""

## In this script we compute computational tricks

import numpy as np

##input : scal is for instance zbar_{1,K1} and vector_species is [zbar_{1,K1}, zbar_{2,K1}, zbar_{3,K1}]

def logsumexp(scal, vector_species):
    c = np.max(vector_species);
    #result = c + np.log(np.sum(np.exp(xx - c)));
    # return result
    y = np.exp(scal - c);
    y_vect = np.exp(vector_species - c);
    result = y / np.sum(y_vect);
    return result
   