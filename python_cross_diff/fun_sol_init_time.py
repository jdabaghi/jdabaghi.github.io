#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:48:54 2021

@author: carnot-smiles
"""

import numpy as np
import matplotlib.pyplot as plt
import math
def sol_init_time_sp1(xx, tol):    
    return 1./4 + 1./4 * np.cos(np.pi * xx)

def sol_init_time_sp2(xx, tol):
    return 1./4 + 1./4 * np.cos(np.pi * xx)
  
def sol_init_time_sp3(xx, tol):
    return 1./2 - 1./2 * np.cos(np.pi * xx)

######################################
###########  PVD PROCESS #############
######################################
    
def sol_init_time_sp1_PVD(xx, tol):
    return np.exp((-1.0/0.04) * (xx - 0.5) * (xx - 0.5))


def sol_init_time_sp2_PVD(xx, tol):
    return xx * xx + tol

def sol_init_time_sp3_PVD(xx, tol):
    return 1 - np.exp((-1.0/0.04) * (xx - 0.5) * (xx - 0.5))

def sol_init_time_sp4_PVD(xx, tol):
    return abs(np.sin(math.pi * xx))


    
def sol_init_time_sp1_disc(xx, tol, L):
    out = np.zeros(len(xx));
    for ii in range(0, len(xx)):
        if (xx[ii] >= 0.0/8.0 * L and xx[ii] <= 3.0/8.0 * L):
            out[ii] = 0.1;
        elif (xx[ii] > 3.0/8.0 * L and xx[ii] <= 5.0/8.0 * L):
            out[ii] = 0.8;
        
        elif (xx[ii] > 5.0/8.0 * L):
            out[ii] = 0.1;    
    
#    plt.plot(xx, out, 'b-^', linewidth = 1)
#    plt.xlabel('space mesh')
#    plt.ylabel('solution u1 at t=0')
#    plt.show()

    return out
        
        
def sol_init_time_sp2_disc(xx, tol, L):
    out = np.zeros(len(xx));
    for ii in range(0, len(xx)):
        if (xx[ii] >= 0.0/8.0 * L and xx[ii] <= 1.0/8.0 * L):
            out[ii] = 0.1;
        elif (xx[ii] > 1.0/8.0 * L and xx[ii] <= 3.0/8.0 * L):
            out[ii] = 0.8;
        elif (xx[ii] > 3.0/8.0 * L and xx[ii] <= 5.0/8.0 * L):
            out[ii] = 0.1;
        elif (xx[ii] > 5.0/8.0 * L and xx[ii] <= 7.0/8.0 * L):
            out[ii] = 0.8; #0.8;
        elif (xx[ii] > 7.0/8.0 * L):
            out[ii] = 0.1
            

#    plt.plot(xx, out, 'r-^', linewidth = 1)
#    plt.xlabel('space mesh')
#    plt.ylabel('solution u2 at t=0')
#    plt.show()   
    return out
#    
def sol_init_time_sp3_disc(xx, tol, L):
    out = np.zeros(len(xx));
    for ii in range(0, len(xx)):
        if (xx[ii] >= 0.0/8.0 * L and xx[ii] <= 1.0/8.0 * L):
            out[ii] = 0.8; 
        
        elif (xx[ii] > 1.0/8.0 * L and xx[ii] <= 7.0/8.0 * L):
            out[ii] = 0.1;
        
        elif (xx[ii] > 7.0/8.0 * L):
            out[ii] = 0.8;  
          
        
#    plt.plot(xx, out, 'g-^', linewidth = 1)
#    plt.xlabel('space mesh')
#    plt.ylabel('solution u3 at t=0')
#    plt.show()
    return out
     
