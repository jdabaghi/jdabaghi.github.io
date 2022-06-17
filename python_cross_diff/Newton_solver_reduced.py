#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:07:52 2021

@author: carnot-smiles
"""

## Newton scheme for reduced pb

import numpy as np
import nonlinear_function
from numpy import linalg as LA
import Jacobian


def Newton_solver_reduced(sol_init_Newton, sol_init, edge_solution,
                  number_cells, number_internal_edges, index_internal_edges,
                  number_species, Dx, Dt, a_star, Mat_coeff_astar, index_different_specie,
                  dist_center_cells, T_edge, nn):
    
    counter = 0;
    tol_Newton = 1e-8;
    error_Newton = 100;
    #sol_init_Newton = np.reshape(sol_init_Newton, (len(sol_init_Newton), 1));
    
    funMat1 = nonlinear_function.function_Ga(sol_init_Newton, sol_init, number_cells, number_species,
                                             Dx, Dt);         
    
#    funMat2 = nonlinear_function.function_Gb(sol_init_Newton, number_species, number_cells,
#                                             number_internal_edges, Dx, Dt, a_star, T_edge);
    
    funMat2 = nonlinear_function.function_Gb_bis(sol_init_Newton, number_species, number_cells,
                                                 number_internal_edges, index_internal_edges, Dx,
                                                 Dt, a_star, T_edge);    
    
#    funMat3 = nonlinear_function.function_Gc(number_cells, number_species, edge_solution,
#                                             nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
    
    ## Compute function Gc with an assembling
    funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge);
    
    ## Compare the two constructions of Gc
    #print(funMat3 - funMat3_bis)
    ## Compute function Gd
    #funMat4 = nonlinear_function.function_Gd(number_species, number_cells, edge_solution, nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
    ## Compute function Gd with an assembling
    funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge)
    ## Compare the two constructions of Gc
    #print(funMat4 - funMat4_bis)
    
    funMat = funMat1 - funMat2 - funMat3 + funMat4;
    ## Compute Jacobian Ga
    Amat1 = Jacobian.Jacfunction_Ga(number_species, number_cells, Dx, Dt);
    ##Check Jacobian
    #test_Jac_Ga = Jacobian.test_Jacobian_Ga(sol_init, sol_init_Newton, number_cells, number_species, Dx, Dt);
    #norm_test_Jac_Ga = LA.norm(test_Jac_Ga, 2);
    #print(norm_test_Jac_Ga)
    
    ## Compute Jacobian Gb
    #Amat2 = Jacobian.Jacfunction_Gb(number_cells, number_species, dist_center_cells, a_star); 
    ## Other way to compute the jacobian matrix of Gb without sparse
    Amat2 = Jacobian.Jacfunction_Gb_bis(number_cells, number_species, dist_center_cells, a_star, index_internal_edges, T_edge)
    
    ##Check Jacobian Gb
    #test_Jac_Gb = Jacobian.test_Jacobian_Gb(sol_init_Newton, number_cells, number_species, dist_center_cells, a_star, number_internal_edges, Dx, Dt, T_edge);
    #norm_test_Jac_Gb = LA.norm(test_Jac_Gb, 2);
    #print(norm_test_Jac_Gb);
    
    Amat3 = Jacobian.Assembling_JacGc(number_species, number_cells, sol_init_Newton, edge_solution, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge); 
    ## Check Jacobian
    #test_Jac_Gc = Jacobian.test_Jacobian_Gc(sol_init_Newton, number_cells, number_species, dist_center_cells, a_star, number_internal_edges, Dx, Dt, T_edge, Mat_coeff_astar, index_different_specie);
    #norm_test_Jac_Gc = LA.norm(test_Jac_Gc, 2);
    #print(norm_test_Jac_Gc);
    
    Amat4 = Jacobian.Assembling_JacGd(number_cells, number_species, sol_init_Newton, edge_solution, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge);
    Amat = Amat1 - Amat2 - Amat3 + Amat4;           
    
    Fmat = np.dot(Amat, sol_init_Newton) - funMat;
    
    init_residual = Fmat - np.dot(Amat, sol_init_Newton);
    norm_init = LA.norm(init_residual, 2);
    
    while error_Newton > tol_Newton:
       
        funMat1 = nonlinear_function.function_Ga(sol_init_Newton, sol_init, number_cells, number_species, Dx, Dt);         
        
        funMat2 = nonlinear_function.function_Gb_bis(sol_init_Newton, number_species, number_cells, 
                                                     number_internal_edges, index_internal_edges, Dx,
                                                     Dt, a_star, T_edge);
        
        #funMat3 = nonlinear_function.function_Gc(number_cells, number_species, edge_solution, nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
        
        funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge);
        
        #funMat4 = nonlinear_function.function_Gd(number_species, number_cells, edge_solution, nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
        
        funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge);

        
        funMat = funMat1 - funMat2 - funMat3 + funMat4;
        
        Amat1 = Jacobian.Jacfunction_Ga(number_species, number_cells, Dx, Dt);

        Amat2 = Jacobian.Jacfunction_Gb_bis(number_cells, number_species, dist_center_cells, a_star, index_internal_edges, T_edge); 

        Amat3 = Jacobian.Assembling_JacGc(number_species, number_cells, sol_init_Newton, edge_solution, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge); 

        Amat4 = Jacobian.Assembling_JacGd(number_cells, number_species, sol_init_Newton, edge_solution, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge);

        Amat = Amat1 - Amat2 - Amat3 + Amat4;           
        
        Fmat = np.dot(Amat, sol_init_Newton) - funMat;
        Sol_newton = LA.solve(Amat, Fmat);  
        ## evaluate norm of ||G^n(U^{k,n})||
        residual = Fmat - np.dot(Amat, Sol_newton);
        residual_norm = LA.norm(residual, 2);
       
        error_Newton = max(abs(Sol_newton - sol_init_Newton));
        #error_Newton = residual_norm / norm_init;
        counter = counter + 1;
        sol_init_Newton = np.copy(Sol_newton);
    
    print("Newton cv in {} iterations at time step {}".format(counter, nn))


    return (Sol_newton, counter)