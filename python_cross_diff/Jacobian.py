#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:26:58 2021

@author: carnot-smiles
"""

## Jacobian function Jac_G

## Jacobian function of G_{i,K}^{a,n}(U^n)
import numpy as np
import scipy.sparse as spsp
import math
import nonlinear_function
import edge_function
import compute_safe_sol
## first def
#def edge_derivative(sol, a, b):
#    loc_jac_edge = np.zeros((2,2));
#    if min(sol[a], sol[b]) <= 0:
#        loc_jac_edge[0, 0] = 0;
#        loc_jac_edge[0, 1] = 0;
#        loc_jac_edge[1, 0] = 0;
#        loc_jac_edge[1, 1] = 0;
#    elif sol[a] == sol[b] and sol[a] >=0:
#        loc_jac_edge[0, 0] = 1;
#        loc_jac_edge[0, 1] = 0;
#        loc_jac_edge[1, 0] = 0;
#        loc_jac_edge[1, 1] = 1;
#    else:
#        loc_jac_edge[0, 0] = 1/(math.log(sol[a]) - math.log(sol[b])) - (sol[a] - sol[b])/(sol[a] * (math.log(sol[a]) - math.log(sol[b]))**2);
#        loc_jac_edge[0, 1] = -1/(math.log(sol[a]) - math.log(sol[b])) + (sol[a] - sol[b])/(sol[b] * (math.log(sol[a]) - math.log(sol[b]))**2);
#        loc_jac_edge[1, 0] = 1/(math.log(sol[a]) - math.log(sol[b])) - (sol[a] - sol[b])/(sol[a] * (math.log(sol[a]) - math.log(sol[b]))**2);
#        loc_jac_edge[1, 1] = -1/(math.log(sol[a]) - math.log(sol[b])) + (sol[a] - sol[b])/(sol[b] * (math.log(sol[a]) - math.log(sol[b]))**2);
#        
#    return loc_jac_edge
## second def
def edge_derivative(sol, a, b):
    err = 1e-8;
    loc_jac_edge = np.zeros((2,2));
    if min(sol[a], sol[b]) <= 0:
        loc_jac_edge[0, 0] = 0;
        loc_jac_edge[0, 1] = 0;
        loc_jac_edge[1, 0] = 0;
        loc_jac_edge[1, 1] = 0;
    elif abs(sol[a] - sol[b]) > err:
        loc_jac_edge[0, 0] = 1/(math.log(sol[a]) - math.log(sol[b])) - (sol[a] - sol[b])/(sol[a] * (math.log(sol[a]) - math.log(sol[b]))**2);
        loc_jac_edge[0, 1] = -1/(math.log(sol[a]) - math.log(sol[b])) + (sol[a] - sol[b])/(sol[b] * (math.log(sol[a]) - math.log(sol[b]))**2);
        loc_jac_edge[1, 0] = 1/(math.log(sol[a]) - math.log(sol[b])) - (sol[a] - sol[b])/(sol[a] * (math.log(sol[a]) - math.log(sol[b]))**2);
        loc_jac_edge[1, 1] = -1/(math.log(sol[a]) - math.log(sol[b])) + (sol[a] - sol[b])/(sol[b] * (math.log(sol[a]) - math.log(sol[b]))**2);
    else:
        loc_jac_edge[0, 0] = 1;
        loc_jac_edge[0, 1] = 0;
        loc_jac_edge[1, 0] = 0;
        loc_jac_edge[1, 1] = 1;        
    return loc_jac_edge




def Jacfunction_Ga(number_species, number_cells, Dx, Dt):

    Jac_Ga = Dx / Dt * np.eye(number_cells * number_species, number_cells * number_species);
    
    return Jac_Ga

### test Jacobian with the defition of the directional derivative

#def test_Jacobian_Ga(Sol_init, Sol_newton, number_cells, number_species, Dx, Dt):
#    #JacA = Jacfunction_Ga(number_species, number_cells, Dx, Dt);
#    Jac_Ga_dir_deriv = np.zeros((number_cells * number_species, number_cells * number_species));
#    hh = 1e-15;
#    uu = np.eye(number_cells * number_species, number_cells * number_species);
#    for ii in range(0, number_cells * number_species):
#        for jj in range(0, number_cells * number_species):
#            FunGa_augmented = nonlinear_function.function_Ga(Sol_newton[ii] + hh * uu[jj, ii], Sol_init[ii], number_cells, number_species, Dx, Dt);
#            Fun_Ga = nonlinear_function.function_Ga(Sol_newton[ii], Sol_init[ii], number_cells, number_species, Dx, Dt);
#            Jac_Ga_dir_deriv[ii, jj] = (FunGa_augmented - Fun_Ga)/hh;
#                
#    testJac_Ga = Jac_Ga_dir_deriv; #JacA - Jac_Ga_dir_deriv;
#    
#    return testJac_Ga

### Jacobian function of G_{i,K}^{b,n}(U^n) with sparse structure
#def Jacfunction_Gb(number_cells, number_species, dist_center_cells, a_star):
#    Jac_b1 = np.zeros((number_cells, number_cells));
#    Jac_b2 = np.zeros((number_cells, number_cells));
#    Jac_b3 = np.zeros((number_cells, number_cells));
#    temp_Jacb = np.zeros((number_cells, number_cells));
#    
#    idex_line_diag_inf = np.arange(1, number_cells - 1);
#    index_row_diag_inf = np.arange(0, number_cells - 2);
#    idex_line_diag = np.arange(1, number_cells - 1);
#    index_row_diag = np.arange(1, number_cells - 1);
#    idex_line_diag_sup = np.arange(1, number_cells - 1);
#    index_row_diag_sup = np.arange(2, number_cells);
#    Val_b1 = a_star * (1/dist_center_cells) * np.ones(number_cells - 2);
#    Val_b2 = Val_b1;
#    Val_b3 = -a_star * (2/dist_center_cells) * np.ones(number_cells - 2);
#    
#    Jac_b1 = spsp.csc_matrix((Val_b1,(idex_line_diag_inf,index_row_diag_inf)), shape=(number_cells, number_cells));
#    Jac_b2 = spsp.csc_matrix((Val_b2,(idex_line_diag_sup,index_row_diag_sup)), shape=(number_cells, number_cells));
#    Jac_b3 = spsp.csc_matrix((Val_b3,(idex_line_diag,index_row_diag)), shape=(number_cells, number_cells));
#    temp_Jacb = Jac_b1 + Jac_b2 + Jac_b3;
#    temp_Jacb[(0,0)] = -a_star * (1/dist_center_cells);
#    temp_Jacb[(0,1)] = a_star * (1/dist_center_cells);
#    temp_Jacb[(-1, number_cells - 1)] = -a_star * (1/dist_center_cells);
#    temp_Jacb[(-1, number_cells - 2)] = a_star * (1/dist_center_cells);
#    
#    Jac_Gb = np.zeros((number_cells * number_species, number_cells * number_species));
#    temp_Jacb = temp_Jacb.todense();
#    ## Assemble the square matrix of size number_species*number_cells using the tridiag matrix temp_Jacb.
#    ##For instance if one has 3 components the matrix Jacb has the shape
#    #Jacb = [A     0      0]
#     #       0     A      0
#      #      0     0      A] with A=tempJac_b
#      
#    for ii in range(0, number_species):
#        Jac_Gb[ii * number_cells : (ii + 1) * number_cells, ii * number_cells : (ii + 1) * number_cells] = temp_Jacb;
#        
#    return Jac_Gb

## Other way to compute the jacobian matrix of Gb without sparse
def Jacfunction_Gb_bis(number_cells, number_species, dist_center_cells, a_star, index_internal_edges, T_edge):
    
    temp_Jac_Gb_bis = np.zeros((number_cells, number_cells));
    Jac_Gb_bis = np.zeros((number_species * number_cells, number_species * number_cells));
    loc_jac_edge = np.zeros((2, 2));
    loc_jac_edge[0, 0] = -a_star * T_edge;
    loc_jac_edge[0, 1] = a_star * T_edge;
    loc_jac_edge[1, 0] = a_star * T_edge;
    loc_jac_edge[1, 1] = -a_star * T_edge;
    
    for ii in index_internal_edges:
        ## find elements that share the edge sigma
        glob_index = [ii - 1, ii];
        
        temp_Jac_Gb_bis[glob_index[0], glob_index[0]] = temp_Jac_Gb_bis[glob_index[0], glob_index[0]] + loc_jac_edge[0, 0];
        
        temp_Jac_Gb_bis[glob_index[0], glob_index[1]] = temp_Jac_Gb_bis[glob_index[0], glob_index[1]] + loc_jac_edge[0, 1];
        
        temp_Jac_Gb_bis[glob_index[1], glob_index[0]] = temp_Jac_Gb_bis[glob_index[1], glob_index[0]] + loc_jac_edge[1, 0];
        
        temp_Jac_Gb_bis[glob_index[1], glob_index[1]] = temp_Jac_Gb_bis[glob_index[1], glob_index[1]] + loc_jac_edge[1, 1];
    
     ## Assemble the square matrix of size number_species*number_cells using the tridiag matrix temp_Jacb.
    ##For instance if one has 3 components the matrix Jacb has the shape
    #Jacb = [A     0      0]
     #       0     A      0
      #      0     0      A] with A=tempJac_b
    for ii in range(0, number_species):
        Jac_Gb_bis[ii * number_cells : (ii + 1) * number_cells, ii * number_cells : (ii + 1) * number_cells] = temp_Jac_Gb_bis;
        
    
    return Jac_Gb_bis
    

### test Jacobian with the defition of the directional derivative
    
#def test_Jacobian_Gb(Sol_newton, number_cells, number_species, dist_center_cells, a_star, number_internal_edges, Dx, Dt, T_edge):
#    #JacB = Jacfunction_Gb(number_cells, number_species, dist_center_cells, a_star);
#    Jac_Gb_dir_deriv = np.zeros((number_cells * number_species, number_cells * number_species));
#    hh = 1e-15;
#    uu = np.eye(number_cells * number_species, number_cells * number_species);
#    
#    for ii in range(0, number_cells * number_species):
#        temp = nonlinear_function.function_Gb(Sol_newton + hh * uu[:, ii], number_species, number_cells, number_internal_edges, Dx, Dt, a_star, T_edge);
#        temp1 = nonlinear_function.function_Gb(Sol_newton, number_species, number_cells, number_internal_edges, Dx, Dt, a_star, T_edge);
#        for jj in range(0, number_cells * number_species):
#            FunGb_augmented = temp[jj];
#            Fun_Gb = temp1[jj];
#            Jac_Gb_dir_deriv[ii, jj] = (FunGb_augmented - Fun_Gb)/hh;
#                
#    testJac_Gb = Jac_Gb_dir_deriv  ##JacB - Jac_Gb_dir_deriv;
#    
#    return testJac_Gb


 ###### JACOBIAN OF JC#######
def Jacobian_edgeGc(edge_sol, number_cells, number_internal_edges, 
                    index_internal_edges, Dx, Dt, sol_current, current_specie, number_species, Mat_coeff_astar, T_edge):
    
    loc_jac_edge = np.zeros((2, 2));
    Jacobian_diag_matrix = np.zeros((number_cells, number_cells));
    counter = 1;
    indices_edge_all_species = np.zeros((number_species, 2), dtype=int);
    for ii in index_internal_edges:
        ## find elements that share the edge sigma
        glob_index = np.array([ii - 1, ii]);
        
        loc_jac_edge = edge_derivative(sol_current, glob_index[0], glob_index[1]);
        
        if counter == 1:
            aa = 0;
            bb = 0;
        else:
            aa = 1;
            bb = 0;
            
        for jj in range(0, number_species):   
             
            indices_edge_all_species[jj, :] = glob_index + jj * number_cells;      
        
        
        temp_dot = np.dot(Mat_coeff_astar[current_specie, :], edge_sol[indices_edge_all_species[:, 0].astype(int), aa]);                     
        temp_dot1 = np.dot(Mat_coeff_astar[current_specie, :], edge_sol[indices_edge_all_species[:, 1].astype(int), bb]);     
        
        Jacobian_diag_matrix[glob_index[0], glob_index[0]] = Jacobian_diag_matrix[glob_index[0], glob_index[0]] + T_edge * (-temp_dot + loc_jac_edge[0, 0] * Mat_coeff_astar[current_specie, current_specie] * (sol_current[glob_index[1]] - sol_current[glob_index[0]]));
        
        Jacobian_diag_matrix[glob_index[0], glob_index[1]] = Jacobian_diag_matrix[glob_index[0], glob_index[1]] + T_edge * (temp_dot + loc_jac_edge[0, 1] * Mat_coeff_astar[current_specie, current_specie] * (sol_current[glob_index[1]] - sol_current[glob_index[0]]));
            
        Jacobian_diag_matrix[glob_index[1], glob_index[0]] = Jacobian_diag_matrix[glob_index[1], glob_index[0]] + T_edge * (temp_dot1 + loc_jac_edge[1, 0] * Mat_coeff_astar[current_specie, current_specie] * (sol_current[glob_index[0]] - sol_current[glob_index[1]]));
            
        Jacobian_diag_matrix[glob_index[1], glob_index[1]] = Jacobian_diag_matrix[glob_index[1], glob_index[1]] + T_edge * (-temp_dot1 + loc_jac_edge[1, 1] * Mat_coeff_astar[current_specie, current_specie] * (sol_current[glob_index[0]] - sol_current[glob_index[1]]));
        
        counter = counter + 1;
        
    return  Jacobian_diag_matrix
    

    
def Jacobian_Gc_cross_diff(edge_sol, sol_current, index_internal_edges, Dx, Dt, number_cells, T_edge):
    
    loc_jac_edge = np.zeros((2, 2));
    counter = 1;
    Jacobian_cross_diff_c = np.zeros((number_cells, number_cells));
    
    for ii in index_internal_edges:
        ## find elements that share the edge sigma
        glob_index = np.array([ii - 1, ii]);
        
        loc_jac_edge = edge_derivative(sol_current, glob_index[0], glob_index[1]);
        
        if counter == 1:
            aa = 0;
            bb = 0;
        else:
            aa = 1;
            bb = 0;
                                       
        Jacobian_cross_diff_c[glob_index[0], glob_index[0]] = Jacobian_cross_diff_c[glob_index[0], glob_index[0]] + T_edge * loc_jac_edge[0, 0] * (sol_current[glob_index[1]] - sol_current[glob_index[0]]);
        
        Jacobian_cross_diff_c[glob_index[0], glob_index[1]] = Jacobian_cross_diff_c[glob_index[0], glob_index[1]] + T_edge * loc_jac_edge[0, 1] * (sol_current[glob_index[1]] - sol_current[glob_index[0]]);
        
        Jacobian_cross_diff_c[glob_index[1], glob_index[0]] = Jacobian_cross_diff_c[glob_index[1], glob_index[0]] + T_edge * loc_jac_edge[1, 0] * (sol_current[glob_index[0]] - sol_current[glob_index[1]]);
        
        Jacobian_cross_diff_c[glob_index[1], glob_index[1]] = Jacobian_cross_diff_c[glob_index[1], glob_index[1]] + T_edge * loc_jac_edge[1, 1] * (sol_current[glob_index[0]] - sol_current[glob_index[1]]);
        
        counter = counter + 1;

    return Jacobian_cross_diff_c

## Assembling jacobian of function Gc
    
def Assembling_JacGc(number_species, number_cells, Sol_newton, edge_solution, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge):
    
    Jac_Gc = np.zeros((number_species * number_cells, number_species * number_cells));
    
    for ii in range(0, number_species):
        restricted_solution_specie = Sol_newton[ii * number_cells : (ii + 1) * number_cells];
        restricted_edge_sol = edge_solution[ii * number_cells : (ii + 1) * number_cells, :];
        for jj in range(0, number_species):
            if ii == jj:
                Jac_Gc[ii * number_cells : (ii + 1) * number_cells, jj * number_cells : (jj + 1) * number_cells] = Jacobian_edgeGc(edge_solution, number_cells, number_internal_edges, index_internal_edges, Dx, Dt, restricted_solution_specie, ii, number_species, Mat_coeff_astar, T_edge);
            else:
                Jac_Gc[ii * number_cells : (ii + 1) * number_cells, jj * number_cells : (jj + 1) * number_cells] = Mat_coeff_astar[ii, jj] * Jacobian_Gc_cross_diff(restricted_edge_sol, restricted_solution_specie, index_internal_edges, Dx, Dt, number_cells, T_edge);
    
    return Jac_Gc


### test Jacobian with the defition of the directional derivative

#def test_Jacobian_Gc(Sol_newton, number_cells, number_species, dist_center_cells, a_star, number_internal_edges, Dx, Dt, T_edge, Mat_coeff_astar, index_different_specie):
#    #JacB = Jacfunction_Gb(number_cells, number_species, dist_center_cells, a_star);
#    Jac_Gc_dir_deriv = np.zeros((number_cells * number_species, number_cells * number_species));
#    hh = 1e-15;
#    uu = np.eye(number_cells * number_species, number_cells * number_species);
#    edge_solution_augmented = np.zeros((number_cells * number_species, 2));
#    edge_solution = np.zeros((number_cells * number_species, 2));
#    for ii in range(0, number_species):
#        edge_solution_augmented[ii * number_cells : (ii + 1) * number_cells, :] = edge_function.edge_solution(Sol_newton[ii * number_cells : (ii + 1) * number_cells] + hh * uu[ii * number_cells : (ii + 1) * number_cells, ii], number_cells, number_internal_edges);  
#        edge_solution[ii * number_cells : (ii + 1) * number_cells, :] = edge_function.edge_solution(Sol_newton[ii * number_cells : (ii + 1) * number_cells], number_cells, number_internal_edges); 
#    
#    for ii in range(0, number_cells * number_species):
#        temp = nonlinear_function.function_Gc(number_cells, number_species, edge_solution_augmented, nonlinear_function.Matrix_Laplacian_flux(Sol_newton + hh * uu[:, ii], number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
#        temp1 = nonlinear_function.function_Gc(number_cells, number_species, edge_solution, nonlinear_function.Matrix_Laplacian_flux(Sol_newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
#        for jj in range(0, number_cells * number_species):
#            FunGc_augmented = temp[jj];
#            Fun_Gc = temp1[jj];
#            Jac_Gc_dir_deriv[ii, jj] = (FunGc_augmented - Fun_Gc)/hh;
#                
#    testJac_Gc = Jac_Gc_dir_deriv;  ##JacB - Jac_Gb_dir_deriv;
#    
#    return testJac_Gc


def Jacobian_edgeGd(sol, edge_sol, number_cells, number_internal_edges, index_internal_edges, Dx, Dt, sol_current, current_specie, number_species, Mat_coeff_astar, T_edge):
    
    loc_jac_edge = np.zeros((2, 2));
    Jacobian_edge_matrix1 = np.zeros((number_cells, number_cells));
    counter = 1;
    indices_edge_all_species = np.zeros((number_species, 2));
    for ii in index_internal_edges:
        ## find elements that share the edge sigma
        glob_index = np.array([ii - 1, ii]);
        ## compute loc jac edge        
        loc_jac_edge = edge_derivative(sol_current, glob_index[0], glob_index[1]);
        
        if counter == 1:
            aa = 0;
            bb = 0;
        else:
            aa = 1;
            bb = 0;
            
        for jj in range(0, number_species):   
             
            indices_edge_all_species[jj, :] = glob_index + jj * (number_cells - 1);      
             
            temp_dot = np.dot(Mat_coeff_astar[current_specie, :], sol[indices_edge_all_species[:, 1].astype(int)] - sol[indices_edge_all_species[:, 0].astype(int)]);                     
            temp_dot1 = np.dot(Mat_coeff_astar[current_specie, :], (sol[indices_edge_all_species[:, 0].astype(int)] - sol[indices_edge_all_species[:, 1].astype(int)]));
             
            Jacobian_edge_matrix1[glob_index[0], glob_index[0]] = Jacobian_edge_matrix1[glob_index[0], glob_index[0]] + T_edge * (loc_jac_edge[0, 0] * temp_dot - Mat_coeff_astar[current_specie, current_specie] * edge_sol[glob_index[0], aa]);
        
            Jacobian_edge_matrix1[glob_index[0], glob_index[1]] = Jacobian_edge_matrix1[glob_index[0], glob_index[1]] + T_edge * (loc_jac_edge[0, 1] * temp_dot + Mat_coeff_astar[current_specie, current_specie] * edge_sol[glob_index[0], aa]);
            
            Jacobian_edge_matrix1[glob_index[1], glob_index[0]] = Jacobian_edge_matrix1[glob_index[1], glob_index[0]] + T_edge * (loc_jac_edge[1, 0] * temp_dot1 + Mat_coeff_astar[current_specie, current_specie] * edge_sol[glob_index[1], bb]);
            
            Jacobian_edge_matrix1[glob_index[1], glob_index[1]] = Jacobian_edge_matrix1[glob_index[1], glob_index[1]] + T_edge * (loc_jac_edge[1, 1] * temp_dot1 - Mat_coeff_astar[current_specie, current_specie] * edge_sol[glob_index[1], bb]);
        
        counter = counter + 1;
        
    return  Jacobian_edge_matrix1


def Jacobian_Gd_cross_diff(edge_sol, index_internal_edges, Dx, Dt, number_cells, T_edge):
    loc_mat = np.zeros((2, 2));
    counter = 1;
    Jacobian_cross_diff = np.zeros((number_cells, number_cells));
    
    for ii in index_internal_edges:
        ## find elements that share the edge sigma
        glob_index = np.array([ii - 1, ii]);
        
        if counter == 1:
            aa = 0;
            bb = 0;
        else:
            aa = 1;
            bb = 0;
        
        loc_mat[0, 0] = - T_edge * edge_sol[glob_index[0], aa];
        
        loc_mat[0, 1] = T_edge * edge_sol[glob_index[0], aa];
        
        loc_mat[1, 0] = T_edge * edge_sol[glob_index[1], bb];
        
        loc_mat[1, 1] = -T_edge * edge_sol[glob_index[1], bb];
                               
        Jacobian_cross_diff[glob_index[0], glob_index[0]] = Jacobian_cross_diff[glob_index[0], glob_index[0]] + loc_mat[0, 0];
        
        Jacobian_cross_diff[glob_index[0], glob_index[1]] = Jacobian_cross_diff[glob_index[0], glob_index[1]] + loc_mat[0, 1];
        
        Jacobian_cross_diff[glob_index[1], glob_index[0]] = Jacobian_cross_diff[glob_index[1], glob_index[0]] + loc_mat[1, 0];
        
        Jacobian_cross_diff[glob_index[1], glob_index[1]] = Jacobian_cross_diff[glob_index[1], glob_index[1]] + loc_mat[1, 1];
        
        counter = counter + 1;

    return Jacobian_cross_diff

## Assembling jacobian of function Gd
def Assembling_JacGd(number_cells, number_species, Sol_newton, edge_solution, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge):
    
    Jac_Gd = np.zeros((number_species * number_cells, number_species * number_cells));
    
    for ii in range(0, number_species):
        restricted_solution_specie = Sol_newton[ii * number_cells : (ii + 1) * number_cells];
        restricted_edge_sol = edge_solution[ii * number_cells : (ii + 1) * number_cells, :];
        for jj in range(0, number_species):
            if ii == jj:
                Jac_Gd[ii * number_cells : (ii + 1) * number_cells, jj * number_cells : (jj + 1) * number_cells] = Jacobian_edgeGd(Sol_newton, restricted_edge_sol, number_cells, number_internal_edges, index_internal_edges, Dx, Dt, restricted_solution_specie, ii, number_species, Mat_coeff_astar, T_edge);
            else:
                Jac_Gd[ii * number_cells : (ii + 1) * number_cells, jj * number_cells : (jj + 1) * number_cells] = Mat_coeff_astar[ii, jj] * Jacobian_Gd_cross_diff(restricted_edge_sol, index_internal_edges, Dx, Dt, number_cells, T_edge);

    return Jac_Gd                           

### reduced model Jacobian
    
def Jac_function_Ga_reduced(reduced_basis, Dx, Dt, number_species, number_cells):
    dim_reduced_basis = reduced_basis.shape[1];
    Ga_reduced = np.zeros((dim_reduced_basis, dim_reduced_basis));
    for ii in range(0, dim_reduced_basis):
        for jj in range(0, dim_reduced_basis):
            Ga_reduced[ii, jj] = Dx/Dt * np.dot(reduced_basis[:, ii], reduced_basis[:, jj]);
        
    return Ga_reduced

def Jac_function_Gb_reduced(reduced_basis, number_species, number_cells, number_internal_edges, index_internal_edges, Dx, Dt, a_star, T_edge):
    dim_reduced_basis = reduced_basis.shape[1];
    Gb_reduced = np.zeros((dim_reduced_basis, dim_reduced_basis));
    for kk in range(0, dim_reduced_basis):
        for ll in range(0, dim_reduced_basis):
            temp_basis = reduced_basis[:, ll];
            func_Gb_bis_all_reduced = np.zeros(number_cells * number_species);
            loc_mat = np.zeros((2, 1));    
            for ii in range(0, number_species):
                func_Gb_bis_reduced = np.zeros(number_cells);  
                for jj in index_internal_edges:
                    ## find elements that share the edge sigma
                    glob_index = np.array([jj - 1, jj]);
                    ## compute loc matrix
                    loc_mat[0] = a_star * T_edge * (temp_basis[glob_index[1] + ii * number_cells] - temp_basis[glob_index[0] + ii * number_cells]);
                    loc_mat[1] = - (a_star * T_edge * (temp_basis[glob_index[1] + ii * number_cells] - temp_basis[glob_index[0] + ii * number_cells]));
                                    
                    func_Gb_bis_reduced[glob_index[0]] = func_Gb_bis_reduced[glob_index[0]] + loc_mat[0];
                    
                    func_Gb_bis_reduced[glob_index[1]] = func_Gb_bis_reduced[glob_index[1]] + loc_mat[1];
                
                func_Gb_bis_all_reduced[ii * number_cells : (ii + 1) * number_cells] = func_Gb_bis_reduced;
                
            
            Gb_reduced[kk, ll] = -np.dot(reduced_basis[:, kk], func_Gb_bis_all_reduced);
            
    return  Gb_reduced


## In this function we compute the derivative from the chain rule

def derivation_chain(sol_safe_reduced_newton, reduced_basis, number_cells, number_species, dim_r):
    #reshape_temp1 = np.zeros((number_cells, number_species));
    reshape_extract_reduced_basis = np.zeros((number_cells, number_species));
    reshape_sol_safe = np.zeros((number_cells, number_species));
    #temp1 = compute_safe_sol.compute_safe_sol(zbar_newton, number_species, number_cells);
    #temp2 = np.zeros(number_cells);
    XXX = np.zeros((number_cells * number_species, dim_r));
    
#    ratio_expzbar_sum_expzbar = compute_safe_sol.compute_safe_sol(zbar_newton, number_species, number_cells);

    for jj in range(0, number_species):
            reshape_sol_safe[:, jj] = sol_safe_reduced_newton[jj * number_cells : (jj + 1) * number_cells];
    
    repeat_reshape_sol_safe = np.matlib.repmat(reshape_sol_safe, number_species, 1);
    for ll in range(0, dim_r):
        extract_reduced_basis = reduced_basis[:, ll];
        
        for jj in range(0, number_species):
            ## Compute the matrix [v_{1,K1}^l, v_{2,K1}^l,.....,v_{Ns,K1}^l
            ##                     v_{1,K2}^l, v_{2,K2}^l,.....,v_{Ns,K2}^l
            ##                                    ......                  
            ##                      v_{1,KN}^l, v_{2,KN}^l,.....,v_{Ns,KN}^l] ]
            
            reshape_extract_reduced_basis[:, jj] = extract_reduced_basis[jj * number_cells : (jj + 1) * number_cells];    
            repeat_reshape_extract_reduced_basis = np.matlib.repmat(reshape_extract_reduced_basis, number_species, 1);
            
        for jj in range(0, number_species * number_cells):            
            XXX[jj, ll] = sol_safe_reduced_newton[jj] * (extract_reduced_basis[jj] - np.dot(repeat_reshape_extract_reduced_basis[jj, :], repeat_reshape_sol_safe[jj, :]));
        
    
    
    ##################################
   
    
    return XXX 

## test the construction of the jacobian with the directional derivative
    
def test_Jacobian_matW_reduced(number_cells, number_species, reduced_basis, coeff_reduced, dim_r):
    
    hh = 1e-7;
    check_mat = np.zeros((number_species * number_cells, dim_r));
    id_Mat = np.eye(dim_r);
    
    for ll in range(0, dim_r):
        coeff_plus_h = coeff_reduced + hh * id_Mat[:, ll];
        coeff_plus_h_dot_basis = np.dot(reduced_basis, coeff_plus_h);
        coeff_dot_basis = np.dot(reduced_basis, coeff_reduced);
        ratio_augmented = compute_safe_sol.compute_safe_sol_bis(coeff_plus_h_dot_basis, number_species, number_cells);
        ratio_no = compute_safe_sol.compute_safe_sol_bis(coeff_dot_basis, number_species, number_cells);
        check_mat[:, ll] = 1/hh * (ratio_augmented - ratio_no);

    return check_mat    
    
def test_Jacobian_reduced(zbar_newton, reduced_basis, sol_init_Newton_reduced, sol_init_reduced, number_cells, number_species, dist_center_cells, a_star, number_internal_edges, index_internal_edges, edge_solution_reduced, Dx, Dt, T_edge, Mat_coeff_astar, index_different_specie, dim_r):
    reshape_numerator = np.zeros((number_cells, number_species));
    test_Jac = np.zeros((dim_r, dim_r));
    hh = 1e-8;
    uu = np.eye(dim_r, dim_r);
    funMat_reduced_augmented = np.zeros(dim_r);
    funMat_reduced = np.zeros(dim_r);
    funMat1 = nonlinear_function.function_Ga(sol_init_Newton_reduced, sol_init_reduced, number_cells, number_species, Dx, Dt);         
        
    funMat2 = nonlinear_function.function_Gb(sol_init_Newton_reduced, number_species, number_cells, 
                                                     number_internal_edges, index_internal_edges, Dx,
                                                     Dt, a_star, T_edge);
        
    funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
                
    funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
    
    funMat = funMat1 - funMat2 - funMat3 + funMat4;
        
    
    for ll in range(0, dim_r):
        
        Solution_reduced_augmented = compute_safe_sol.compute_safe_sol(zbar_newton + hh * np.dot(reduced_basis, uu[:, ll]), number_species, number_cells);
         
        funMat1_augmented = nonlinear_function.function_Ga(Solution_reduced_augmented, sol_init_reduced, number_cells, number_species, Dx, Dt);         
        
        funMat2_augmented = nonlinear_function.function_Gb(Solution_reduced_augmented, number_species, number_cells, 
                                                     number_internal_edges, index_internal_edges, Dx,
                                                     Dt, a_star, T_edge);
        
        funMat3_augmented = nonlinear_function.function_Gc_bis(Solution_reduced_augmented, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
                
        funMat4_augmented = nonlinear_function.function_Gd_bis(Solution_reduced_augmented, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
    
        funMat_augmented = funMat1_augmented - funMat2_augmented - funMat3_augmented + funMat4_augmented;
    
        ## Compute the reduced nonlinear function
        funMat_reduced_augmented = np.dot(np.transpose(reduced_basis), funMat_augmented);            
        funMat_reduced = np.dot(np.transpose(reduced_basis), funMat);
            
        test_Jac[:, ll] = (funMat_reduced_augmented - funMat_reduced)/hh;
    
#    for ll in range(0, dim_r):
#        numerator = np.exp(zbar_newton + hh * np.dot(reduced_basis, uu[:, ll]));
#        
#        for ii in range(0, number_species):
#            reshape_numerator[:, ii] = numerator[ii * number_cells : (ii + 1) * number_cells];
#        
#        sum_reshape_numerator = np.sum(reshape_numerator, 1);
#        
#        ## Compute from the reduced pb a safe solution satisfying the properties of continuous pb
#        uhat = (np.copy(numerator))/(np.tile(sum_reshape_numerator, (1, number_species)));
#        
#        uhat = np.copy(uhat.flatten());
#        
#        funMat1_augmented = nonlinear_function.function_Ga(uhat, sol_init_reduced, number_cells, number_species, Dx, Dt);         
#        
#        funMat2_augmented = nonlinear_function.function_Gb(uhat, number_species, number_cells, 
#                                                     number_internal_edges, index_internal_edges, Dx,
#                                                     Dt, a_star, T_edge);
#        
#        funMat3_augmented = nonlinear_function.function_Gc_bis(uhat, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
#                
#        funMat4_augmented = nonlinear_function.function_Gd_bis(uhat, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
#    
#        funMat_augmented = funMat1_augmented - funMat2_augmented - funMat3_augmented + funMat4_augmented;
#    
#        ## Compute the reduced nonlinear function
#        funMat_reduced_augmented = np.dot(np.transpose(reduced_basis), funMat_augmented);            
#        funMat_reduced = np.dot(np.transpose(reduced_basis), funMat);
#            
#        test_Jac[:, ll] = (funMat_reduced_augmented - funMat_reduced)/hh; 
       
    return test_Jac
