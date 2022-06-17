#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:27:34 2021

@author: carnot-smiles
"""

import numpy as np
import numpy.matlib
import scipy.sparse as spsp


#def Matrix_Laplacian_flux(Sol_newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge):
#    
#    
#    ## Create a matrix that computes D_{K,ϭ}(u{i,K}^n)
#    extract_sol = np.reshape(Sol_newton, (number_species, number_cells));
#    diff_extract_sol = np.diff(extract_sol);
#    diff_operator = np.zeros((number_species * number_internal_edges, 2));
#    diff_operator[:, 0] = T_edge * np.reshape(diff_extract_sol, (1, number_species * number_internal_edges));
#    diff_operator[:, 1] = -T_edge * np.reshape(diff_extract_sol, (1, number_species * number_internal_edges));
#    
#    ##Create a sparse matrix from the previous matrix diff_operator
#    II = range(0, number_species * number_internal_edges);
#    index_diag_inf = []; #np.zeros(number_species * number_internal_edges);
#    index_diag_sup = []; #np.zeros(number_species * number_internal_edges);
#    
#    for ii in range(0, number_species):
#        index_diag_inf[ii * number_internal_edges : (ii + 1) * number_internal_edges] = range(ii * (number_internal_edges + 1), ii * (number_internal_edges + 1) + number_internal_edges);
#       # index_diag_sup[ii * (number_species + 1) : (ii + 1) * (number_species + 1)] = range(ii * (number_internal_edges + 1) + 1, (ii + 1) * (number_internal_edges + 1));
#    
#    index_diag_sup = [index_diag_inf[ii] + 1 for ii in range(0, len(index_diag_inf))];
#    
#    Val1 = diff_operator[:, 0];
#    Val2 = diff_operator[:, 1];
#    temp_Matrix_flux1 = spsp.coo_matrix((Val1,(II, index_diag_inf)), shape=(number_internal_edges * number_species, number_cells * number_species)).toarray();
#    temp_Matrix_flux2 = spsp.coo_matrix((Val2,(II, index_diag_sup)), shape=(number_internal_edges * number_species, number_cells * number_species)).toarray();
#    Matrix_flux = temp_Matrix_flux1 + temp_Matrix_flux2;
#    #Matrix_flux = Matrix_flux.todense();
#    
#    return Matrix_flux

## Function that computes the nonlinear function G^n(U^n)=0

def function_Ga(Sol_newton, Sol_init, number_cells, number_species, Dx, Dt):
    
    ## Compute function G_{i,K}^{a,n}(U^n). 
    ## G_{i,K}^{a,n}(U^n) = m_K/Dt * (u_{i,K}^n - u_{i,K}^{n-1})
    fun_Ga = Dx/Dt * (Sol_newton - Sol_init);
    
    return fun_Ga

#def function_Gb(Sol_newton, number_species, number_cells, number_internal_edges, Dx, Dt, a_star, T_edge):
#    
#    #G_{i,K}^{b,n}(U^n) = Sum_{ϭ \in EKint} a* T_{ϭ} D_{K,ϭ} u_{i}^n
#
#    fun_Gb = a_star * sum(Matrix_Laplacian_flux(Sol_newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge));
#    
#    return fun_Gb


def function_Gb(Sol_newton, number_species, number_cells, number_internal_edges, index_internal_edges, Dx, Dt, a_star, T_edge):
    
    func_Gb_bis_all = np.zeros(number_cells * number_species);
    loc_mat = np.zeros((2, 1));    
    for ii in range(0, number_species):
        func_Gb_bis = np.zeros(number_cells);  
        for jj in index_internal_edges:
            ## find elements that share the edge sigma
            glob_index = np.array([jj - 1, jj]);
            ## compute loc matrix
            loc_mat[0] = a_star * T_edge * (Sol_newton[glob_index[1] + ii * number_cells] - Sol_newton[glob_index[0] + ii * number_cells]);
            loc_mat[1] = - (a_star * T_edge * (Sol_newton[glob_index[1] + ii * number_cells] - Sol_newton[glob_index[0] + ii * number_cells]));
                            
            func_Gb_bis[glob_index[0]] = func_Gb_bis[glob_index[0]] + loc_mat[0];
            
            func_Gb_bis[glob_index[1]] = func_Gb_bis[glob_index[1]] + loc_mat[1];
        
        func_Gb_bis_all[ii * number_cells : (ii + 1) * number_cells] = func_Gb_bis;
        
               
        
    return  func_Gb_bis_all
    
    
    
#def function_Gc(number_cells, number_species, edge_solution, Matrix_flux, Mat_coeff_astar, index_different_specie):
#    
#    fun_Gc2 = np.zeros(number_cells * number_species);
#    
#    for ii in range(0, number_species):
#        ## extract for i=0 [A-a*](0,1), [A-a*](0,2)... for i=1, extract [A-a*](1,0), [A-a*](1, 2)...
#        extract_other_indices = Mat_coeff_astar[ii, index_different_specie[ii, :]];
#        for jj in range(0, number_cells):
#            current_Flux1 = Matrix_flux[:, ii * number_cells + jj];
#            index_nonzero_Flux1 = np.nonzero(current_Flux1);
#            temp_nonzero_Flux1 = np.transpose(current_Flux1[index_nonzero_Flux1]);
#            indices_other_edge = number_cells * index_different_specie[ii, :] + jj;
#            temp1 = [np.dot(edge_solution[indices_other_edge, 0], extract_other_indices), np.dot(edge_solution[indices_other_edge, 1], extract_other_indices)];        
#            if jj == 0 or jj == number_cells - 1:
#                temp_nonzero_Flux1 = [temp_nonzero_Flux1, 0];
#            
#        fun_Gc2[ii * number_cells + jj] = np.dot(temp1, temp_nonzero_Flux1);
#    
#    fun_Gc = fun_Gc2; #+ fun_Gc1
#    
#    return fun_Gc

## Other way to compute the function Gc by assembling

def function_Gc_bis(solution, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge):
    
    fun_Gc_bis = np.zeros(number_cells * number_species);
    
    for ii in range(0, number_species):
        temp_solution = solution[ii * number_cells : (ii + 1) * number_cells];
        counter = 0;
        global_cross_diff = np.zeros(number_cells);
        for jj in index_internal_edges:
            loc_cross_diff = np.zeros(2, );
            if counter == 0:
                aa = 0;
                bb = 0;
            else:
                aa = 1;
                bb = 0;
            ## find elements that share the edge sigma
            glob_index = [jj - 1, jj];
            extract_Mat_cross_diff = Mat_coeff_astar[ii, :];
            indices_edge = np.arange(0, number_species) * (number_cells) + counter;
            indices_edge1 = np.arange(0, number_species) * (number_cells) + counter + 1;
            extract_edge1 = edge_solution[indices_edge, aa];
            extract_edge2 = edge_solution[indices_edge1, bb];
            diff_sol1 = np.repeat(temp_solution[glob_index[1]] - temp_solution[glob_index[0]], number_species);
            diff_sol2 = np.repeat(temp_solution[glob_index[0]] - temp_solution[glob_index[1]], number_species);

            loc_cross_diff[0] = sum(T_edge * extract_Mat_cross_diff * extract_edge1 * diff_sol1);
            loc_cross_diff[1] = sum(T_edge * extract_Mat_cross_diff * extract_edge2 * diff_sol2);
            
            global_cross_diff[glob_index[0]] = global_cross_diff[glob_index[0]] + loc_cross_diff[0];
            global_cross_diff[glob_index[1]] = global_cross_diff[glob_index[1]] + loc_cross_diff[1];
            counter = counter + 1;
        
        fun_Gc_bis[ii * number_cells : (ii + 1) * number_cells] = global_cross_diff;
    
    return fun_Gc_bis


#def function_Gd(number_species, number_cells, edge_solution, Matrix_flux, Mat_coeff_astar, index_different_specie):
#    
#    
#    fun_Gd2 = np.zeros(number_cells * number_species); 
#        
#    for ii in range(0, number_species):
#        extract_other_indices = Mat_coeff_astar[ii, index_different_specie[ii, :]];
#        for jj in range(0, number_cells):
#            indices_other_flux = number_cells * index_different_specie[ii, :] + jj;
#            current_Flux = Matrix_flux[:, indices_other_flux];
#            index_nonzero_Flux = np.nonzero(current_Flux);
#            nonzero_Flux = current_Flux[index_nonzero_Flux];
#            indices_other_edge = edge_solution[ii * number_cells + jj, :];
#                    
#            if jj == 0 or jj == number_cells - 1:
#                nonzero_Flux = np.concatenate((nonzero_Flux, np.zeros(2, )));
#            else:
#                nonzero_Flux = np.reshape(nonzero_Flux, (2, 2));
#                            
#            nonzero_Flux = np.reshape(nonzero_Flux, (2, 2));
#
#            temp2 = [np.dot(nonzero_Flux[0, :], np.transpose(extract_other_indices)), np.dot(nonzero_Flux[1, :], np.transpose(extract_other_indices))];        
#            
#            fun_Gd2[ii * number_cells + jj] = np.dot(temp2, indices_other_edge);    
#                
#    fun_Gd = fun_Gd2 #+fun_Gd1 ;
#    
#    return fun_Gd

## Other way to compute the function Gc by assembling
    
def function_Gd_bis(solution, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge):
    
    fun_Gd_bis = np.zeros(number_cells * number_species);
    for ii in range(0, number_species):
        temp_edge_solution = edge_solution[ii * number_cells : (ii + 1) * number_cells, :];
        counter = 0;
        global_cross_diff = np.zeros(number_cells);
        for jj in index_internal_edges:
            loc_cross_diff = np.zeros(2, );
            if counter == 0:
                aa = 0;
                bb = 0;
            else:
                aa = 1;
                bb = 0;
            ## find elements that share the edge sigma
            glob_index = [jj - 1, jj];
            extract_Mat_cross_diff = Mat_coeff_astar[ii, :];
            indices_sol = np.arange(0, number_species) * (number_cells) + counter;
            indices_sol1 = np.arange(0, number_species) * (number_cells) + counter + 1;
            diff_sol1 = solution[indices_sol1] - solution[indices_sol];
            diff_sol2 = solution[indices_sol] - solution[indices_sol1];
                        
            temp_extract_edge1 = temp_edge_solution[counter, aa];
            extract_edge1 = np.repeat(temp_extract_edge1, number_species);
            
            temp_extract_edge2 = temp_edge_solution[counter + 1, bb];
            extract_edge2 = np.repeat(temp_extract_edge2, number_species);
            
            loc_cross_diff[0] = sum(T_edge * extract_Mat_cross_diff * extract_edge1 * diff_sol1);
            loc_cross_diff[1] = sum(T_edge * extract_Mat_cross_diff * extract_edge2 * diff_sol2);
            
            global_cross_diff[glob_index[0]] = global_cross_diff[glob_index[0]] + loc_cross_diff[0];
            global_cross_diff[glob_index[1]] = global_cross_diff[glob_index[1]] + loc_cross_diff[1];
            counter = counter + 1;
        
        fun_Gd_bis[ii * number_cells : (ii + 1) * number_cells] = global_cross_diff;
    
    return fun_Gd_bis   


