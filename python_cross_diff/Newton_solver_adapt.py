#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:02:05 2022

@author: carnot-smiles
"""

## Time Adaptivity for Newton resolution

import numpy as np
import nonlinear_function
from numpy import linalg as LA
import Jacobian
import matplotlib.pyplot as plt
import compute_safe_sol
import edge_function
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres


def Newton_solver_adapt_fine_pb(sol_init_Newton, sol_init, edge_solution,
                  number_cells, number_internal_edges, index_internal_edges,
                  number_species, Dx, Dt, a_star, Mat_coeff_astar, index_different_specie,
                  dist_center_cells, T_edge, nn):
    
    counter = 0;
    tol_Newton = 1e-8;
    error_Newton = 100;
    Nmax_Newton = 100;
    plot_error_Newton = np.zeros(Nmax_Newton);
    plot_error_Newton[0] = 100; 

    funMat1 = nonlinear_function.function_Ga(sol_init_Newton, sol_init, number_cells, number_species,
                                             Dx, Dt);         
    
  
    funMat2 = nonlinear_function.function_Gb(sol_init_Newton, number_species, number_cells,
                                                 number_internal_edges, index_internal_edges, Dx,
                                                 Dt, a_star, T_edge);    
        
    ## Compute function Gc with an assembling
    funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge);
    
    ## Compute function Gd
    #funMat4 = nonlinear_function.function_Gd(number_species, number_cells, edge_solution, nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
    ## Compute function Gd with an assembling
    funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge)
    
    funMat = funMat1 - funMat2 - funMat3 + funMat4;
    ## Compute Jacobian Ga
    Amat1 = Jacobian.Jacfunction_Ga(number_species, number_cells, Dx, Dt);
    ##Check Jacobian Ga
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
        
        counter = counter + 1;
       
        funMat1 = nonlinear_function.function_Ga(sol_init_Newton, sol_init, number_cells, number_species, Dx, Dt);         
        
        funMat2 = nonlinear_function.function_Gb(sol_init_Newton, number_species, number_cells, 
                                                     number_internal_edges, index_internal_edges, Dx,
                                                     Dt, a_star, T_edge);
        
        #funMat3 = nonlinear_function.function_Gc(number_cells, number_species, edge_solution, nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
        
        funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge);
        
        #funMat4 = nonlinear_function.function_Gd(number_species, number_cells, edge_solution, nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
        
        funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton, number_cells, index_internal_edges, number_species, edge_solution, Mat_coeff_astar, T_edge);

        funMat = funMat1 - funMat2 - funMat3 + funMat4;
        
        #Amat1 = Jacobian.Jacfunction_Ga(number_species, number_cells, Dx, Dt);

        #Amat2 = Jacobian.Jacfunction_Gb_bis(number_cells, number_species, dist_center_cells, a_star, index_internal_edges, T_edge); 

        Amat3 = Jacobian.Assembling_JacGc(number_species, number_cells, sol_init_Newton, edge_solution, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge); 

        Amat4 = Jacobian.Assembling_JacGd(number_cells, number_species, sol_init_Newton, edge_solution, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge);

        Amat = Amat1 - Amat2 - Amat3 + Amat4;           
        ## Conditionning of the Jacobian matrix
        #print(np.linalg.cond(Amat,np.inf))
        
        Fmat = np.dot(Amat, sol_init_Newton) - funMat;
        
        ## Jacobi preconditioneur
        Pcond = np.diag(np.diag(Amat));
        inv_Pcond = LA.inv(Pcond);     
#        Sol_newton = LA.solve(np.dot(inv_Pcond, Amat), np.dot(inv_Pcond, Fmat));  
        
        #Sol_newton = LA.solve(Amat, Fmat);
        
        ## GMRES 
        
        Sol_newton, exit_code = gmres(np.dot(inv_Pcond, Amat), np.dot(inv_Pcond, Fmat));
        
        ## evaluate L2 norm of ||G^n(U^{k,n})||
        #residual = Fmat - np.dot(Amat, Sol_newton);
        #residual_norm = LA.norm(residual, 2);
        

        ## evaluate Linf norm
        plot_error_Newton[counter] = max(abs(Sol_newton - sol_init_Newton));
        error_Newton = np.copy(plot_error_Newton[counter]);
        #error_Newton = residual_norm / norm_init;
        sol_init_Newton = np.copy(Sol_newton);
        
        if counter == Nmax_Newton - 1:
            break;
    
##    ## plot error   
#    plt.loglog(np.arange(1, counter + 1), plot_error_Newton[1 : counter + 1], 'r-o', linewidth = 1)
#    plt.xlabel('Newton iterations')
#    plt.ylabel('Error')
#    plt.show()    
#    
    print("Newton cv in {} iterations at time step {} and error is {}".format(counter, nn, error_Newton))

#    if min(Sol_newton) < 0:
#         print('Error :  solution should be positive')
    if min(Sol_newton) < 0:
        temp_neg = Sol_newton < 0;
        if np.max(Sol_newton[temp_neg]) > -1e-5:
            Sol_newton = abs(Sol_newton);
        else:
            print('Error: solution should be positive!')

    return (Sol_newton, counter)
    


def Newton_solver_adapt(reduced_basis, sol_init_Newton_reduced, newton_coeff_reduced, sol_init_reduced, coeff_reduced_init, edge_solution_reduced,
                      number_cells, number_internal_edges, index_internal_edges,
                      number_species, Dx, Dt, a_star, Mat_coeff_astar, index_different_specie,
                      dist_center_cells, T_edge, nn, choose_reduced_model, center_space_mesh):
        
    counter = 0;
    tol_Newton = 1e-8;
    error_Newton = 100;
    Nmax_Newton = 50;
    plot_error_Newton = np.zeros(Nmax_Newton);
    plot_error_Newton[0] = 100; 
    dim_r = reduced_basis.shape[1];
    funMat_reduced = np.zeros(dim_r);
    matW = np.zeros((number_species * number_cells, dim_r));

     ## Compute Jacobian Ga
    Amat1 = Jacobian.Jacfunction_Ga(number_species, number_cells, Dx, Dt);
    
    Amat2 = Jacobian.Jacfunction_Gb_bis(number_cells, number_species, dist_center_cells, a_star, index_internal_edges, T_edge);
      
    
    while error_Newton > tol_Newton:
        
        counter = counter + 1;
       
        funMat1 = nonlinear_function.function_Ga(sol_init_Newton_reduced, sol_init_reduced, number_cells, number_species, Dx, Dt);         
         
        ## TEST WITH LOG SUM EXP TRICK
#        funMat1_TEST = nonlinear_function.function_Ga(ratio_expzbar_sum_expzbar, ratio_expzbar_sum_expzbar_prev, number_cells, number_species,
#                                             Dx, Dt);
        
        funMat2 = nonlinear_function.function_Gb(sol_init_Newton_reduced, number_species, number_cells, 
                                                     number_internal_edges, index_internal_edges, Dx,
                                                     Dt, a_star, T_edge);
        ## TEST WITH LOG SUM EXP TRICK
#        funMat2_TEST = nonlinear_function.function_Gb(ratio_expzbar_sum_expzbar, number_species, number_cells,
#                                                 number_internal_edges, index_internal_edges, Dx,
#                                                 Dt, a_star, T_edge);   
         
        funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
        
        ## TEST WITH LOG SUM EXP TRICK                             
#        funMat3_TEST = nonlinear_function.function_Gc_bis(ratio_expzbar_sum_expzbar, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
 
    
        funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
    
        ## TEST WITH LOG SUM EXP TRICK 
#        funMat4_TEST = nonlinear_function.function_Gd_bis(ratio_expzbar_sum_expzbar, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge)

        funMat = funMat1 - funMat2 - funMat3 + funMat4;
        
        ## TEST WITH LOG SUM EXP TRICK 
#        funMat_TEST = funMat1_TEST - funMat2_TEST - funMat3_TEST + funMat4_TEST;
        
        ## Compute the reduced nonlinear function
        
        funMat_reduced = np.dot(np.transpose(reduced_basis), funMat);
        
        ## TEST WITH LOG SUM EXP TRICK 
#        funMat_reduced_TEST = np.dot(np.transpose(reduced_basis), funMat_TEST);

        Amat3 = Jacobian.Assembling_JacGc(number_species, number_cells, sol_init_Newton_reduced, edge_solution_reduced, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge); 

        ## TEST WITH LOG SUM EXP TRICK 
#        Amat3_TEST = Jacobian.Assembling_JacGc(number_species, number_cells, ratio_expzbar_sum_expzbar, edge_solution_reduced, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge); 

        Amat4 = Jacobian.Assembling_JacGd(number_cells, number_species, sol_init_Newton_reduced, edge_solution_reduced, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge);

        ## TEST WITH LOG SUM EXP TRICK
#        Amat4_TEST = Jacobian.Assembling_JacGd(number_cells, number_species, sol_init_Newton_reduced, edge_solution_reduced, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge);

        Amat = Amat1 - Amat2 - Amat3 + Amat4; 
        
        ## TEST WITH LOG SUM EXP TRICK
#        Amat_TEST = Amat1 - Amat2 - Amat3_TEST + Amat4_TEST;
    
        matW = Jacobian.derivation_chain(sol_init_Newton_reduced, reduced_basis, number_cells, number_species, dim_r);

        ## Check construction of jacobian matrix      
        #check_matW = Jacobian.test_Jacobian_matW_reduced(number_cells, number_species, reduced_basis, newton_coeff_reduced, dim_r);
        
#        if np.max(abs(matW - check_matW)) > 1e-6:
#            print("error in construction of the matrix W at iterations {} at time step {}".format(counter, nn))
#            print(np.max(abs(matW - check_matW)))
        
        ##### ADAPTIVE MESH REFINEMENT BIS ########
#        if np.max(abs(matW)) < 1e-10:
#            sol_init_Newton_reduced = np.copy(sol_init_reduced);
#            Solution_reduced = np.copy(sol_init_Newton_reduced);
#            newton_coeff_reduced = np.copy(coeff_reduced_init);
#            counter = 1e9;
#            break;
            
            
        
        ## Chain rule to compute the reduced Jacobian
        Amat_reduced = np.dot(np.transpose(reduced_basis), np.dot(Amat, matW));           
       
        Fmat_reduced = np.dot(Amat_reduced, newton_coeff_reduced) - funMat_reduced;     
        
        #Amat_reduced_bis = np.dot(np.transpose(reduced_basis), np.dot(Amat, check_matW));
        
        #Fmat_reduced_bis = np.dot(Amat_reduced_bis, newton_coeff_reduced) - funMat_reduced;
        
        ## Preconditionning
        Pcond = np.diag(np.diag(Amat_reduced));
        inv_Pcond = LA.inv(Pcond);     
        
        ## Conditionning of the Jacobian matrix
        #print(np.linalg.cond(Amat,np.inf))
        
         ## GMRES
        Sol_newton_coeff_reduced, exit_code = gmres(np.dot(inv_Pcond, Amat_reduced), np.dot(inv_Pcond, Fmat_reduced));
        
        #Sol_newton_coeff_reduced_bis, exit_code = gmres(np.dot(inv_Pcond, Amat_reduced_bis), np.dot(inv_Pcond, Fmat_reduced_bis));

        
        ## Compute the Newton solution which is the coefficients ie the decomposition of zbar in the reduced basis 
        #Sol_newton_coeff_reduced = LA.solve(Amat_reduced, Fmat_reduced);  
    
        ## Compute the solution zbar_newton
        zbar_newton = np.sum(Sol_newton_coeff_reduced * reduced_basis, 1);
    
         ## evaluate L2 norm of ||G^n(U^{k,n})||
#        #residual = Fmat - np.dot(Amat, Sol_newton);
#        #residual_norm = LA.norm(residual, 2);
        
        ## Compute reduced solution \bar{Ukn} 
        Solution_reduced = compute_safe_sol.compute_safe_sol_bis(zbar_newton, number_species, number_cells);
        
     
         ## evaluate Linf norm for safe solution Ubar
        plot_error_Newton[counter] = max(abs(Solution_reduced - sol_init_Newton_reduced));
        
#        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
#        ax1.plot(center_space_mesh, Solution_reduced[0 : number_cells], 'b-o', linewidth = 1)
#        ax1.set_xlabel('abcissa')
#        ax1.set_ylabel('u1 safe reduced at newton iter k=%d' %counter)
#        ax2.plot(center_space_mesh, Solution_reduced[number_cells : 2 * number_cells], 'r-^', linewidth = 1)
#        ax2.set_xlabel('abcissa')
#        ax2.set_ylabel('u2 safe reduced at newton iter k=%d' %counter)
#        ax3.plot(center_space_mesh, Solution_reduced[2 * number_cells : 3 * number_cells], 'g-s', linewidth = 1)
#        ax3.set_xlabel('abcissa')
#        ax3.set_ylabel('u3 safe reduced at newton iter k=%d' %counter)
        #file_name = "reduced_sol at k {} and n {}".format(counter, nn)
        #fig.savefig(os.path.join(my_path2, file_name))        
        #plt.show()
        #plt.close()
#        
        error_Newton = np.copy(plot_error_Newton[counter]);
                
        if counter == Nmax_Newton - 1:
            break;
        
        newton_coeff_reduced = np.copy(Sol_newton_coeff_reduced);
        
        ## TEST WITH LOG SUM EXP TRICK
#        newton_coeff_reduced_TEST = np.copy(newton_coeff_reduced_TEST);

        
        sol_init_Newton_reduced = np.copy(Solution_reduced);
    
        for ii in range(0, number_species):
    
                edge_solution_reduced[ii * number_cells : (ii + 1) * number_cells, :] = edge_function.edge_solution_reduced_safe(sol_init_Newton_reduced[ii * number_cells : (ii + 1) * number_cells], number_cells, number_species, number_internal_edges);  

    return (error_Newton, newton_coeff_reduced, Solution_reduced, counter)



def Newton_solver_adapt_unsafe_reduced_pb(reduced_basis, sol_init_Newton_reduced, newton_coeff_reduced, sol_init_reduced, coeff_reduced_init, edge_solution_reduced,
                  number_cells, number_internal_edges, index_internal_edges,
                  number_species, Dx, Dt, a_star, Mat_coeff_astar, index_different_specie,
                  dist_center_cells, T_edge, nn):
    
    counter = 0;
    tol_Newton = 1e-8;
    error_Newton = 100;
    Nmax_Newton = 50;
    plot_error_Newton = np.zeros(Nmax_Newton);
    plot_error_Newton[0] = 100; 
    dim_r = reduced_basis.shape[1];
    funMat_reduced = np.zeros(dim_r);
    
    ## we evaluate the Jacobian matrix G at reduced solution U^{n,k-1}
    funMat1 = nonlinear_function.function_Ga(sol_init_Newton_reduced, sol_init_reduced, number_cells, number_species,
                                             Dx, Dt);         
    
#    funMat2 = nonlinear_function.function_Gb(sol_init_Newton, number_species, number_cells,
#                                             number_internal_edges, Dx, Dt, a_star, T_edge);
    
    funMat2 = nonlinear_function.function_Gb(sol_init_Newton_reduced, number_species, number_cells,
                                                 number_internal_edges, index_internal_edges, Dx,
                                                 Dt, a_star, T_edge);    
        
    ## Compute function Gc with an assembling
    funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
    
    ## Compare the two constructions of Gc
    #print(funMat3 - funMat3_bis)
    ## Compute function Gd
    #funMat4 = nonlinear_function.function_Gd(number_species, number_cells, edge_solution, nonlinear_function.Matrix_Laplacian_flux(sol_init_Newton, number_species, number_cells, number_internal_edges, Dx, Dt, T_edge), Mat_coeff_astar, index_different_specie);
    ## Compute function Gd with an assembling
    funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge)
    ## Compare the two constructions of Gc
    #print(funMat4 - funMat4_bis)
    
    funMat = funMat1 - funMat2 - funMat3 + funMat4;
    
    funMat_reduced = np.dot(np.transpose(reduced_basis), funMat);
        
    ## Compute Jacobian Ga
    Amat1 = Jacobian.Jacfunction_Ga(number_species, number_cells, Dx, Dt);
    
    Amat2 = Jacobian.Jacfunction_Gb_bis(number_cells, number_species, dist_center_cells, a_star, index_internal_edges, T_edge);
    
    Amat3 = Jacobian.Assembling_JacGc(number_species, number_cells, sol_init_Newton_reduced, edge_solution_reduced, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge); 
    
    Amat4 = Jacobian.Assembling_JacGd(number_cells, number_species, sol_init_Newton_reduced, edge_solution_reduced, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge);
    
    Amat = Amat1 - Amat2 - Amat3 + Amat4;

    ## Chain rule
    Amat_reduced = np.dot(np.transpose(reduced_basis), np.dot(Amat, reduced_basis));           
    
    Fmat_reduced = np.dot(Amat_reduced, newton_coeff_reduced) - funMat_reduced;
    
    init_residual = Fmat_reduced - np.dot(Amat_reduced, newton_coeff_reduced);
    #norm_init = LA.norm(init_residual, 2);
    
    while error_Newton > tol_Newton:
        
        counter = counter + 1;
       
        funMat1 = nonlinear_function.function_Ga(sol_init_Newton_reduced, sol_init_reduced, number_cells, number_species, Dx, Dt);         
        
        funMat2 = nonlinear_function.function_Gb(sol_init_Newton_reduced, number_species, number_cells, 
                                                     number_internal_edges, index_internal_edges, Dx,
                                                     Dt, a_star, T_edge);
        
        funMat3 = nonlinear_function.function_Gc_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);
                
        funMat4 = nonlinear_function.function_Gd_bis(sol_init_Newton_reduced, number_cells, index_internal_edges, number_species, edge_solution_reduced, Mat_coeff_astar, T_edge);

        funMat = funMat1 - funMat2 - funMat3 + funMat4;
       
        ## Compute the reduced nonlinear function
        
        funMat_reduced = np.dot(np.transpose(reduced_basis), funMat);
        
        #Amat1 = Jacobian.Jacfunction_Ga(number_species, number_cells, Dx, Dt);

        #Amat2 = Jacobian.Jacfunction_Gb_bis(number_cells, number_species, dist_center_cells, a_star, index_internal_edges, T_edge); 

        Amat3 = Jacobian.Assembling_JacGc(number_species, number_cells, sol_init_Newton_reduced, edge_solution_reduced, number_internal_edges, index_internal_edges, Dx, Dt, Mat_coeff_astar, T_edge); 

        Amat4 = Jacobian.Assembling_JacGd(number_cells, number_species, sol_init_Newton_reduced, edge_solution_reduced, Mat_coeff_astar, index_internal_edges, number_internal_edges, Dx, Dt, T_edge);

        Amat = Amat1 - Amat2 - Amat3 + Amat4; 

        ## Chain rule to compute the reduced Jacobian
        Amat_reduced = np.dot(np.transpose(reduced_basis), np.dot(Amat, reduced_basis));           
    
        Fmat_reduced = np.dot(Amat_reduced, newton_coeff_reduced) - funMat_reduced;          
        ## Conditionning of the Jacobian matrix
        #print(np.linalg.cond(Amat,np.inf))
        
        ## Preconditionning
        Pcond = np.diag(np.diag(Amat_reduced));
        inv_Pcond = LA.inv(Pcond);    
        
        ## GMRES
        Sol_newton_coeff_reduced, exit_code = gmres(np.dot(inv_Pcond, Amat_reduced), np.dot(inv_Pcond, Fmat_reduced));

        #Sol_newton_coeff_reduced = LA.solve(Amat_reduced, Fmat_reduced);  
        
        
        Sol_newton_reduced = np.sum(Sol_newton_coeff_reduced * reduced_basis, 1);
        ## evaluate L2 norm of ||G^n(U^{k,n})||
        #residual = Fmat - np.dot(Amat, Sol_newton);
        #residual_norm = LA.norm(residual, 2);
        
         ## evaluate Linf norm
        plot_error_Newton[counter] = max(abs(Sol_newton_coeff_reduced - newton_coeff_reduced));
        error_Newton = np.copy(plot_error_Newton[counter]);
        
        if counter == Nmax_Newton - 1:
            break;
        
        newton_coeff_reduced = np.copy(Sol_newton_coeff_reduced);
        sol_init_Newton_reduced = np.copy(Sol_newton_reduced);
   
       
    print("Newton cv in {} iterations at time step {} and error is {}".format(counter, nn, error_Newton))


    return (newton_coeff_reduced, Sol_newton_reduced, counter)