#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:15:54 2021

@author: Jad Dabaghi
"""
## Finite volume scheme cross diffusion system

import numpy as np
from numpy import linalg as LA

show_plot = False;

import matplotlib as mpl

if not show_plot:
    mpl.use('Agg') 
    
import matplotlib.pyplot as plt
#from scipy.misc import imsave
#from PIL import Image

import os
import fun_sol_init_time
import edge_function
import Newton_solver
import solution_property
import check_mass_conservation
import orthogonal_projection
import my_simpson
import check_properties_fine_sol
import check_properties_reduced_sol
import reduced_model_case
import compute_safe_sol
import Colors
import compute_error

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--SVD', type = float, default = 0.1)
args = parser.parse_args()

## SVD parameters
tol_SVD = args.SVD;
print("tol SVD is equal to {}".format(tol_SVD))
suffix = "SVD_{}".format(tol_SVD)
directory1 = "image_code_fine_pb_PVD_online_{}".format(suffix)


#parent_dir1 = "/home/carnot-smiles/Postdoc_ENPC/code_python"
#my_path1 = os.path.join(parent_dir1, directory1)
#os.makedirs(directory1, exist_ok = True) 
#os.mkdir(directory1);
os.system("mkdir -p " + directory1)


directory2 = "image_code_reduced_pb_PVD_online_notsafe{}".format(suffix)
#parent_dir2 = "/home/carnot-smiles/Postdoc_ENPC/code_python/image_Code"
#my_path2 = os.path.join(parent_dir2, directory2);
#os.mkdir(directory2);
#os.makedirs(directory2, exist_ok = True)
os.system("mkdir -p " + directory2)

directory3 = "plot_image_validation_PVD_notsafe{}".format(suffix)
os.system("mkdir -p " + directory3)

#plt.show = lambda: None if not show_plot else plt.show
#plt.close = lambda: None if not close_plot else plt.close 



## Finite volume scheme: 
#G_{i,K}^n(U^n) = m_K/Dt * (u_{i,K}^n - u_{i,K}^{n-1}) + Sum_{\sigma \in EKint} F_{i,K \sigma}^n(U^n)=0
# G_{i,K}^n(U^n)  = G_{i,K}^{a,n}(U^n) - G_{i,K}^{b,n}(U^n) - G_{i,K}^{c,n}(U^n) + G_{i,K}^{d,n}(U^n)
#G_{i,K}^{a,n}(U^n) = m_K/Dt * (u_{i,K}^n - u_{i,K}^{n-1})
#G_{i,K}^{b,n}(U^n) = Sum_{ϭ \in EKint} a* T_{ϭ} D_{K,ϭ} u_{i}^n
#G_{i,K}^{d,n}(U^n) = Sum_{ϭ \in EKint} (\sum_{j=1}^N (aij - a*) u_{iϭ}^n D_{K,ϭ} u_{j}^n)

###############################################################################
############## CONTINUOUS OR DISCONTINUOUS CASE AND MODEL######################
###############################################################################
continuous_case = 1;                 ##----->  discontinuous case = 0
choose_reduced_model = 1;            ##----->  not safe model = 1
PVD_process = 1;                     ##----->  4 species = 1
number_species = 4;
compute_snapshots_offline = 0;
compute_snapshots_online = 0;
compute_reduced_sol = 1;
load_snapshots_offline = 1;
load_snapshots_online = 1;
load_full_snapshots = 1;
load_reduced_sol_offline = 1;
load_reduced_sol_online = 0;
save_reduced_sol = 1;
online_stage = 1;
SVD_decomposition = 1;
validation_stage = 1;
tol_eps = 1e-6;
print('the number of species is {}'.format(number_species))        
number_parameter_mu = 20;
number_parameter_mu_online = 20;
## Online stage ---->1 otherwise 0
tol_astar = 0.5;
 
###############################################################################
########################### SPATIAL DOMAIN ####################################
###############################################################################

L = 1.0;
domain = [0, L];
number_cells = 100;
Dx = L/number_cells;
space_mesh = np.linspace(0, L, number_cells + 1);
center_space_mesh = np.zeros(number_cells);
number_internal_edges = number_cells - 1;
index_internal_edges = np.arange(1, number_cells);
edge_measure = 1.0;

for ii in range(0, number_cells):
    center_space_mesh[ii] =  0.5 * (space_mesh[ii] + space_mesh[ii + 1]);

## distance between center cells    
dist_center_cells = abs(center_space_mesh[1] - center_space_mesh[0]);
T_edge = edge_measure/dist_center_cells;

###############################################################################
########################### TIME DISCRETIZATION ###############################
###############################################################################

Tf = 0.5;
Time_interval = [0, Tf];
DT = Tf/100.0;
Dt = DT/10.0;
number_time_step = Tf/Dt;
index_time_step = np.arange(0, number_time_step + 1, dtype = int);
time_mesh = np.linspace(0, Tf, number_time_step + 1);
time_observable = np.arange(0, Tf + DT, DT, dtype = float);
indices_coarse_grid = index_time_step[0 : len(index_time_step) : int(DT/Dt)];

# Initial solution n=0
Sol_init = np.zeros(number_cells * number_species);
Sol_init_reshape = np.zeros((number_cells, number_species));
if continuous_case == 1:
    
    if PVD_process == 1:
        
        for ii in range(0, number_species):
            if ii == 0:
                Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp1_PVD(center_space_mesh, tol_eps);       
            elif ii == 1:
                Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp2_PVD(center_space_mesh, tol_eps);
            elif ii == 2:
                Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp3_PVD(center_space_mesh, tol_eps);
            else :
                Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp4_PVD(center_space_mesh, tol_eps);
    
    else:
        
        for ii in range(0, number_species):
            if ii == 0:
                Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp1(center_space_mesh, tol_eps);       
            elif ii == 1:
                Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp2(center_space_mesh, tol_eps);
            else:
                Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp3(center_space_mesh, tol_eps);
else:
    
    for ii in range(0, number_species):
        if ii == 0:
            Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp1_disc(center_space_mesh, tol_eps, L);       
        elif ii == 1:
            Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp2_disc(center_space_mesh, tol_eps, L);
        else:
            Sol_init[ii * number_cells : (ii + 1) * number_cells] = fun_sol_init_time.sol_init_time_sp3_disc(center_space_mesh, tol_eps, L);


## Renormalizing the initial conditions

TOTO_init = np.zeros(number_species * number_cells);
for ii in range(0, number_species):
    Sol_init_reshape[:, ii] = Sol_init[ii * number_cells : (ii + 1) * number_cells];

sum_init_all_cells = np.sum(Sol_init_reshape, axis = 1);

for ii in range(0, number_species):    
    TOTO_init[ii * number_cells : (ii + 1) * number_cells] = (Sol_init_reshape[:, ii])/sum_init_all_cells;


## plot init solution u1
plt.plot(center_space_mesh, TOTO_init[0 : number_cells], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label='t=0')
plt.xlabel('abcissa', fontsize = 14)
plt.ylabel('Solution u1', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
#plt.axis('equal')
#plt.xlim(0, L)
plt.legend(fontsize = 13)      
plt.tight_layout()
file_name = "Sol_init_u1_T0.pdf"
#plt.savefig(os.path.join(my_path1, file_name))   
plt.savefig(os.path.join(directory1, file_name))
if show_plot:
    plt.show()
plt.close()

## plot init solution u2
plt.plot(center_space_mesh, TOTO_init[number_cells : 2 * number_cells], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label='t=0')
plt.xlabel('abcissa', fontsize = 14)
plt.ylabel('Solution u2', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
#plt.axis('equal')
#plt.xlim(0, L)
plt.legend(fontsize = 13)      
plt.tight_layout()
file_name = "Sol_init_u2_T0.pdf"
plt.savefig(os.path.join(directory1, file_name))
if show_plot:
    plt.show()
plt.close()

## plot init solution u3
plt.plot(center_space_mesh, TOTO_init[2 * number_cells : 3 * number_cells], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label='t=0')
plt.xlabel('abcissa', fontsize = 14)
plt.ylabel('Solution u3', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
#plt.axis('equal')
#plt.xlim(0, L)
plt.legend(fontsize = 13)      
plt.tight_layout()
file_name = "Sol_init_u3_T0.pdf"
plt.savefig(os.path.join(directory1, file_name))
if show_plot:
    plt.show()
plt.close()

## Plot the fourth specie for PVD process

if PVD_process == 1:
   
    plt.plot(center_space_mesh, TOTO_init[3 * number_cells : 4 * number_cells], marker = 's', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label='t=0')
    plt.xlabel('abcissa', fontsize = 14)
    plt.ylabel('Solution u4', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    #plt.axis('equal')
    #plt.xlim(0, L)
    plt.legend(fontsize = 13)      
    plt.tight_layout()
    file_name = "Sol_init_u4_T0.pdf"
    plt.savefig(os.path.join(directory1, file_name))
    if show_plot:
        plt.show()
    plt.close()
    


###############################################################################
############### CROSS DIFF MATRICES############################################
###############################################################################

    

#Mat_coeff_all = np.zeros((number_parameter_mu, number_species, number_species));
#
#for ll in range(0, number_parameter_mu_online):
#    temp_Mat = np.zeros((number_species, number_species));
#    generate_numbers = np.random.uniform(0, 1, int(0.5 * (number_species * (number_species - 1))));
#    ## Fill the cross-diff matrix. It contains N*(N-1)/2 elements as the matrix is zero on diagonal and is symmetric
#    for ii in range(0, number_species - 1):
#        for jj in range(ii + 1, number_species):
#            temp_Mat[ii, jj] = np.random.uniform(0, 1);
#    
#    temp_Mat_transpose = np.transpose(temp_Mat);
#    Mat_coeff_all[ll, :, :] = temp_Mat + temp_Mat_transpose;

###############################################################################
############### CROSS DIFF MATRICES ONLINE STAGE ##############################
###############################################################################

#Mat_coeff_all_online = np.zeros((number_parameter_mu_online, number_species, number_species));
#
#for ll in range(0, number_parameter_mu_online):
#    temp_Mat = np.zeros((number_species, number_species));
#    generate_numbers = np.random.uniform(0, 1, int(0.5 * (number_species * (number_species - 1))));
#    ## Fill the cross-diff matrix. It contains N*(N-1)/2 elements as the matrix is zero on diagonal and is symmetric
#    for ii in range(0, number_species - 1):
#        for jj in range(ii + 1, number_species):
#            temp_Mat[ii, jj] = np.random.uniform(0, 1);
#    
#    temp_Mat_transpose = np.transpose(temp_Mat);
#    Mat_coeff_all_online[ll, :, :] = temp_Mat + temp_Mat_transpose;


###############################################################################
############### SAVE THE CROSS DIFF MATRICES ##################################
###############################################################################

#Mat_reshaped =  Mat_coeff_all.reshape(Mat_coeff_all.shape[0], -1);   

#if PVD_process == 1:
#    np.savetxt("cross_diff_Mat_PVD.txt", Mat_reshaped)
#else:
#    np.savetxt("cross_diff_Mat.txt", Mat_reshaped)


###############################################################################
############### SAVE THE CROSS DIFF MATRICES ONLINE STAGE #####################
###############################################################################

#Mat_reshaped_online = Mat_coeff_all_online.reshape(Mat_coeff_all_online.shape[0], -1);
#
#if PVD_process == 1:
#    np.savetxt("cross_diff_Mat_PVD_online.txt", Mat_reshaped_online)
#else:
#    np.savetxt("cross_diff_Mat_online.txt", Mat_reshaped_online)


        
##########################################################
############### LOAD THE CROSS DIFF MATRIX ###############
##########################################################

if PVD_process == 1:
    loaded_Mat = np.loadtxt("cross_diff_Mat_PVD.txt")
else:
    loaded_Mat = np.loadtxt("cross_diff_Mat.txt") 

Mat_coeff_all = loaded_Mat.reshape(loaded_Mat.shape[0], loaded_Mat.shape[1] // number_species, number_species)  

###################################################################
############### LOAD THE CROSS DIFF MATRICES ONLINE ###############
###################################################################

if PVD_process == 1:
    loaded_Mat_online = np.loadtxt("cross_diff_Mat_PVD_online.txt")
else:
    loaded_Mat_online = np.loadtxt("cross_diff_Mat_online.txt") 

   
Mat_coeff_all_online = loaded_Mat_online.reshape(loaded_Mat_online.shape[0], loaded_Mat_online.shape[1] // number_species, number_species)  


## begining of the script. Loop on all parameters. For snapsohots solution we save only the solutions at time observable
snapshots_sol = np.zeros((number_species * number_cells, number_parameter_mu * len(time_observable[1 : ])));
snapshots_sol_online = np.zeros((number_species * number_cells, number_parameter_mu_online * len(time_observable[1 : ])));
full_snapshots_sol = np.zeros((number_species * number_cells, (number_parameter_mu_online + number_parameter_mu) * len(time_observable[1 : ])));


index_different_specie = np.array([[1, 2], [0, 2], [0, 1]]);  
mass_init = np.zeros((number_species, number_cells));
mass_init_domain = np.zeros(number_species);
concentration_species_inf_domain = np.zeros(number_species);
mass_init_reduced = np.zeros((number_species, number_cells));
mass_init_domain_reduced = np.zeros(number_species);
mass_inf_domain_reduced = np.zeros(number_species);
mass_conservation = np.zeros((len(index_time_step), number_species));
concentration_species_inf_domain_reduced = np.zeros(number_species);
number_Newton_iter_param_mu = np.zeros(number_parameter_mu);
number_Newton_iter_reduced_param_mu = np.zeros(number_parameter_mu);

 

## Mass at init time and at infinite time (it can be computed because the solution reach a constant profile) 
        
for ii in range(0, number_species):
    mass_init[ii, :] = Dx * TOTO_init[ii * number_cells : (ii + 1) * number_cells];
#   mass_init[ii, :] = my_simpson.my_simpson(center_space_mesh, Sol_init[ii * number_cells : (ii + 1) * number_cells], number_cells);             
    mass_init_domain[ii] = np.sum(mass_init[ii, :]);
    concentration_species_inf_domain[ii] = (mass_init_domain[ii])/(number_cells * Dx);         
    
    
## plot init mass
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
#ax1.plot(center_space_mesh, mass_init[0, :], 'r-o', linewidth = 1)
#ax1.set_xlabel('abcissa')
#ax1.set_ylabel('mass of u1 at time T=0')
#ax2.plot(center_space_mesh, mass_init[1, :], 'b-^', linewidth = 1)
#ax2.set_xlabel('abcissa')
#ax2.set_ylabel('mass of u2 at time T=0')
#ax3.plot(center_space_mesh, mass_init[2, :], 'g-s', linewidth = 1)
#ax3.set_xlabel('abcissa')
#ax3.set_ylabel('mass of u3 at time T=0')
#file_name = "fine_pb_mass_init_T " + str(0)
#plt.savefig(file_name)
#plt.show()
#plt.close()
#fig.savefig(os.path.join(my_path1, file_name))        
#fig.clear()



####################################################################################
################### DEFINITION OF a* FOR HIGH FIDELITY SOLUTIONS   #############################    
####################################################################################

struct_a_star = np.zeros(number_parameter_mu);

for ll in range(0, number_parameter_mu):
    Mat_coeff = np.copy(Mat_coeff_all[ll, :, :]);
    temp = np.reshape(Mat_coeff, Mat_coeff.shape[0] * Mat_coeff.shape[1], 0);
    temp_pos = temp[temp > 0];

    ## definition of a*
    struct_a_star[ll] = min(max(temp_pos), max(min(temp_pos), tol_astar * (Dx**2)/Dt));    
        
####################################################################################
################### Compute the high fidelity solutions OFFLINE STAGE   #############################    
####################################################################################

if compute_snapshots_offline == 1:    
    for ll in range(0, number_parameter_mu):
        
        print("Simulation with parameter mu equal to {}".format(ll))
        ## Init newton solution
        Sol_newton = np.copy(TOTO_init);
        Sol_init = np.copy(TOTO_init);
        entropy = np.zeros(len(index_time_step));
        Solution = np.zeros((number_cells * number_species, len(index_time_step)));
        entropy[0] = solution_property.entropy_property(Sol_newton, Dx, number_cells, number_species);
        entropy_inf = number_cells * Dx * np.dot(concentration_species_inf_domain, np.log(concentration_species_inf_domain));
        Solution[:, 0] = np.copy(TOTO_init);
        mass_conservation[0, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution, 0, mass_init_domain, Dx);  
        number_newton_iter = np.zeros(len(index_time_step));
        
        ## cross-diffusion matrix
        Mat_coeff = np.copy(Mat_coeff_all[ll, :, :]);
       
        
        ## definition of a*
        a_star = np.copy(struct_a_star[ll]);
        
        ## definition of the matrix giving A - astar    
        Mat_coeff_astar = Mat_coeff - a_star;
            
        for nn in index_time_step[1 :]:       
        
            ## computation of the edge function u_{j,ϭ}^n 
            edge_solution = np.zeros((number_cells * number_species, 2));
            
            for ii in range(0, number_species):
                edge_solution[ii * number_cells : (ii + 1) * number_cells, :] = edge_function.edge_solution(Sol_newton[ii * number_cells : (ii + 1) * number_cells], number_cells, number_internal_edges);  
        
            
            
            ### Newton solver
            
            Solution[:, nn], number_newton_iter[nn] = Newton_solver.Newton_solver(Sol_newton, Sol_init, edge_solution, number_cells, number_internal_edges,
                                                   index_internal_edges, number_species, Dx, Dt, a_star, Mat_coeff_astar,
                                                   index_different_specie, dist_center_cells, T_edge, nn);
            
            
            ## Update
            Sol_init = np.copy(Solution[:, nn]);
            Sol_newton = np.copy(Sol_init);
            
                        
            ## Entropy property
            entropy[nn] = solution_property.entropy_property(Solution[:, nn], Dx, number_cells, number_species);
        
            ## check sum concentration = 1 
            check_sum_concentrations = 0;
            for ii in range(0, number_species):
                check_sum_concentrations = np.copy(check_sum_concentrations) +  Solution[ii * number_cells : (ii + 1) * number_cells, nn];
                
            average_check_sum_concentrations = sum(check_sum_concentrations)/number_cells;
            
            if abs(average_check_sum_concentrations - 1) > 1e-2:
                print('error: the sum of concentrations of species should be around 1')
                    
            ## check mass conservation
            mass_conservation[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution, nn, mass_init_domain, Dx);
    
    #    ## Compute E(T--->+inf) - E(T) in log log scale
    #    plt.loglog(index_time_step, abs(entropy - entropy[-1]), marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Entropy computation")
    #    plt.loglog(index_time_step, abs(entropy - entropy_inf), marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Entropy computation with constant profile")
    #    plt.xlabel('time', fontsize = 14)  
    #    plt.ylabel(r'$|E_{\mathcal{T}}(T \rightarrow +\infty) - E_{\mathcal{T}}(t)|$', fontsize = 14)
    #    plt.xticks(fontsize = 14)
    #    plt.yticks(fontsize = 14)
    #    plt.legend(fontsize = 13)    
    #    plt.tight_layout()
    #    file_name = "Entropy_inf_mu {}.pdf".format(ll)
    #    plt.savefig(os.path.join(directory1, file_name))
    #    if show_plot:
    #        plt.show()
    #    plt.close()
    
        ## Compute matrix of snapshots
        
        snapshots_sol[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])] = Solution[:, indices_coarse_grid[1 : ]];

    ## total Number of Newton iterations per parameter mu
    
    #number_Newton_iter_param_mu[ll] = np.sum(number_newton_iter[1 :]);
#    
#                                                                                      
###############################################################################
############## END OFFLINE RESOLUTION #########################################
###############################################################################                                                                                     
#                                                                                      
#                                                                                      
###############################################################################
######################## SAVE SNAPSHOTS MATRIX ################################
###############################################################################
##if PVD_process == 1:
##    np.savetxt("Snapshots_Mat_PVD.txt", snapshots_sol)     
##else:
##    np.savetxt("Snapshots_Mat.txt", snapshots_sol)
#
#
#
#
#
###############################################################################
#####################LOAD SNAPSHOTS MATRIX OFFLINE ############################
###############################################################################
if load_snapshots_offline == 1:
    if PVD_process == 1:
        snapshots_sol = np.loadtxt("Snapshots_Mat_PVD.txt");
    else:    
        snapshots_sol = np.loadtxt("Snapshots_Mat.txt");   
    
    
###############################################################################
############# PLOT ALL PROPERTIES OF FINE SOLUTION#############################
#################### FROM SNAPSHOTS ###########################################
###############################################################################
#
#### Plot mass conservation : WARNING MASS IS CONSIDERED AT TIME OBSERVABLE
#mass_conservation = np.zeros((len(time_observable[1 :]), number_species));
#
#linf_error_time_mass = np.zeros(number_parameter_mu);
#for ll in range(0, number_parameter_mu):
#    extract_snapshots_mu = snapshots_sol[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
#    for nn in range(0, len(time_observable[1 :])):
#        mass_conservation[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, extract_snapshots_mu, nn, mass_init_domain, Dx);
#    linf_error_time_mass[ll] = np.max(np.max(abs(mass_conservation - np.matlib.repmat(mass_init_domain, len(time_observable[1 :]), 1)), 0));             
#
### plot mass conservation 
#plt.plot(np.arange(0, number_parameter_mu), linf_error_time_mass, marker = 'o', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "mass deviation from initial mass")
#plt.xlabel(r' parameter $\mu$', fontsize = 14)
#plt.ylabel(r' mass', fontsize = 14)
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)        
#plt.xlim(0, number_parameter_mu)
#plt.legend(fontsize = 13)
#plt.tight_layout()
#file_name = "mass_conservation_fine_pb.pdf"
#plt.savefig(os.path.join(directory1, file_name))   
#if show_plot:
#    plt.show()
#plt.close()
#
### Plot entropy
#for ll in range(0, number_parameter_mu):
#    extract_snapshots_mu = snapshots_sol[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
#    entropy = np.zeros(len(time_observable[1 :]));
#    for nn in range(0, len(time_observable[1 :])):        
#        
#        entropy[nn] = solution_property.entropy_property(extract_snapshots_mu[:, nn], Dx, number_cells, number_species);
##            
#    plt.plot(indices_coarse_grid[1 :], entropy, marker = 's', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'fine problem $\mu$ = {0}'.format(ll))
#    plt.xlabel('time step', fontsize = 14)
#    plt.ylabel('Entropy', fontsize = 14)
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)
#    plt.tight_layout()
#    file_name = "fine_pb_entropy_mu_{}.pdf".format(ll)
#    plt.savefig(os.path.join(directory1, file_name))
#    if show_plot:
#        plt.show()
#    plt.close()   
#        
#
#### Check that solution is always positive
#min_x_min_mu_sol = check_properties_fine_sol.check_positivity(snapshots_sol, number_species, number_cells, time_observable,
#                     number_parameter_mu, directory1, show_plot);
#
#plt.plot(indices_coarse_grid[1 :], min_x_min_mu_sol, marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'positivity')
#plt.xlabel('time step', fontsize = 14)
#plt.ylabel('Solution', fontsize = 14)
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)
#plt.legend(fontsize = 13)
#plt.tight_layout()
#file_name = "check_fine_pb_positivity_sol.pdf"
#plt.savefig(os.path.join(directory1, file_name))        
#if show_plot:
#    plt.show()
#plt.close() 
#                                                          
#                                                             
#### Check that solution is always below than 1
#max_x_max_mu_sol = check_properties_fine_sol.check_below_one(snapshots_sol, number_species, number_cells, time_observable,
#                     number_parameter_mu, directory1, show_plot);
#
#plt.plot(indices_coarse_grid[1 :], max_x_max_mu_sol, marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'Solution max')
#plt.xlabel('time step', fontsize = 14)
#plt.ylabel('Solution', fontsize = 14)
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)
#plt.legend(fontsize = 13)
#plt.tight_layout()
#file_name = "check_fine_pb_sol_below_1.pdf"
#plt.savefig(os.path.join(directory1, file_name))        
#if show_plot:
#    plt.show()
#plt.close()                                                            
#
#### Check that sum species is always equal to 1
#min_x_min_mu_reshape_sum_snap = check_properties_fine_sol.check_sum_species_equal_one(number_cells, number_species, number_parameter_mu,
#                               snapshots_sol, time_observable, directory1, show_plot);
#
#plt.plot(indices_coarse_grid[1 :], min_x_min_mu_reshape_sum_snap, marker = 'o', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = r'Volume filling constraint')
#plt.xlabel('time step', fontsize = 14)
#plt.ylabel('Solution', fontsize = 14)
#plt.xticks(fontsize = 14)
#plt.yticks(fontsize = 14)
#plt.legend(fontsize = 13)
#plt.tight_layout()
#file_name = "check_fine_pb_sum_sol_equal_1.pdf"
#plt.savefig(os.path.join(directory1, file_name))        
#if show_plot:
#    plt.show()
#plt.close()
#
#
#### PLOT SNAPSHOTS IN SAME GRAPH
#for ll in range(0, number_parameter_mu):
#    extract_snapshots_mu = snapshots_sol[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
#    ## plot solution at time observable
#    for nn in range(0, len(time_observable[1 :])):        
#        
#        ## plot solution
#        if PVD_process == 1:
#            plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[3 * number_cells : 4 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "Solution u4 at n = {}".format(indices_coarse_grid[nn + 1]))
#    
#        else:
#            plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.xlabel('abcissa', fontsize = 14)
#        
#        plt.ylabel('Solution', fontsize = 14)
#        plt.xticks(fontsize = 14)
#        plt.yticks(fontsize = 14)
#        plt.axis('equal')
#        plt.xlim(0, L)
#        plt.legend(fontsize = 13)
#        plt.tight_layout()
#        file_name = "Solution_at_time_{}_and_mu_{}.pdf".format(indices_coarse_grid[nn + 1], ll)
#        plt.savefig(os.path.join(directory1, file_name))   
#        if show_plot:
#            plt.show()
#        plt.close()
#       
    
    
########################################################################################
#####################COMPUTE HIGH FIDELITY SOLUTIONS IN ONLINE STAGE####################
########################################################################################

if compute_snapshots_online == 1:

    ## Create a structure for astar for the high fidelity solutions    
    struct_a_star = np.zeros(number_parameter_mu_online);
    for ll in range(0, number_parameter_mu_online):
        Mat_coeff = np.copy(Mat_coeff_all_online[ll, :, :]);
        temp = np.reshape(Mat_coeff, Mat_coeff.shape[0] * Mat_coeff.shape[1], 0);
        temp_pos = temp[temp > 0];
    
        ## definition of a*
        struct_a_star[ll] = min(max(temp_pos), max(min(temp_pos), tol_astar * (Dx**2)/Dt));
        
    
    ## Compute the high fidelity solutions    
    
    for ll in range(0, number_parameter_mu_online):
        
        print("Simulation for online fine resolution with parameter mu equal to {}".format(ll))
        ## Init newton solution
        Sol_newton = np.copy(TOTO_init);
        Sol_init = np.copy(TOTO_init);
        entropy = np.zeros(len(index_time_step));
        Solution = np.zeros((number_cells * number_species, len(index_time_step)));
        entropy[0] = solution_property.entropy_property(Sol_newton, Dx, number_cells, number_species);
        entropy_inf = number_cells * Dx * np.dot(concentration_species_inf_domain, np.log(concentration_species_inf_domain));
        Solution[:, 0] = np.copy(TOTO_init);
        mass_conservation[0, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution, 0, mass_init_domain, Dx);  
        number_newton_iter = np.zeros(len(index_time_step));
        
        ## cross-diffusion matrix
        Mat_coeff = np.copy(Mat_coeff_all_online[ll, :, :]);
       
        ## definition of a*
        a_star = np.copy(struct_a_star[ll]);
        
        ## definition of the matrix giving A - astar    
        Mat_coeff_astar = Mat_coeff - a_star;
            
        for nn in index_time_step[1 :]:       
        
            ## computation of the edge function u_{j,ϭ}^n 
            edge_solution = np.zeros((number_cells * number_species, 2));
            
            for ii in range(0, number_species):
                edge_solution[ii * number_cells : (ii + 1) * number_cells, :] = edge_function.edge_solution(Sol_newton[ii * number_cells : (ii + 1) * number_cells], number_cells, number_internal_edges);  
        
            
            
            ### Newton solver
            
            Solution[:, nn], number_newton_iter[nn] = Newton_solver.Newton_solver(Sol_newton, Sol_init, edge_solution, number_cells, number_internal_edges,
                                                   index_internal_edges, number_species, Dx, Dt, a_star, Mat_coeff_astar,
                                                   index_different_specie, dist_center_cells, T_edge, nn);
            
            
            ## Update
            Sol_init = np.copy(Solution[:, nn]);
            Sol_newton = np.copy(Sol_init);
            
                        
            ## Entropy property
            entropy[nn] = solution_property.entropy_property(Solution[:, nn], Dx, number_cells, number_species);
        
            ## check sum concentration = 1 
            check_sum_concentrations = 0;
            for ii in range(0, number_species):
                check_sum_concentrations = np.copy(check_sum_concentrations) +  Solution[ii * number_cells : (ii + 1) * number_cells, nn];
                
            average_check_sum_concentrations = sum(check_sum_concentrations)/number_cells;
            
            if abs(average_check_sum_concentrations - 1) > 1e-2:
                print('error: the sum of concentrations of species should be around 1')
                    
            ## check mass conservation
            mass_conservation[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution, nn, mass_init_domain, Dx);
    
    #    ## Compute E(T--->+inf) - E(T) in log log scale
    #    plt.loglog(index_time_step, abs(entropy - entropy[-1]), marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Entropy computation")
    #    plt.loglog(index_time_step, abs(entropy - entropy_inf), marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Entropy computation with constant profile")
    #    plt.xlabel('time', fontsize = 14)  
    #    plt.ylabel(r'$|E_{\mathcal{T}}(T \rightarrow +\infty) - E_{\mathcal{T}}(t)|$', fontsize = 14)
    #    plt.xticks(fontsize = 14)
    #    plt.yticks(fontsize = 14)
    #    plt.legend(fontsize = 13)    
    #    plt.tight_layout()
    #    file_name = "Entropy_inf_mu {}.pdf".format(ll)
    #    plt.savefig(os.path.join(directory1, file_name))
    #    if show_plot:
    #        plt.show()
    #    plt.close()
    
        ## Compute matrix of snapshots
        
        snapshots_sol_online[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])] = Solution[:, indices_coarse_grid[1 : ]];

####################################################
############## END ONLINE FINE RESOLUTION ###############
####################################################                                                                                      
#                                                                                      
#                                                                                      
####################################################
############## SAVE SNAPSHOTS ONLINE MATRIX ###############
####################################################
#if PVD_process == 1:
#    np.savetxt("Snapshots_Mat_PVD_online.txt", snapshots_sol_online)     
#else:
#    np.savetxt("Snapshots_Mat_online.txt", snapshots_sol_online)


#########################################################
############ LOAD SNAPSHOTS MATRIX ONLINE ###############
#########################################################

if load_snapshots_online == 1:
    if PVD_process == 1:
        snapshots_sol_online = np.loadtxt("Snapshots_Mat_PVD_online.txt");
    else:    
        snapshots_sol_online = np.loadtxt("Snapshots_Mat_online.txt");
    
    
####################    ###########################################################
############ PLOT SNAPSHOTS MATRIX ONLINE #####################################
###############################################################################
    
#for ll in range(0, number_parameter_mu_online):
#    extract_snapshots_mu = snapshots_sol_online[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
#    ## plot solution at time observable
#    for nn in range(0, len(time_observable[1 :])):        
#        
#        ## plot solution
#        if PVD_process == 1:
#            plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[3 * number_cells : 4 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "Solution u4 at n = {}".format(indices_coarse_grid[nn + 1]))
#    
#        else:
#            plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
#            plt.xlabel('abcissa', fontsize = 14)
#        
#        plt.ylabel('Solution', fontsize = 14)
#        plt.xticks(fontsize = 14)
#        plt.yticks(fontsize = 14)
#        plt.axis('equal')
#        plt.xlim(0, L)
#        plt.legend(fontsize = 13)
#        plt.tight_layout()
#        file_name = "Solution_at_time_{}_and_mu_{}_online.pdf".format(indices_coarse_grid[nn + 1], ll + number_parameter_mu_online)
#        plt.savefig(os.path.join(directory1, file_name))   
#        if show_plot:
#            plt.show()
#        plt.close()    
    

#################################################################
############    FULL SNAPSHOTS SOLUTION ##############################
################################################################# 
    
full_snapshots_sol = np.concatenate((snapshots_sol, snapshots_sol_online), axis = 1);
    
###################################################################
############## SAVE FULL SNAPSHOTS MATRIX #########################
###################################################################

#if PVD_process == 1:
#    np.savetxt("Full_Snapshots_Mat_PVD.txt", full_snapshots_sol)     
#else:
#    np.savetxt("Full_Snapshots_Mat.txt", full_snapshots_sol)
    
        
###############################################################################
###############################################################################
###############################################################################


###############################################################################
#########################  POD REDUCTION MODEL ################################
###############################################################################  


###############################################################################
###############################################################################
###############################################################################

###############################################################################
#################### SINGULAR VALUE DECOMPOSITION #############################
###############################################################################
#

if SVD_decomposition == 1:

    if choose_reduced_model == 2:
        ## Compute the log of snapshots solutions
        V_log, S_log, WT_log = LA.svd(np.log(snapshots_sol), full_matrices = False);   
        # create N*Ne, Nt * mu Sigma matrix
    #    Sigma_log = np.zeros((number_cells * number_species, len(index_time_step[1 :]) * number_parameter_mu));
    #    # populate Sigma with min(N*Ne, Nt * mu) diagonal matrix
    #    for ii in range(0, min(number_cells * number_species, len(index_time_step[1 :]) * number_parameter_mu)):
    #        Sigma_log[ii, ii] = S_log[ii];
        
        ## plot log singular values
    #    plt.plot(range(0, min(number_cells * number_species, len(index_time_step[1 :]) * number_parameter_mu)), S_log, 'b-^', linewidth = 1)
    #    plt.xlabel('indices')
    #    plt.ylabel('log singular values')
    #    plt.tight_layout()
    #    file_name = "log_singular_values" 
    #    plt.savefig(os.path.join(directory2, file_name))        
    #    if show_plot:
    #        plt.show()
    #    plt.close()
        
        Sigma_log = S_log;
        
        ## plot log singular values
        plt.plot(range(0, number_cells * number_species), Sigma_log, 'b-^', linewidth = 1)
        plt.xlabel('indices')
        plt.ylabel('log singular values')
        plt.tight_layout()
        file_name = "log_singular_values" 
        plt.savefig(os.path.join(directory2, file_name))        
        if show_plot:
            plt.show()
        plt.close()
    
        ## Safe reduced model. Add identity bloc matrix to the matrix V_log.
        Id_matrix = np.zeros((number_species * number_cells, number_species));
        for ii in range(0, number_species):
                Id_matrix[ii * number_cells : (ii + 1) * number_cells, ii] = np.ones(number_cells);
        
        #V_log = np.concatenate((np.copy(V_log), Id_matrix), axis=1);
            
        Square_S = Sigma_log**2;
    
        #Square_S = S_log**2;
        
        sum_Square_S = 0;
    
    #    for ii in range(0, min(Sigma_log.shape[0], Sigma_log.shape[1])):
    #        if sum(Square_S[ii :]) < tol_SVD:
    #            temp_dim_r = ii;
    #            print('dimension of reduced basis is {} and tolSVD is {}'.format(temp_dim_r, tol_SVD))
    #            break
            
        for ii in range(0, len(Sigma_log)):
            if sum(Square_S[ii :]) < tol_SVD:
                temp_dim_r = ii;
                print('DIMENSION OF REDUCED BASIS r = {}'.format(temp_dim_r))
                break    
            
        #### pure SVD ###
    #    dim_r = temp_dim_r + number_species;
    #    reduced_basis = np.concatenate((V_log[:, 0 : temp_dim_r], Id_matrix), axis=1);
    #    Qmat, Rmat = np.linalg.qr(reduced_basis);
    #    reduced_basis = np.copy(Qmat);
    #    #check_orthogonalization = np.max(np.dot(reduced_basis, np.transpose(reduced_basis)) - np.ones((np.max(np.shape(reduced_basis))), np.max(np.shape(reduced_basis))));
    
        ####### add initial condition to the basis from SVD decomposition #########
        
        dim_r = temp_dim_r + number_species + 1;
        reduced_basis = np.column_stack((V_log[:, 0 : temp_dim_r], Id_matrix, np.log(TOTO_init)));
        Qmat, Rmat = np.linalg.qr(reduced_basis);
        reduced_basis = np.copy(Qmat);
        
        if dim_r > min(number_species * number_cells, len(time_observable[1 :]) * number_parameter_mu):
            print('dimension of modified reduced basis {} exceeds maximum authorized {}'.format(dim_r, number_species * number_cells))        
        
        
    else:
        V, S, WT = LA.svd(snapshots_sol, full_matrices = False);
        # create N*Ne, Nt * mu Sigma matrix
    #    Sigma = np.zeros((number_cells * number_species, len(time_observable[1 :]) * number_parameter_mu));
    #    # populate Sigma with min(N*Ne, Nt * mu) diagonal matrix
    #    for ii in range(0, min(number_cells * number_species, len(index_time_step[1 :]) * number_parameter_mu)):
    #        Sigma[ii, ii] = S[ii];
    #        
    #    ## plot singular values
    #    plt.plot(range(0, min(number_cells * number_species, len(index_time_step[1 :]) * number_parameter_mu)), S, 'b-^', linewidth = 1)
    #    plt.xlabel('indices')
    #    plt.ylabel('singular values')
    #    plt.tight_layout()
    #    file_name = "singular_values" 
    #    plt.savefig(os.path.join(directory2, file_name))        
    #    if show_plot:
    #        plt.show()
    #    plt.close()
        
        Sigma = S;
        
        ## plot singular values
        plt.plot(range(0, number_cells * number_species), Sigma, 'b-^', linewidth = 1)
        plt.xlabel('indices')
        plt.ylabel('singular values')
        plt.tight_layout()
        file_name = "singular_values" 
        plt.savefig(os.path.join(directory2, file_name))        
        if show_plot:
            plt.show()
        plt.close()
        
        Square_S = Sigma**2;
        sum_Square_S = 0;
        
        for ii in range(0, len(Sigma)):
            if sum(Square_S[ii :]) < tol_SVD:
                temp_dim_r = ii;
                print('DIMENSION OF REDUCED BASIS r = {}'.format(temp_dim_r))
                break
    
    #    for ii in range(0, min(Sigma.shape[0], Sigma.shape[1])):
    #        if sum(Square_S[ii :]) < tol_SVD:
    #            dim_r = ii;
    #            print('dimension of reduced basis is {}'.format(dim_r))
    #            break
        
        
        if temp_dim_r > min(number_species * number_cells, len(time_observable[1 :]) * number_parameter_mu):
            print('dimension of modified reduced basis {} exceeds maximum authorized {}'.format(temp_dim_r, number_species * number_cells))        
        
        ## pure SVD
    #    reduced_basis = V[:, 0 : dim_r];
    #    check_orthogonalization = np.max(np.dot(reduced_basis, np.transpose(reduced_basis)) - np.ones((np.max(np.shape(reduced_basis))), np.max(np.shape(reduced_basis))));
    #    
        ## add initial condition to the basis from SVD decomposition
        reduced_basis = np.column_stack((V[:, 0 : temp_dim_r], TOTO_init));
        Qmat, Rmat = np.linalg.qr(reduced_basis);
        reduced_basis = np.copy(Qmat);
        check_orthogonalization = np.max(np.dot(reduced_basis, np.transpose(reduced_basis)) - np.ones((np.max(np.shape(reduced_basis))), np.max(np.shape(reduced_basis))));
        dim_r = temp_dim_r + 1;
    
        if check_orthogonalization > 1e-13:
            print('Warning! Reduced basis is not orthogonal!')
#
#
###############################################################################
######################### SAVE REDUCED BASIS ##################################
###############################################################################
#        
##np.savetxt("reduced_basis_Mat_{}.txt".format(suffix), reduced_basis)     
#        
###############################################################################
####################### LOAD REDUCED BASIS ####################################
###############################################################################
#    
##reduced_basis = np.loadtxt("reduced_basis_Mat.txt")
#
            

    
###############################################################################
#################### SOLVE REDUCED PB #########################################
###############################################################################  
#
#        
### All cross diffusion matrices : 3D matrix
if online_stage == 1:
    param_factor = number_parameter_mu_online;
else:
    param_factor = number_parameter_mu;        

all_reduced_sol = np.zeros((number_species * number_cells, param_factor * len(time_observable)));
   
Mat_coeff_all_reduced = np.zeros((param_factor, number_species, number_species));

if online_stage == 1:
    Mat_coeff_all_reduced = Mat_coeff_all_online;
else:
    Mat_coeff_all_reduced = Mat_coeff_all;

if compute_reduced_sol == 1:
    
    for ll in range(0, param_factor):
            
        print("Simulation of reduced model with parameter mu equal to {}".format(ll))
        ## cross-diffusion matrix
        Mat_coeff = np.copy(Mat_coeff_all_reduced[ll, :, :]);
    #    temp = np.reshape(Mat_coeff, Mat_coeff.shape[0] * Mat_coeff.shape[1], 0);
    #    temp_pos = temp[temp > 0];
    #    a_star = min(max(temp_pos), max(min(temp_pos), tol_astar * (Dx**2)/Dt));
    
        Mat_coeff_astar = Mat_coeff - struct_a_star[ll];
            
        Solution_reduced = np.zeros((number_cells * number_species, len(index_time_step)));
        whole_coeff_reduced = np.zeros((dim_r, len(index_time_step)));
        whole_zbar = np.zeros((number_cells * number_species, len(index_time_step)));
        number_newton_iter_reduced = np.zeros(len(index_time_step));
        error_mass_reduced = np.zeros((len(index_time_step), number_species));
        mass_domain_reduced_specie_i = np.zeros((len(index_time_step), number_species));
        check_sum_concentrations_reduced = np.zeros((len(index_time_step), 1));
        reshape_Solution_init = np.zeros((number_cells, number_species));
        reshape_exp_zbar_init = np.zeros((number_cells, number_species));
        entropy_reduced = np.zeros(len(index_time_step));
        
        if choose_reduced_model == 2:
            
            z_init = np.log(TOTO_init + tol_eps);
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
            ax1.plot(center_space_mesh, z_init[0 : number_cells], 'b-o', linewidth = 1)
            ax1.set_xlabel('abcissa')
            ax1.set_ylabel('z1=ln(u1) safe reduced at T0')
            ax2.plot(center_space_mesh, z_init[number_cells : 2 * number_cells], 'r-^', linewidth = 1)
            ax2.set_xlabel('abcissa')
            ax2.set_ylabel('z2=ln(u2) safe reduced at T0')
            ax3.plot(center_space_mesh, z_init[2 * number_cells : 3 * number_cells], 'g-s', linewidth = 1)
            ax3.set_xlabel('abcissa')
            ax3.set_ylabel('z3=ln(u3) safe reduced at T0')
            plt.tight_layout()
            file_name = "z_init at T0" 
            fig.savefig(os.path.join(directory2, file_name))        
            if show_plot:
                plt.show()
            plt.close()
            fig.clear()
    
            ## Compute the projection of z_init on the reduced basis. It gives \overline{z_init} which is zbar at T0
            coeff_reduced_init, zbar_init = orthogonal_projection.orthogonal_projection(reduced_basis, z_init, dim_r);
           
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
            ax1.plot(center_space_mesh, zbar_init[0 : number_cells], 'b-o', linewidth = 1)
            ax1.set_xlabel('abcissa')
            ax1.set_ylabel(r'$\overline{Z}_1^0 = \Pi(z1^0)$')
            ax2.plot(center_space_mesh, zbar_init[number_cells : 2 * number_cells], 'r-^', linewidth = 1)
            ax2.set_xlabel('abcissa')
            ax2.set_ylabel(r'$\overline{Z}_2^0 = \Pi(z2^0)$')
            ax3.plot(center_space_mesh, zbar_init[2 * number_cells : 3 * number_cells], 'g-s', linewidth = 1)
            ax3.set_xlabel('abcissa')
            ax3.set_ylabel(r'$\overline{Z}_3^0 = \Pi(z3^0)$')
            plt.tight_layout()
            file_name = "zbar_init at T0" 
            fig.savefig(os.path.join(directory2, file_name))        
            if show_plot:
                plt.show()
            plt.close()
            fig.clear()
    
            ## Compute the safe solution at T0 using exp log sum trick
            Solution_reduced[:, 0] = compute_safe_sol.compute_safe_sol_bis(zbar_init, number_species, number_cells);
            
            ## Plot reduced solution at t=0
            
            plt.plot(center_space_mesh, Solution_reduced[0 : number_cells, 0], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label='t=0')
            plt.xlabel('abcissa', fontsize = 14)
            plt.ylabel('Reduced solution u1', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)      
            plt.tight_layout()
            file_name = "Sol_reduced_init_u1_T0.pdf"
            plt.savefig(os.path.join(directory2, file_name))   
            if show_plot:
                plt.show()
            plt.close()
            
            plt.plot(center_space_mesh, Solution_reduced[number_cells : 2 * number_cells, 0], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label='t=0')
            plt.xlabel('abcissa', fontsize = 14)
            plt.ylabel('Reduced solution u2', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)      
            plt.tight_layout()
            file_name = "Sol_reduced_init_u2_T0.pdf"
            plt.savefig(os.path.join(directory2, file_name))   
            if show_plot:
                plt.show()
            plt.close()
            
            plt.plot(center_space_mesh, Solution_reduced[2 * number_cells : 3 * number_cells, 0], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label='t=0')
            plt.xlabel('abcissa', fontsize = 14)
            plt.ylabel('Reduced solution u3', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)      
            plt.tight_layout()
            file_name = "Sol_reduced_init_u3_T0.pdf"
            plt.savefig(os.path.join(directory2, file_name))   
            if show_plot:
                plt.show()
            plt.close()
            
    
            for ii in range(0, number_species):
                
                
                mass_init_reduced[ii, :] = Dx * Solution_reduced[ii * number_cells : (ii + 1) * number_cells, 0];
                mass_init_domain_reduced[ii] = np.sum(mass_init_reduced[ii, :]);
                concentration_species_inf_domain_reduced[ii] = (mass_init_domain_reduced[ii])/(number_cells * Dx); 
                
            entropy_inf_reduced = number_cells * Dx * np.dot(concentration_species_inf_domain_reduced, np.log(concentration_species_inf_domain_reduced));
    
                        
            ## Entropy
            entropy_reduced[0] = solution_property.entropy_property(Solution_reduced[:, 0], Dx, number_cells, number_species);
            
            ## Check sum concentrations
            check_sum_concentrations_reduced[0] = np.mean(Solution_reduced[0 : number_cells, 0] + Solution_reduced[number_cells : 2 * number_cells, 0] + Solution_reduced[2 * number_cells : 3 * number_cells, 0]);
    
            ## Initializing the newton method for reduced coefficients of size r and for reduced solution of size N * Ne
            whole_coeff_reduced[:, 0] = np.copy(coeff_reduced_init);
            whole_zbar[:, 0] = np.copy(zbar_init);
            newton_coeff_reduced = np.copy(coeff_reduced_init);
            sol_init_reduced = np.copy(Solution_reduced[:, 0]);
            Sol_newton_reduced = np.copy(Solution_reduced[:, 0]);
            
            mass_conservation_reduced = np.zeros((len(index_time_step), number_species));
            mass_conservation_reduced[0, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution_reduced, 0, mass_init_domain_reduced, Dx);  
    
        
            for nn in index_time_step[1 : ]:       
                reshape_Solution = np.zeros((number_cells, number_species));
                ## Need to computation of the edge function u_{j,ϭ}^n 
                edge_solution_reduced = np.zeros((number_cells * number_species, 2));
            
                for ii in range(0, number_species):
        
                    edge_solution_reduced[ii * number_cells : (ii + 1) * number_cells, :] = edge_function.edge_solution_reduced_safe(Sol_newton_reduced[ii * number_cells : (ii + 1) * number_cells], number_cells, number_species, number_internal_edges);  
        
                ### Newton solver
            
                whole_coeff_reduced[:, nn], Solution_reduced[:, nn], number_newton_iter_reduced[nn] = Newton_solver.Newton_solver_reduced_safe(reduced_basis,
                                                   Sol_newton_reduced, newton_coeff_reduced, sol_init_reduced, coeff_reduced_init, edge_solution_reduced, number_cells, number_internal_edges,
                                                   index_internal_edges, number_species, Dx, Dt, struct_a_star[ll], Mat_coeff_astar,
                                                   index_different_specie, dist_center_cells, T_edge, nn, choose_reduced_model, center_space_mesh);
                
                
                
                print(max(Solution_reduced[:, nn]))                               
                                   
                ## Update
                if min(Solution_reduced[:, nn]) < 0:
                    print('reduced solution has neg components at time {}'.format(nn))            
                    temp_reduced_sol = Solution_reduced[:, nn];
                    temp_neg = temp_reduced_sol < 0;
                    if np.max(temp_reduced_sol[temp_neg]) > -1e-5:
                        Solution_reduced[:, nn] = abs(Solution_reduced[:, nn]);
                    else:
                        print('Error: solution should be positive!')
                            
                sol_init_reduced = np.copy(Solution_reduced[:, nn]);
                Sol_newton_reduced = np.copy(sol_init_reduced);
                coeff_reduced_init = np.copy(whole_coeff_reduced[:, nn]);
                newton_coeff_reduced = np.copy(coeff_reduced_init);
                       
                ## Entropy property
                entropy_reduced[nn] = solution_property.entropy_property(Solution_reduced[:, nn], Dx, number_cells, number_species);
            
                ## Mass conservation 
                check_sum_concentrations_reduced[nn, :] = np.mean(Solution_reduced[0 : number_cells, nn] + Solution_reduced[number_cells : 2 * number_cells, nn] + Solution_reduced[2 * number_cells : 3 * number_cells, nn]);
               
                ## Check mass conservation
                mass_conservation_reduced[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution_reduced, nn, mass_init_domain_reduced, Dx);
            
        else:
            coeff_reduced_init, sol_init_reduced = orthogonal_projection.orthogonal_projection(reduced_basis, TOTO_init, dim_r);
            Solution_reduced[:, 0] = np.copy(sol_init_reduced);
            whole_coeff_reduced[:, 0] = np.copy(coeff_reduced_init);
            
            
            ## Plot reduced solution at t=0
            
            plt.plot(center_space_mesh, Solution_reduced[0 : number_cells, 0], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label='t=0')
            plt.xlabel('abcissa', fontsize = 14)
            plt.ylabel('Reduced solution u1', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)      
            plt.tight_layout()
            file_name = "Sol_reduced_init_u1_T0.pdf"
            plt.savefig(os.path.join(directory2, file_name))   
            if show_plot:
                plt.show()
            plt.close()
            
            plt.plot(center_space_mesh, Solution_reduced[number_cells : 2 * number_cells, 0], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label='t=0')
            plt.xlabel('abcissa', fontsize = 14)
            plt.ylabel('Reduced solution u2', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)      
            plt.tight_layout()
            file_name = "Sol_reduced_init_u2_T0.pdf"
            plt.savefig(os.path.join(directory2, file_name))   
            if show_plot:
                plt.show()
            plt.close()
            
            plt.plot(center_space_mesh, Solution_reduced[2 * number_cells : 3 * number_cells, 0], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label='t=0')
            plt.xlabel('abcissa', fontsize = 14)
            plt.ylabel('Reduced solution u3', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)      
            plt.tight_layout()
            file_name = "Sol_reduced_init_u3_T0.pdf"
            plt.savefig(os.path.join(directory2, file_name))   
            if show_plot:
                plt.show()
            plt.close()
            
            if PVD_process == 1:
                plt.plot(center_space_mesh, Solution_reduced[3 * number_cells : 4 * number_cells, 0], marker = 's', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label='t=0')
                plt.xlabel('abcissa', fontsize = 14)
                plt.ylabel('Reduced solution u4', fontsize = 14)
                plt.xticks(fontsize = 14)
                plt.yticks(fontsize = 14)
                plt.axis('equal')
                plt.xlim(0, L)
                plt.legend(fontsize = 13)      
                plt.tight_layout()
                file_name = "Sol_reduced_init_u4_T0.pdf"
                plt.savefig(os.path.join(directory2, file_name))   
                if show_plot:
                    plt.show()
                plt.close()
            
                
            for ii in range(0, number_species):
                mass_init_reduced[ii, :] = Dx * Solution_reduced[ii * number_cells : (ii + 1) * number_cells, 0];
                mass_init_domain_reduced[ii] = np.sum(mass_init_reduced[ii, :]);
                concentration_species_inf_domain_reduced[ii] = (mass_init_domain_reduced[ii])/(number_cells * Dx); 
                
            entropy_inf_reduced = number_cells * Dx * np.dot(concentration_species_inf_domain_reduced, np.log(concentration_species_inf_domain_reduced));
    
            ## Entropy
            entropy_reduced[0] = solution_property.entropy_property(Solution_reduced[:, 0], Dx, number_cells, number_species);
            
            ## Initializing the newton method for reduced coefficients of size r and for reduced solution of size N * Ne
            
            newton_coeff_reduced = np.copy(coeff_reduced_init);
            Sol_newton_reduced = np.copy(sol_init_reduced);
         
            check_sum_concentrations_reduced = 0;
            for ii in range(0, number_species):
                check_sum_concentrations_reduced = np.copy(check_sum_concentrations_reduced) + Solution_reduced[ii * number_cells : (ii + 1) * number_cells, 0];
            
            average_check_sum_concentrations_reduced = sum(check_sum_concentrations_reduced)/number_cells;
            
            if abs(average_check_sum_concentrations_reduced - 1) > 1e-2:
                print('error: the sum of concentrations of species should be around 1')
    #                
            
            mass_conservation_reduced = np.zeros((len(index_time_step), number_species));
            mass_conservation_reduced[0, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution_reduced, 0, mass_init_domain, Dx);  
        #    for ii in range(0, number_species):
        #        mass_reduced_specie_i = my_simpson.my_simpson(center_space_mesh, Solution_reduced[ii * number_cells : (ii + 1) * number_cells, 0], number_cells);             
        #        mass_domain_reduced_specie_i[0, ii] = np.sum(mass_reduced_specie_i);
        #    
            for nn in index_time_step[1 : ]:       
                reshape_Solution = np.zeros((number_cells, number_species));
                ## Need to computation of the edge function u_{j,ϭ}^n 
                edge_solution_reduced = np.zeros((number_cells * number_species, 2));
            
                for ii in range(0, number_species):
        
                    edge_solution_reduced[ii * number_cells : (ii + 1) * number_cells, :] = edge_function.edge_solution(Sol_newton_reduced[ii * number_cells : (ii + 1) * number_cells], number_cells, number_internal_edges);  
        
                ### Newton solver
            
                whole_coeff_reduced[:, nn], Solution_reduced[:, nn], number_newton_iter_reduced[nn] = Newton_solver.Newton_solver_reduced(reduced_basis, Sol_newton_reduced,
                                   newton_coeff_reduced, sol_init_reduced, coeff_reduced_init, edge_solution_reduced, number_cells, number_internal_edges,
                                   index_internal_edges, number_species, Dx, Dt, struct_a_star[ll], Mat_coeff_astar, index_different_specie,
                                   dist_center_cells, T_edge, nn);
                
    
                ## Update
                
                if min(Solution_reduced[:, nn]) < 0:
                    print('reduced solution has neg components at time {}'.format(nn))
                    
                sol_init_reduced = np.copy(Solution_reduced[:, nn]);
                Sol_newton_reduced = np.copy(sol_init_reduced);
                coeff_reduced_init = np.copy(whole_coeff_reduced[:, nn]);
                newton_coeff_reduced = np.copy(coeff_reduced_init);
                       
                ## Entropy property
                entropy_reduced[nn] = solution_property.entropy_property(Solution_reduced[:, nn], Dx, number_cells, number_species);
            
                ## check sum concentrations = 1 or not. This is an average per time step 
     
                check_sum_concentrations_reduced = 0;
                
                for ii in range(0, number_species):
                    check_sum_concentrations_reduced = np.copy(check_sum_concentrations_reduced) +  Solution_reduced[ii * number_cells : (ii + 1) * number_cells, nn];
              
                average_check_sum_concentrations_reduced = np.mean(check_sum_concentrations_reduced);
            
                if abs(average_check_sum_concentrations_reduced - 1) > 1e-2:    
                    print('error: the sum of concentrations of species should be around 1')
                
                ## Check mass conservation
                mass_conservation_reduced[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, Solution_reduced, nn, mass_init_domain_reduced, Dx);  
        
        ##############              #############
        ############## END OF CASES #############
        ##############              #############
        
        ## total Number of Newton iterations per parameter mu
        
        number_Newton_iter_reduced_param_mu[ll] = np.sum(number_newton_iter_reduced[1 :]);
              
       
        ## Compute matrix of all reduced solutions. Warning : here it contains for each mu the init guess
        all_reduced_sol[:, ll * (len(time_observable)) : (ll + 1) * (len(time_observable))] = Solution_reduced[:, indices_coarse_grid];
      
        
#    ## plot entropy dissipation
#    plt.plot(index_time_step, entropy_reduced, marker = 's', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'reduced problem $\mu$ = {0}'.format(ll))
#    plt.xlabel('time', fontsize = 14)
#    plt.ylabel('Entropy', fontsize = 14)
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)
#    plt.tight_layout()
#    file_name = "reduced_pb_entropy_mu {}.pdf".format(ll)
#    plt.savefig(os.path.join(directory2, file_name))        
#    if show_plot:
#        plt.show()
#    plt.close()
#
#
#    ## plot number of newton iterations per time step
#   
#    plt.plot(index_time_step[1 :], number_newton_iter_reduced[1 :], marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'reduced problem $\mu$ = {0}'.format(ll))
#    plt.xlabel('time', fontsize = 14)
#    plt.ylabel('Number of Newton iterations', fontsize = 14)
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)
#    plt.tight_layout()
#    file_name = "reduced_pb_newton_iter_mu {}.pdf".format(ll)
#    plt.savefig(os.path.join(directory2, file_name))        
#    if show_plot:
#        plt.show()
#    plt.close()
#
#    ## Plot sum concentrations. It should not be equal to one.
#    plt.plot(index_time_step, check_sum_concentrations_reduced, marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'reduced problem $\mu$ = {0}'.format(ll))
#    plt.xlabel('time', fontsize = 14)
#    plt.ylabel('Sum of concentrations', fontsize = 14)
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)
#    plt.tight_layout()
#    file_name = "reduced_pb_sum_concentrations_mu {}.pdf".format(ll)
#    plt.savefig(os.path.join(directory2, file_name))        
#    if show_plot:
#           plt.show()
#    plt.close()
#    
#    
#    ## Plot mass conservation for reduced problem
#    plt.plot(index_time_step, mass_conservation_reduced[:, 0], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'mass reduced problem $\mu$ = {0}'.format(ll))
#    plt.plot(index_time_step, mass_init_domain[0] * np.ones(len(index_time_step)), marker = '^', markersize = 4, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 1, label = "mass init")    
#    #plt.plot(index_time_step, mass_conservation[:, 0], marker = 's', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'mass fine problem $\mu$ = {0}'.format(ll))
#    plt.xlabel('time', fontsize = 14)
#    plt.ylabel(r'mass conservation $u_1$', fontsize = 14)
#    #plt.ylim(0.5 * np.min(mass_conservation[:, 0]), 1.5 * np.max(mass_conservation[:, 0]))
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)
#    plt.tight_layout()
#    file_name = "mass_conservation_u1_reduced_mu {}.pdf".format(ll)
#    plt.savefig(os.path.join(directory2, file_name))        
#    if show_plot:
#        plt.show()
#    plt.close()
#    
#    plt.plot(index_time_step, mass_conservation_reduced[:, 1], marker = 'o', markersize = 6, mec = Colors.black, mfc = Colors.black, color = Colors.black, linewidth = 2, label = r'mass reduced problem $\mu$ = {0}'.format(ll))
#    plt.plot(index_time_step, mass_init_domain[1] * np.ones(len(index_time_step)), marker = '^', markersize = 4, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 1, label = "mass init")    
#    #plt.plot(index_time_step, mass_conservation[:, 1], marker = 's', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'mass fine problem $\mu$ = {0}'.format(ll))
#    plt.xlabel('time', fontsize = 14)
#    plt.ylabel(r'mass conservation $u_2$', fontsize = 14)
#    #plt.ylim(0.5 * np.min(mass_conservation[:, 1]), 1.5 * np.max(mass_conservation[:, 1]))
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)
#    plt.tight_layout()
#    file_name = "mass_conservation_u2_reduced_mu {}.pdf".format(ll)
#    plt.savefig(os.path.join(directory2, file_name))        
#    if show_plot:
#        plt.show()
#    plt.close()
#    
#    plt.plot(index_time_step, mass_conservation_reduced[:, 2], marker = 'o', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = r'mass reduced problem $\mu$ = {0}'.format(ll))
#    plt.plot(index_time_step, mass_init_domain[2] * np.ones(len(index_time_step)), marker = '^', markersize = 4, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 1, label = "mass init")    
#    #plt.plot(index_time_step, mass_conservation[:, 2], marker = 's', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'mass fine problem $\mu$ = {0}'.format(ll))
#    plt.xlabel('time', fontsize = 14)
#    plt.ylabel(r'mass conservation $u_3$', fontsize = 14)
#    #plt.ylim(0.5 * np.min(mass_conservation[:, 2]), 1.5 * np.max(mass_conservation[:, 2]))
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)
#    plt.tight_layout()
#    file_name = "mass_conservation_u3_reduced_mu {}.pdf".format(ll)
#    plt.savefig(os.path.join(directory2, file_name))        
#    if show_plot:
#        plt.show()
#    plt.close()
#    
#
#   ## Compute E(T--->+inf) - E(T) for reduced pb in log log scale
#    plt.loglog(index_time_step, abs(entropy_reduced - entropy_reduced[-1]), marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Entropy computation")
#    plt.loglog(index_time_step, abs(entropy_reduced - entropy_inf_reduced),  marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Entropy computation with constant profile")
#    plt.xlabel('time', fontsize = 14)
#    plt.ylabel(r'$|E_{\mathcal{T}}(T \rightarrow +\infty) - E_{\mathcal{T}}(t)|$', fontsize = 14)
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)
#    plt.legend(fontsize = 13)    
#    plt.tight_layout()
#    file_name = "Entropy_inf_reduced_mu {}.pdf".format(ll)
#    plt.savefig(os.path.join(directory2, file_name))        
#    if show_plot:
#        plt.show()
#    plt.close()
#    
#

###############################################################################
############################ SAVE REDUCED SOL #################################
###############################################################################

if save_reduced_sol == 1:    
    np.savetxt("reduced_sol_Mat_online_{}.txt".format(suffix), all_reduced_sol)   
    reduced_sol_online = np.copy(all_reduced_sol);
###############################################################################
#################LOAD REDUCED SOL OFFLINE #####################################
###############################################################################

if load_reduced_sol_offline == 1:
    reduced_sol_offline = np.loadtxt("reduced_sol_Mat_{}.txt".format(suffix))
            
###############################################################################
###################### LOAD REDUCED SOL ONLINE ################################
###############################################################################

if load_reduced_sol_online == 1:
    reduced_sol_online = np.loadtxt("reduced_sol_Mat_online_{}.txt".format(suffix))  
    
    
  

###############################################################################
############################ FULL REDUCED SOL #################################
###############################################################################

if load_reduced_sol_online == 1:
    Full_all_reduced_sol = np.concatenate((reduced_sol_offline, reduced_sol_online), axis = 1)    

    
###############################################################################
############################ SAVE FULL REDUCED SOL USELESS ############################
###############################################################################
#if online_stage == 1:    
#    if PVD_process == 1:
#        np.savetxt("Full_reduced_sol_Mat_PVD_{}.txt".format(suffix), Full_all_reduced_sol)     
#    else:
#        np.savetxt("Full_reduced_sol_Mat_{}.txt".format(suffix), Full_all_reduced_sol)

###############################################################################
####################### PROPERTIES REDUCED SOLUTION ###########################
###############################################################################

if validation_stage == 0:
    
    ### Extract the solutions without initial guess
    
    all_indices = np.arange(0, all_reduced_sol.shape[1]);
    indices_without_time_init = np.linspace(0, len(time_observable) * (number_parameter_mu), number_parameter_mu + 1, dtype = int);
    extract = np.setdiff1d(all_indices, indices_without_time_init);
    init_temp = np.zeros(len(all_indices), dtype = bool);
    init_temp[extract] = True;    
    extract_reduced_sol_Mat = all_reduced_sol[:, init_temp];
    
    ### Plot mass conservation : WARNING MASS IS CONSIDERED AT TIME OBSERVABLE. For Validation case we need to compute the reduced init mass. 
    
    
    mass_conservation_reduced = np.zeros((len(time_observable[1 :]), number_species));
    
    linf_error_time_mass = np.zeros(number_parameter_mu);
    for ll in range(0, number_parameter_mu):
        extract_snapshots_mu = extract_reduced_sol_Mat[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
        for nn in range(0, len(time_observable[1 :])):
            mass_conservation_reduced[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, extract_snapshots_mu, nn, mass_init_domain_reduced, Dx);
        linf_error_time_mass[ll] = np.max(np.max(abs(mass_conservation_reduced - np.matlib.repmat(mass_init_domain_reduced, len(time_observable[1 :]), 1)), 0));             
    
    ## plot mass conservation 
    plt.plot(np.arange(0, number_parameter_mu), linf_error_time_mass, marker = 'o', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "mass deviation from initial mass")
    plt.xlabel(r' parameter $\mu$', fontsize = 14)
    plt.ylabel(r' mass', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)        
    plt.xlim(0, number_parameter_mu)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "mass_conservation_reduced_pb.pdf"
    plt.savefig(os.path.join(directory2, file_name))   
    if show_plot:
        plt.show()
    plt.close()
    
    ## Plot entropy
    for ll in range(0, number_parameter_mu):
        extract_snapshots_mu = extract_reduced_sol_Mat[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
        entropy_reduced = np.zeros(len(time_observable[1 :]));
        
        for nn in range(0, len(time_observable[1 :])):                
            entropy_reduced[nn] = solution_property.entropy_property(extract_snapshots_mu[:, nn], Dx, number_cells, number_species);
    #            
        plt.plot(indices_coarse_grid[1 :], entropy_reduced, marker = 's', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'reduced problem $\mu$ = {}'.format(ll))
        plt.xlabel('time step', fontsize = 14)
        plt.ylabel('Entropy', fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 13)
        plt.tight_layout()
        file_name = "reduced_pb_entropy_mu_{}.pdf".format(ll)
        plt.savefig(os.path.join(directory2, file_name))
        if show_plot:
            plt.show()
        plt.close()
    
    
    ## Check that solution is always positive for safe reduced model. For not safe reduced model it could be negative
     
    min_x_min_mu_sol = check_properties_reduced_sol.check_positivity(all_reduced_sol, number_species, number_cells, time_observable,
                         number_parameter_mu, directory2, show_plot);
    
    plt.plot(indices_coarse_grid, min_x_min_mu_sol, marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'positivity')
    plt.xlabel('time step', fontsize = 14)
    plt.ylabel('Reduced solution', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "check_reduced_pb_positivity_sol.pdf"
    plt.savefig(os.path.join(directory2, file_name))        
    if show_plot:
        plt.show()
    plt.close()
    ## Check that solution is always below than 1
    
    
    max_x_max_mu_sol = check_properties_reduced_sol.check_below_one(extract_reduced_sol_Mat, number_species, number_cells, time_observable,
                         number_parameter_mu, directory2, show_plot);
    
    plt.plot(indices_coarse_grid[1 :], max_x_max_mu_sol, marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'Solution max')
    plt.xlabel('time step', fontsize = 14)
    plt.ylabel('Reduced solution', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "check_reduced_pb_sol_below_1.pdf"
    plt.savefig(os.path.join(directory2, file_name))        
    if show_plot:
        plt.show()
    plt.close()                                                                
                                                                    
    ## Check that sum species is always equal to 1
    
    min_x_min_mu_reshape_sum_snap = check_properties_reduced_sol.check_sum_species_equal_one(number_cells, number_species, number_parameter_mu,
                                    extract_reduced_sol_Mat, time_observable, directory2, show_plot);
    
    plt.plot(indices_coarse_grid[1 :], min_x_min_mu_reshape_sum_snap, marker = 'o', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = r'Volume filling constraint')
    plt.xlabel('time step', fontsize = 14)
    plt.ylabel('Reduced solution', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "check_reduced_pb_sol_equal_1.pdf"
    plt.savefig(os.path.join(directory2, file_name))        
    if show_plot:
        plt.show()
    plt.close()                         
    
    #### PLOT SOL IN SAME GRAPH
    for ll in range(0, number_parameter_mu):
        extract_snapshots_mu = all_reduced_sol[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
        ## plot solution at time observable
        for nn in range(0, len(time_observable[1 :])):        
            
            ## plot solution
            if PVD_process == 1:
                plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[3 * number_cells : 4 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "Solution u4 at n = {}".format(indices_coarse_grid[nn + 1]))
        
            else:
                plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.xlabel('abcissa', fontsize = 14)
            
            plt.ylabel('Solution', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)
            plt.tight_layout()
            file_name = "Solution_reduced_at_time_{}_and_mu_{}.pdf".format(indices_coarse_grid[nn + 1], ll)
            plt.savefig(os.path.join(directory2, file_name))   
            if show_plot:
                plt.show()
            plt.close()
    
    ## Compute Linf error in mu, space and time sup_mu ||...||_{L^{inf}(0,T,Omega)}
        
    error_Linf_mu_space_time = compute_error.compute_Linf_error(number_species, number_cells, number_parameter_mu,
                           time_observable, extract_reduced_sol_Mat, snapshots_sol,
                           directory2, show_plot, PVD_process);
          
    
    ## Compute Linf error in mu, L2 in space and Linf in time :  sup_mu ||...||_{L^{inf}(0,T,L²(Omega))}
    
    error_Linf_mu_L2_space_Linf_time = compute_error.compute_L2_error(number_species, number_cells, number_parameter_mu,
                           time_observable, extract_reduced_sol_Mat, snapshots_sol,
                           directory2, show_plot, PVD_process);

else:
    
    ### Extract the solutions without initial guess
    #
    all_indices = np.arange(0, reduced_sol_offline.shape[1]);
    indices_without_time_init = np.linspace(0, len(time_observable) * (number_parameter_mu), number_parameter_mu + 1, dtype = int);
    extract = np.setdiff1d(all_indices, indices_without_time_init);
    init_temp = np.zeros(len(all_indices), dtype = bool);
    init_temp[extract] = True;    
    extract_reduced_sol_Mat_offline = reduced_sol_offline[:, init_temp];
    
    ### Plot mass conservation : WARNING MASS IS CONSIDERED AT TIME OBSERVABLE. For Validation case we need to compute the reduced init mass. 
   
    if choose_reduced_model == 2:
                
        z_init = np.log(TOTO_init + tol_eps);            
        coeff_reduced_init, zbar_init = orthogonal_projection.orthogonal_projection(reduced_basis, z_init, dim_r);
        Solution_reduced_init_time = compute_safe_sol.compute_safe_sol_bis(zbar_init, number_species, number_cells);
        	
        for ii in range(0, number_species):
                                       
            mass_init_reduced[ii, :] = Dx * Solution_reduced_init_time[ii * number_cells : (ii + 1) * number_cells];
            mass_init_domain_reduced[ii] = np.sum(mass_init_reduced[ii, :]);
                    
    else:
        coeff_reduced_init, sol_init_reduced = orthogonal_projection.orthogonal_projection(reduced_basis, TOTO_init, dim_r);
        Solution_reduced_init_time = np.copy(sol_init_reduced);
        for ii in range(0, number_species):
            mass_init_reduced[ii, :] = Dx * Solution_reduced_init_time[ii * number_cells : (ii + 1) * number_cells];
            mass_init_domain_reduced[ii] = np.sum(mass_init_reduced[ii, :]);
    
    mass_conservation_reduced = np.zeros((len(time_observable[1 :]), number_species));
    
    linf_error_time_mass_offline = np.zeros(number_parameter_mu);
    for ll in range(0, number_parameter_mu):
        extract_snapshots_mu_offline = extract_reduced_sol_Mat_offline[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
        for nn in range(0, len(time_observable[1 :])):
            mass_conservation_reduced[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, extract_snapshots_mu_offline, nn, mass_init_domain_reduced, Dx);
        linf_error_time_mass_offline[ll] = np.max(np.max(abs(mass_conservation_reduced - np.matlib.repmat(mass_init_domain_reduced, len(time_observable[1 :]), 1)), 0));             
    
    all_indices = np.arange(0, reduced_sol_online.shape[1]);
    indices_without_time_init = np.linspace(0, len(time_observable) * (number_parameter_mu_online), number_parameter_mu_online + 1, dtype = int);
    extract = np.setdiff1d(all_indices, indices_without_time_init);
    init_temp = np.zeros(len(all_indices), dtype = bool);
    init_temp[extract] = True;    
    extract_reduced_sol_Mat_online = reduced_sol_online[:, init_temp];
    
    linf_error_time_mass_online = np.zeros(number_parameter_mu_online);
    
    for ll in range(0, number_parameter_mu_online):
        extract_snapshots_mu_online = extract_reduced_sol_Mat_online[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
        for nn in range(0, len(time_observable[1 :])):
            mass_conservation_reduced[nn, :] = check_mass_conservation.check_mass_conservation(number_species, center_space_mesh, number_cells, extract_snapshots_mu_online, nn, mass_init_domain_reduced, Dx);
        linf_error_time_mass_online[ll] = np.max(np.max(abs(mass_conservation_reduced - np.matlib.repmat(mass_init_domain_reduced, len(time_observable[1 :]), 1)), 0));             
    
    linf_error_time_mass = np.concatenate((linf_error_time_mass_offline, linf_error_time_mass_online), axis = 0);
    
    ## plot mass conservation 
    plt.plot(np.arange(0, number_parameter_mu + number_parameter_mu_online), linf_error_time_mass, marker = 'o', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "mass deviation from initial mass")
    plt.xlabel(r' parameter $\mu$', fontsize = 14)
    plt.ylabel(r' mass', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)        
    plt.xlim(0, number_parameter_mu + number_parameter_mu_online)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "mass_conservation_validation_reduced_pb.pdf"
    plt.savefig(os.path.join(directory3, file_name))   
    if show_plot:
        plt.show()
    plt.close()
    
    ## Plot entropy
    for ll in range(0, number_parameter_mu):
        entropy_reduced_offline = np.zeros(len(time_observable[1 :]));
        
        for nn in range(0, len(time_observable[1 :])):                
            entropy_reduced_offline[nn] = solution_property.entropy_property(extract_snapshots_mu_offline[:, nn], Dx, number_cells, number_species);
                
    for ll in range(0, number_parameter_mu_online):
        entropy_reduced_online = np.zeros(len(time_observable[1 :]));
        
        for nn in range(0, len(time_observable[1 :])):                
            entropy_reduced_online[nn] = solution_property.entropy_property(extract_snapshots_mu_online[:, nn], Dx, number_cells, number_species);
    
    entropy_reduced = np.concatenate((entropy_reduced_offline, entropy_reduced_online), axis = 0);
    
    for ll in range(0, number_parameter_mu):
        plt.plot(indices_coarse_grid[1 :], entropy_reduced_offline, marker = 's', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'reduced problem $\mu$ = {}'.format(ll))
        plt.xlabel('time step', fontsize = 14)
        plt.ylabel('Entropy', fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 13)
        plt.tight_layout()
        file_name = "reduced_pb_entropy_validation_mu_{}.pdf".format(ll)
        plt.savefig(os.path.join(directory3, file_name))
        if show_plot:
            plt.show()
        plt.close()
        
    for ll in range(number_parameter_mu, number_parameter_mu + number_parameter_mu_online):
        plt.plot(indices_coarse_grid[1 :], entropy_reduced_offline, marker = 's', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'reduced problem $\mu$ = {}'.format(ll))
        plt.xlabel('time step', fontsize = 14)
        plt.ylabel('Entropy', fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 13)
        plt.tight_layout()
        file_name = "reduced_pb_entropy_validation_mu_{}.pdf".format(ll)
        plt.savefig(os.path.join(directory3, file_name))
        if show_plot:
            plt.show()
        plt.close()    
    
    
    ## Check that solution is always positive for safe reduced model. For not safe reduced model it could be negative
     
    min_x_min_mu_sol = check_properties_reduced_sol.check_positivity(Full_all_reduced_sol, number_species, number_cells, time_observable,
                         number_parameter_mu + number_parameter_mu_online, directory3, show_plot);
    
    plt.plot(indices_coarse_grid, min_x_min_mu_sol, marker = 'o', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = r'positivity')
    plt.xlabel('time step', fontsize = 14)
    plt.ylabel('Reduced solution', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "check_reduced_pb_positivity_sol_validation.pdf"
    plt.savefig(os.path.join(directory3, file_name))        
    if show_plot:
        plt.show()
    plt.close()
    
    
    ## Check that solution is always below than 1
    
    max_x_max_mu_sol = check_properties_reduced_sol.check_below_one(Full_all_reduced_sol, number_species, number_cells, time_observable,
                         number_parameter_mu + number_parameter_mu_online, directory3, show_plot);
    
    ## changed here 25 April 2022 : indices_coarse_grid[1, :] ----> indices_coarse_grid
    plt.plot(indices_coarse_grid, max_x_max_mu_sol, marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = r'Solution max')
    plt.xlabel('time step', fontsize = 14)
    plt.ylabel('Reduced solution', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "check_reduced_pb_sol_below_1_validation.pdf"
    plt.savefig(os.path.join(directory3, file_name))        
    if show_plot:
        plt.show()
    plt.close()                                                                
                                                                    
    ## Check that sum species is always equal to 1
    
    min_x_min_mu_reshape_sum_snap = check_properties_reduced_sol.check_sum_species_equal_one(number_cells, number_species, number_parameter_mu + number_parameter_mu_online,
                                    Full_all_reduced_sol, time_observable, directory3, show_plot);
    
    ## changed here 25 April 2022 : indices_coarse_grid[1, :] ----> indices_coarse_grid                                                                                         
    plt.plot(indices_coarse_grid, min_x_min_mu_reshape_sum_snap, marker = 'o', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = r'Volume filling constraint')
    plt.xlabel('time step', fontsize = 14)
    plt.ylabel('Reduced solution', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 13)
    plt.tight_layout()
    file_name = "check_reduced_pb_sol_equal_1_validation.pdf"
    plt.savefig(os.path.join(directory3, file_name))        
    if show_plot:
        plt.show()
    plt.close()                         
    
    #### PLOT SOL IN SAME GRAPH
#    for ll in range(0, number_parameter_mu):
#        extract_snapshots_mu = reduced_sol_offline[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
#        ## plot solution at time observable
#        for nn in range(0, len(time_observable[1 :])):        
#            
#            ## plot solution
#            if PVD_process == 1:
#                plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
#                plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
#                plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
#                plt.plot(center_space_mesh, extract_snapshots_mu[3 * number_cells : 4 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "Solution u4 at n = {}".format(indices_coarse_grid[nn + 1]))
#        
#            else:
#                plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
#                plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
#                plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
#                plt.xlabel('abcissa', fontsize = 14)
#            
#            plt.ylabel('Solution', fontsize = 14)
#            plt.xlabel('abcissa', fontsize = 14)
#            plt.xticks(fontsize = 14)
#            plt.yticks(fontsize = 14)
#            plt.axis('equal')
#            plt.xlim(0, L)
#            plt.legend(fontsize = 13)
#            plt.tight_layout()
#            file_name = "Solution_reduced_at_time_{}_and_mu_{}.pdf".format(indices_coarse_grid[nn + 1], ll)
#            plt.savefig(os.path.join(directory3, file_name))   
#            if show_plot:
#                plt.show()
#            plt.close()
#    
    
    for ll in range(0, number_parameter_mu_online):
        extract_snapshots_mu = reduced_sol_online[:, ll * len(time_observable[1 :]) : (ll + 1) * len(time_observable[1 :])];
        ## plot solution at time observable
        for nn in range(0, len(time_observable[1 :])):        
            
            ## plot solution
            if PVD_process == 1:
                plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[3 * number_cells : 4 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, label = "Solution u4 at n = {}".format(indices_coarse_grid[nn + 1]))
        
            else:
                plt.plot(center_space_mesh, extract_snapshots_mu[0 : number_cells, nn], marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2, label = "Solution u1 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[number_cells : 2 * number_cells, nn], marker = '^', markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, label = "Solution u2 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.plot(center_space_mesh, extract_snapshots_mu[2 * number_cells : 3 * number_cells, nn], marker = 's', markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, label = "Solution u3 at n = {}".format(indices_coarse_grid[nn + 1]))
                plt.xlabel('abcissa', fontsize = 14)
            
            plt.ylabel('Solution', fontsize = 14)
            plt.xlabel('abcissa', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.axis('equal')
            plt.xlim(0, L)
            plt.legend(fontsize = 13)
            plt.tight_layout()
            file_name = "Solution_reduced_at_time_{}_and_mu_{}.pdf".format(indices_coarse_grid[nn + 1], ll + number_parameter_mu_online)
            plt.savefig(os.path.join(directory3, file_name))   
            if show_plot:
                plt.show()
            plt.close()
    
    ## Compute Linf error in mu, space and time sup_mu ||...||_{L^{inf}(0,T,Omega)}
        
    error_Linf_mu_space_time = compute_error.compute_Linf_error(number_species, number_cells, number_parameter_mu + number_parameter_mu_online,
                           time_observable, np.concatenate((extract_reduced_sol_Mat_offline, extract_reduced_sol_Mat_online), axis = 1), full_snapshots_sol,
                           directory3, show_plot, PVD_process);
          
    
    ## Compute Linf error in mu, L2 in space and Linf in time :  sup_mu ||...||_{L^{inf}(0,T,L²(Omega))}
    
    error_Linf_mu_L2_space_Linf_time = compute_error.compute_L2_error(number_species, number_cells, number_parameter_mu + number_parameter_mu_online,
                           time_observable, np.concatenate((extract_reduced_sol_Mat_offline, extract_reduced_sol_Mat_online), axis = 1), full_snapshots_sol,
                           directory3, show_plot, PVD_process); 
#                                                                
#                                                                                         
#
#                                                                                         
with open("parameter_pb_validation_{}.txt".format(suffix), "w") as text_file:
    text_file.write("PVD case Online validation")
    text_file.write("\n tol_SVD=: %f" % tol_SVD)
    text_file.write("\n number species=: %f" % number_species)
    text_file.write("\n L=: %f" % L)
    text_file.write("\n number cells=: %f" % number_cells)
    text_file.write("\n Dx=: %f" % Dx)
    text_file.write("\n Tfinal=: %f" % Tf)
    text_file.write("\n DT=: %f" % DT)
    text_file.write("\n Dt=: %f" % Dt)
    text_file.write("\n number time step=: %f" % number_time_step)
    text_file.write("\n dim reduced space r=: %f" % temp_dim_r)
    text_file.write("\n dim reduced space r*=: %f" % dim_r)
    text_file.write('\n Linf error mu space time {0}\n'.format(error_Linf_mu_space_time))
    text_file.write('\n Linf error mu L2 space Linf time {0}\n'.format(error_Linf_mu_L2_space_Linf_time))
    text_file.write('\n max Linf error mu L2 space Linf time {0}\n'.format(np.max(error_Linf_mu_L2_space_Linf_time)))
    text_file.write('\n min Linf error mu L2 space Linf time {0}\n'.format(np.min(error_Linf_mu_L2_space_Linf_time)))
    text_file.write('\n mass_deviation {0}\n'.format(linf_error_time_mass))
    text_file.write('\n max_mass_deviation {0}\n'.format(np.max(linf_error_time_mass)))
    text_file.write('\n min_mass_deviation {0}\n'.format(np.min(linf_error_time_mass)))
    text_file.write('\n Positivity of solution {0}\n'.format(min_x_min_mu_sol))
    text_file.write('\n max Positivity of solution {0}\n'.format(np.max(min_x_min_mu_sol)))
    text_file.write('\n min Positivity of solution {0}\n'.format(np.min(min_x_min_mu_sol)))
    text_file.write('\n sum concentrations equal to 1 {0}\n'.format(min_x_min_mu_reshape_sum_snap))
#    text_file.write("%%%%%%% FINE PROBLEM %%%%%%%")
#    #text_file.write('\n total_number_Newton_iter_per_mu {0}\n'.format(number_Newton_iter_param_mu))
#    #text_file.write("%%%%%%% REDUCED PROBLEM %%%%%%%")
#    #text_file.write('\n total_number_Newton_iter_reduced_per_mu {0}\n'.format(number_Newton_iter_reduced_param_mu))
    text_file.close()

print("%%%%%%%%%End of simulation%%%%%%%%%%%%")    
