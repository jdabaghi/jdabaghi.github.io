#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:08:58 2022

@author: carnot-smiles
"""


import numpy as np
import matplotlib.pyplot as plt
import Colors
import os
from numpy import linalg as LA

#######

    ## Compute Linf error in mu, space and time sup_mu ||...||_{L^{inf}(0,T,Omega)}


def compute_Linf_error(number_species, number_cells, number_parameter_mu,
                       index_time_step, all_reduced_sol, snapshots_sol,
                       my_path2, show_plot, PVD_process):



    save_error_Linf_time = np.zeros((number_parameter_mu, number_species));
    
        
    ## Plot Linf error between fine and reduced model    
    for ll in range(0, number_parameter_mu):
        
        temp_error = abs(all_reduced_sol[:, ll * len(index_time_step[1 :]) : (ll + 1) * len(index_time_step[1 :])] - snapshots_sol[:, ll * len(index_time_step[1 :]) : (ll + 1) * len(index_time_step[1 :])]);
        
        if PVD_process == 1:
            Linf_error_fine_reduced_u1 = np.max(temp_error[0 : number_cells, :], 0);
            Linf_error_fine_reduced_u2 = np.max(temp_error[number_cells : 2 * number_cells, :], 0);
            Linf_error_fine_reduced_u3 = np.max(temp_error[2 * number_cells : 3 * number_cells, :], 0);
            Linf_error_fine_reduced_u4 = np.max(temp_error[3 * number_cells : 4 * number_cells, :], 0);
    
            save_error_Linf_time[ll, 0] = np.max(Linf_error_fine_reduced_u1);
            save_error_Linf_time[ll, 1] = np.max(Linf_error_fine_reduced_u2);
            save_error_Linf_time[ll, 2] = np.max(Linf_error_fine_reduced_u3);
            save_error_Linf_time[ll, 3] = np.max(Linf_error_fine_reduced_u4);
        else:
            
            Linf_error_fine_reduced_u1 = np.max(temp_error[0 : number_cells, :], 0);
            Linf_error_fine_reduced_u2 = np.max(temp_error[number_cells : 2 * number_cells, :], 0);
            Linf_error_fine_reduced_u3 = np.max(temp_error[2 * number_cells : 3 * number_cells, :], 0);
            
            save_error_Linf_time[ll, 0] = np.max(Linf_error_fine_reduced_u1);
            save_error_Linf_time[ll, 1] = np.max(Linf_error_fine_reduced_u2);
            save_error_Linf_time[ll, 2] = np.max(Linf_error_fine_reduced_u3);
        
        
#        plt.plot(index_time_step[1 :], Linf_error_fine_reduced_u1, marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2,
#                 label = r' specie $i=1$ and $\mu$ = {}'.format(ll))
#        plt.xlabel('time', fontsize = 14)
#        plt.ylabel(r'$|| u_{\mu}(t) - u_{\mu}^{\mathrm{red}}(t) ||_{L^{\infty}(\Omega)}$', fontsize = 14)
#        plt.xticks(fontsize = 14)
#        plt.yticks(fontsize = 14)
#        plt.legend(fontsize = 13)
#        plt.tight_layout()
#        file_name = "error_sol_u1_reduced_at_mu_{}.pdf".format(ll)
#        plt.savefig(os.path.join(my_path2, file_name))        
#        if show_plot:
#            plt.show()
#        plt.close()
#        
#        plt.plot(index_time_step[1 :], Linf_error_fine_reduced_u2, marker = '^', 
#                 markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, 
#                 label = r' specie $i=2$ and $\mu$ = {}'.format(ll))
#        plt.xlabel('time', fontsize = 14)
#        plt.ylabel(r'$|| u_{\mu}(t) - u_{\mu}^{\mathrm{red}}(t) ||_{L^{\infty}(\Omega)}$', fontsize = 14)
#        plt.xticks(fontsize = 14)
#        plt.yticks(fontsize = 14)
#        plt.legend(fontsize = 13)
#        plt.tight_layout()
#        file_name = "error_sol_u2_reduced_at_mu_{}.pdf".format(ll)
#        plt.savefig(os.path.join(my_path2, file_name))        
#        if show_plot:
#            plt.show()
#        plt.close()
#        
#        plt.plot(index_time_step[1 :], Linf_error_fine_reduced_u3, marker = 's', 
#                 markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, 
#                 label = r' specie $i=3$ and $\mu$ = {}'.format(ll))
#        plt.xlabel('time', fontsize = 14)
#        plt.ylabel(r'$L^{\mathrm{inf}}$ error specie $i = 3$', fontsize = 14)
#        plt.xticks(fontsize = 14)
#        plt.yticks(fontsize = 14)
#        plt.legend(fontsize = 13)
#        plt.tight_layout()
#        file_name = "error_sol_u3_reduced_at_mu_{}.pdf".format(ll)
#        plt.savefig(os.path.join(my_path2, file_name))        
#        if show_plot:
#            plt.show()
#        plt.close()
#        
#        if PVD_process == 1:
#            plt.plot(index_time_step[1 :], Linf_error_fine_reduced_u4, marker = 's', 
#                     markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, 
#                     label = r' specie $i=4$ and $\mu$ = {}'.format(ll))
#            plt.xlabel('time', fontsize = 14)
#            plt.ylabel(r'$L^{\mathrm{inf}}$ error specie $i = 4$', fontsize = 14)
#            plt.xticks(fontsize = 14)
#            plt.yticks(fontsize = 14)
#            plt.legend(fontsize = 13)
#            plt.tight_layout()
#            file_name = "error_sol_u4_reduced_at_mu_{}.pdf".format(ll)
#            plt.savefig(os.path.join(my_path2, file_name))        
#            if show_plot:
#                plt.show()
#            plt.close()
        
        
    error_Linf_mu_space_time = np.max(save_error_Linf_time, 0);
    
    return error_Linf_mu_space_time



def compute_L2_error(number_species, number_cells, number_parameter_mu,
                       index_time_step, all_reduced_sol, snapshots_sol,
                       my_path2, show_plot, PVD_process):



    L2_error_fine_reduced_u1 = np.zeros(len(index_time_step[1 :]));
    L2_error_fine_reduced_u2 = np.zeros(len(index_time_step[1 :]));
    L2_error_fine_reduced_u3 = np.zeros(len(index_time_step[1 :]));
    L2_error_fine_reduced_u4 = np.zeros(len(index_time_step[1 :]));

    save_error_L2_time = np.zeros((number_parameter_mu, number_species));
    
        
    ## Plot Linf error between fine and reduced model    
    for ll in range(0, number_parameter_mu):
        
        temp_error = all_reduced_sol[:, ll * len(index_time_step[1 :]) : (ll + 1) * len(index_time_step[1 :])] - snapshots_sol[:, ll * len(index_time_step[1 :]) : (ll + 1) * len(index_time_step[1 :])];
        
        for jj in range(0, len(index_time_step[1 :])):
            if PVD_process == 1:
                L2_error_fine_reduced_u1[jj] = LA.norm(temp_error[0 : number_cells, jj]);
                L2_error_fine_reduced_u2[jj] = LA.norm(temp_error[number_cells : 2 * number_cells, jj]);
                L2_error_fine_reduced_u3[jj] = LA.norm(temp_error[2 * number_cells : 3 * number_cells, jj]);
                L2_error_fine_reduced_u4[jj] = LA.norm(temp_error[3 * number_cells : 4 * number_cells, jj]);
            else:       
                L2_error_fine_reduced_u1[jj] = LA.norm(temp_error[0 : number_cells, jj]);
                L2_error_fine_reduced_u2[jj] = LA.norm(temp_error[number_cells : 2 * number_cells, jj]);
                L2_error_fine_reduced_u3[jj] = LA.norm(temp_error[2 * number_cells : 3 * number_cells, jj]);
        
        if PVD_process == 1:
            save_error_L2_time[ll, 0] = np.max(L2_error_fine_reduced_u1);
            save_error_L2_time[ll, 1] = np.max(L2_error_fine_reduced_u2);
            save_error_L2_time[ll, 2] = np.max(L2_error_fine_reduced_u3);
            save_error_L2_time[ll, 3] = np.max(L2_error_fine_reduced_u4);
        else:
            save_error_L2_time[ll, 0] = np.max(L2_error_fine_reduced_u1);
            save_error_L2_time[ll, 1] = np.max(L2_error_fine_reduced_u2);
            save_error_L2_time[ll, 2] = np.max(L2_error_fine_reduced_u3);
        
        
        plt.plot(index_time_step[1 :], L2_error_fine_reduced_u1, marker = 'o', markersize = 6, mec = Colors.OrangeRed2, mfc = Colors.OrangeRed2, color = Colors.OrangeRed2, linewidth = 2,
                 label = r' specie $i=1$ and $\mu$ = {}'.format(ll))
        plt.xlabel('time', fontsize = 14)
        plt.ylabel(r'$|| u_{\mu}(t) - u_{\mu}^{\mathrm{red}}(t) ||_{L^{2}(\Omega)}$', fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 13)
        plt.tight_layout()
        file_name = "L2_error_sol_u1_reduced_at_mu_{}.pdf".format(ll)
        plt.savefig(os.path.join(my_path2, file_name))        
        if show_plot:
            plt.show()
        plt.close()
        
        plt.plot(index_time_step[1 :], L2_error_fine_reduced_u2, marker = '^', 
                 markersize = 6, mec = Colors.midnight_blue, mfc = Colors.midnight_blue, color = Colors.midnight_blue, linewidth = 2, 
                 label = r' specie $i=2$ and $\mu$ = {}'.format(ll))
        plt.xlabel('time', fontsize = 14)
        plt.ylabel(r'$|| u_{\mu}(t) - u_{\mu}^{\mathrm{red}}(t) ||_{L^{2}(\Omega)}$', fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 13)
        plt.tight_layout()
        file_name = "L2_error_sol_u2_reduced_at_mu_{}.pdf".format(ll)
        plt.savefig(os.path.join(my_path2, file_name))        
        if show_plot:
            plt.show()
        plt.close()
        
        plt.plot(index_time_step[1 :], L2_error_fine_reduced_u3, marker = 's', 
                 markersize = 6, mec = Colors.dark_green, mfc = Colors.dark_green, color = Colors.dark_green, linewidth = 2, 
                 label = r' specie $i=3$ and $\mu$ = {}'.format(ll))
        plt.xlabel('time', fontsize = 14)
        plt.ylabel(r'$|| u_{\mu}(t) - u_{\mu}^{\mathrm{red}}(t) ||_{L^{2}(\Omega)}$', fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 13)
        plt.tight_layout()
        file_name = "L2_error_sol_u3_reduced_at_mu_{}.pdf".format(ll)
        plt.savefig(os.path.join(my_path2, file_name))        
        if show_plot:
            plt.show()
        plt.close()
        
        if PVD_process == 1:
            plt.plot(index_time_step[1 :], L2_error_fine_reduced_u4, marker = 's', 
                 markersize = 6, mec = Colors.MediumOrchid3, mfc = Colors.MediumOrchid3, color = Colors.MediumOrchid3, linewidth = 2, 
                 label = r' specie $i=4$ and $\mu$ = {}'.format(ll))
            plt.xlabel('time', fontsize = 14)
            plt.ylabel(r'$|| u_{\mu}(t) - u_{\mu}^{\mathrm{red}}(t) ||_{L^{2}(\Omega)}$', fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.legend(fontsize = 13)
            plt.tight_layout()
            file_name = "L2_error_sol_u4_reduced_at_mu_{}.pdf".format(ll)
            plt.savefig(os.path.join(my_path2, file_name))        
            if show_plot:
                plt.show()
            plt.close()
        
    error_Linf_mu_L2_space_time = np.max(save_error_L2_time, 0);
    
    return error_Linf_mu_L2_space_time