#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:45:26 2021

@author: carnot-smiles
"""

#def reduced_model_case(i, Dx):
#        switcher={
#                0:'Not safe reduced model',
#                1:'Safe reduced model',
#                      }
#        return switcher.get(i,"Invalid reduced model")
    
def reduced_model_case(x):
    return {
        'a': 1,
        'b': 2,
    }.get(x, 0)    # 0 is default if x not found