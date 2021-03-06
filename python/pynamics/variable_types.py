# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
import pynamics

class Variable(sympy.Symbol):
    def __new__(self,name):
        obj = sympy.Symbol.__new__(self,name)
        pynamics.addself(obj,name)
        return obj

class Constant(sympy.Symbol):
    def __new__(self,name,value,system):
        obj = sympy.Symbol.__new__(self,name)
        obj.value = value
        system.add_constant(obj,value)
        pynamics.addself(obj,name)
        return obj

class Differentiable(sympy.Symbol):
    ii = 0    
    def __new__(cls,sys,name=None,limit = 3,ii=0):
        if name==None:
            name = 'x{0:d}'.format(cls.ii)
            cls.ii+=1

        differentiables = []

        for jj in range(ii,limit):
            

            if jj==0:
                subname = name
                variable = sympy.Symbol.__new__(cls,subname)
            else:
                subname = name+'_'+'d'*jj
                variable = sympy.Symbol.__new__(cls,subname)

            sys.add_q(variable,jj)
            pynamics.addself(variable,subname)
            differentiables.append(variable)
        for a,a_d in zip(differentiables[:-1],differentiables[1:]):
            sys.add_derivative(a,a_d)
        return differentiables 
