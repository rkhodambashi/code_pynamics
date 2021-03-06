# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
import numpy

class Output(object):
    def __init__(self,y_exp,system):
        import sympy
        self.y_expression = sympy.Matrix(y_exp)
        cons_s = list(system.constants.keys())
        self.cons_v = [system.constants[key] for key in cons_s]
        self.fy_expression = sympy.lambdify(system.get_q(0)+system.get_q(1)+cons_s,self.y_expression)
    def calc(self,x):
        self.y = numpy.array([self.fy_expression(*(item.tolist()+self.cons_v)) for item in x]).squeeze()
        return self.y
