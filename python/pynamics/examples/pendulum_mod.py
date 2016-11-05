# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
#pynamics.script_mode = True
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle

import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()

def der(eq,system):
    b = sympy.Number(0)
    for item in system.get_q(0)+system.get_q(1):
        b+=eq.diff(item)*system.derivative(item)
    return b

lA = Constant('lA',1,system)

mA = Constant('mA',1,system)

g = Constant('g',9.81,system)
b = Constant('b',1e0,system)
k = Constant('k',1e1,system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant('preload1',0*pi/180,system)

x,x_d,x_dd = Differentiable(system,'x')
y,y_d,y_dd = Differentiable(system,'y')
f = Variable('f')

initialvalues = {}
initialvalues[x]=1
initialvalues[x_d]=0
initialvalues[y]=0
initialvalues[y_d]=0

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')

system.set_newtonian(N)

pNA=0*N.x
pAB=pNA+x*N.x+y*N.y
vAB=pAB.time_derivative(N,system)

ParticleA = Particle(system,pAB,mA,'ParticleA')

system.addforce(-b*vAB,vAB)

system.addforcegravity(-g*N.y)

x1 = ParticleA.pCM.dot(N.x)
y1 = ParticleA.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()

pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')

v = pAB-pNA
u = (v.dot(v))**.5

eq1 = [(v.dot(v)) - lA**2]
eq1_dd=der(der(eq1[0],system),system)
eq = [eq1_dd]

#system.addforce(-f*u,vAB)

statevariables = system.get_q(0)+system.get_q(1)
augmented = [f]

func1 = system.create_state_space_constrained(f,ma,eq)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,args=(),rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,KE-PE,x,y],system)
y = output.calc(states)
pynamics.toc()

plt.figure(1)
plt.hold(True)
plt.plot(y[:,0],y[:,1])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,2])

plt.figure(3)
plt.hold(True)
plt.plot(t,y[:,3])
plt.show()
