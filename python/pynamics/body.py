# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
import pynamics

class BodyGeneric(object):
    def __init__(self,name,frame,pCM,vCM,aCM,wNBody,alNBody,mass,inertia,system):
        self.name = name
        self.frame = frame
        self.system = system
        self.pCM = pCM
        self.vCM=vCM
        self.aCM=aCM
        self.mass = mass
        self.inertia= inertia
        
        self.gravityvector = None
        self.forcegravity = None        
        
        self.wNBody = wNBody
        self.alNBody=alNBody
        
        self.effectiveforce = self.mass*self.aCM
        self.momentofeffectiveforce= self.inertia.dot(self.alNBody)+self.wNBody.cross(self.inertia.dot(self.wNBody))
        self.KE = .5*mass*self.vCM.dot(self.vCM) + .5*self.wNBody.dot(self.inertia.dot(self.wNBody))
        self.linearmomentum = self.mass*self.vCM
        self.angularmomentum = self.inertia.dot(self.wNBody)
        
        self.system.bodies.append(self)
        self.adddynamics()
        pynamics.addself(self,name)

    def adddynamics(self):
        self.system.addeffectiveforce(self.effectiveforce,self.vCM)
        self.system.addeffectiveforce(self.momentofeffectiveforce,self.wNBody)
#        self.system.addmomentum(self.linearmomentum,self.vCM)
#        self.system.addmomentum(self.angularmomentum,self.wNBody)
        self.system.addKE(self.KE)

    def addforcegravity(self,gravityvector):
        self.gravityvector = gravityvector
        self.forcegravity = self.mass*self.gravityvector
        self.system.addforce(self.forcegravity,self.vCM)
        
    def __repr__(self):
        return self.name+'(body)'
#        return self.name+' <frame {0:#x}>'.format(self.__hash__())
    def __str__(self):
        return self.name+'(body)'
#        return self.name+' <frame {0:#x}>'.format(self.__hash__())


class Body(object):
    def __init__(self,name,frame,pCM,mass,inertia,system):
        self.name = name
        self.frame = frame
        self.system = system
        self.pCM = pCM
        self.mass = mass
        self.inertia= inertia
        self.vCM=self.pCM.diff_in_parts(self.system.newtonian,self.system)
        self.aCM=self.vCM.diff_in_parts(self.system.newtonian,self.system)
        
        self.gravityvector = None
        self.forcegravity = None        
        
        self.wNBody = self.system.newtonian.getw_(self.frame)
        self.alNBody=self.wNBody.diff_in_parts(self.system.newtonian,self.system)
        
        self.effectiveforce = self.mass*self.aCM
        self.momentofeffectiveforce= self.inertia.dot(self.alNBody)+self.wNBody.cross(self.inertia.dot(self.wNBody))
        self.KE = .5*mass*self.vCM.dot(self.vCM) + .5*self.wNBody.dot(self.inertia.dot(self.wNBody))
#        self.linearmomentum = self.mass*self.vCM
#        self.angularmomentum = self.inertia.dot(self.wNBody)
        
        self.system.bodies.append(self)
        self.adddynamics()
        pynamics.addself(self,name)

    def adddynamics(self):
        self.system.addeffectiveforce(self.effectiveforce,self.vCM)
        self.system.addeffectiveforce(self.momentofeffectiveforce,self.wNBody)
#        self.system.addmomentum(self.linearmomentum,self.vCM)
#        self.system.addmomentum(self.angularmomentum,self.wNBody)
        self.system.addKE(self.KE)

    def addforcegravity(self,gravityvector):
        self.gravityvector = gravityvector
        self.forcegravity = self.mass*self.gravityvector
        self.system.addforce(self.forcegravity,self.vCM)
        
    def __repr__(self):
        return self.name+'(body)'
#        return self.name+' <frame {0:#x}>'.format(self.__hash__())
    def __str__(self):
        return self.name+'(body)'
#        return self.name+' <frame {0:#x}>'.format(self.__hash__())

