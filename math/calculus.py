#!/usr/bin/env python3
import os,sys
import numpy as np

class DegenerateValues:
    def __init__(self,v,tol=None,use_ratio=False):
        self.rat=use_ratio
        if tol is None:
            if self.rat: tol=1.01
            else: tol=0.001
        self.tol=tol
        self.raw_data=v
        self._evaluate()
    
    def _evaluate(self):
        self.argument=dict()
        self.values=dict()
        
        taken=np.zeros_like(self.raw_data).astype(bool)
        while not np.all(taken):
            knew_idx=np.where(~taken)[0][0]
            self.values[self.raw_data[knew_idx]]=[self.raw_data[knew_idx]]
            self.argument[self.raw_data[knew_idx]]=[knew_idx]
            taken[knew_idx]=True
            no_change=False
            
            while not no_change:
                #print("\tLLoop")
                no_change=True
                for idx in np.where(~taken)[0]:
                    #print("\t\t",idx)
                    for k in self.values:
                        
                        for el in self.values[k]:
                            if self.check(el,self.raw_data[idx]):
                                no_change=False
                                self.argument[k].append(idx)
                                self.values[k].append(self.raw_data[idx])
                                taken[idx]=True
                                break
                
    
    def check(self,v1,v2):
        tol=self.tol
        if self.rat:
            return ((np.abs(v1/v2)<tol and np.abs(v1/v2)>1) or (np.abs(v2/v1)<tol and np.abs(v2/v1)>1))
        else: return np.abs(v1-v2)<tol
                            
                
            

class SingleVariableFunction:
    def __init__(self, fx, x_domain=None, analytical_derivative=None, hard_domain=True):
        self.fx=fx
        self.domain=(-np.inf,np.inf) if x_domain is None else (float(x_domain[0]), float(x_domain[1]))
        self.enforce_domain=hard_domain
        _,_,exin=self.practical_domain(16)
        try: res=self.fx(exin).squeeze()
        except: raise ValueError("Function failed to operate on a numpy array of size "+str(len(exin)))
        if len(res)!=len(exin): raise ValueError("Function operating on a list of inputs should give output of same size. Here "+str(len(exin))+" -> "+str(len(res)))

        if analytical_derivative is not None:
            if type(analytical_derivative)!=SingleVariableFunction: self.der=SingleVariableFunction(analytical_derivative, self.domain)
            else: self.der=analytical_derivative
        else:
            self.der=None


    def _direct_call(self,x): return self.fx(x)
    def practical_domain(self, n_points, x_max=None, x_min=None, buffer=0):
        if x_max is None:
            if x_min is not None: x_max=(-x_min if x_min<0 else x_min+1)
            else: x_max=1.0
        if x_min is None: x_min=(-x_max if x_max>0 else x_max-1)
        assert not (np.isinf(x_min) or np.isinf(x_max)), "Cannot have infinite range for a practical domain!"

        ll=self.domain[0]
        if np.isneginf(ll): ll=x_min
        ul=self.domain[1]
        if np.isinf(ul): ul=x_max
        return ll,ul,np.linspace(ll+buffer,ul-buffer,n_points)


    def __call__(self,x, silent=False):
        # Assume x is a numpy array or a single float
        if np.any(x<self.domain[0]).item() or np.any(x>self.domain[1]).item():
            if self.enforce_domain: raise ValueError("Some/All inputs were outside function domain")
            else:
                if not silent: print("WARN: Some inputs may be outside function's declared domain")
        return self._direct_call(x)

    def getDerivative(self, x, silent=False):
        if self.der is None: raise ValueError("Analytical Derivative not provided. To use the numerical derivative call getNumericalDerivative or getAnyDerivative")
        return self.der(x, silent)
    
    def getNumericalDerivative(self, x, step=1e-3, converge=False, max_iter=100, tol=0.01, safe_divide=True): # If converge is set to true, keep reducing step size till derivative converges to 1% (tol) of it's value
        der_est=(self.__call__(x+step,silent=True)-self.__call__(x))/step
        if not converge: return der_est
        
        converged=np.zeros_like(x).astype(bool)
        steps=np.ones_like(x).astype(float)*step
        
        iterno=0
        while not np.all(converged) and iterno<max_iter:
            iterno+=1
            
            der_est2=(self.__call__(x[~converged]+steps[~converged]/2,silent=True)-self.__call__(x[~converged]))/(steps[~converged]/2)
            rel_err=np.abs((der_est2-der_est[~converged])/(der_est[~converged]+(1e-32 if safe_divide else 0)))
            fail=(np.isnan(rel_err) | np.isinf(rel_err) | np.isneginf(rel_err))
            convchk=converged[~converged]
            #print("Lengths",len(convchk),len(rel_err),len(fail))
            convchk[fail | (rel_err<tol)]=True
            
            steps[~converged]/=2
            derchk=der_est[~converged]
            
            derchk[~fail]=der_est2[~fail]
            der_est[~converged]=derchk
            converged[~converged]=convchk
            
        return der_est
    
    def getAnyDerivative(self, x, step=1e-3, converge=False, **kwargs):
        if self.der is not None: return self.getDerivative(x)
        else: return self.getNumericalDerivative(x,step,converge,**kwargs)
    
    def gradientDescent(self, x, alpha=1e-3, tol=1e-4, max_iter=1000, der_step=1e-3, der_tol=None, safe_divide=True): # Descent by gradient till x stops moving by more than tol
        if der_tol is not None:
            convder=True
        else:
            convder=False
        converged=np.zeros_like(x).astype(bool)
        iterno=0
        x_ret=np.ones_like(x).astype(float)*x
        while not np.all(converged) and iterno<max_iter:
            iterno+=1
            filt=(~converged)
            xchk=x_ret[filt]
            derchk=self.getAnyDerivative(xchk, der_step, converge=convder, tol=der_tol, safe_divide=safe_divide)
            xchk-=derchk*alpha
            dx=np.abs((x_ret[filt]-xchk)/(x_ret[filt]+(0 if not safe_divide else 1e-32)))
            fail=(np.isnan(dx) | np.isinf(dx) | np.isneginf(dx))
            convchk=converged[filt] 
            convchk[fail | (dx<tol)]=True # Stop if dx is nan or lower than tolerence
            xchk[fail]=x_ret[filt][fail] # Reset x values if they become nan and stop.
            
            converged[filt]=convchk
            x_ret[filt]=xchk
        return x_ret
