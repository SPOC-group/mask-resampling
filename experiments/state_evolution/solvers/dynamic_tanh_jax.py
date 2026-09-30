#!/usr/bin/env python3
"""JAX GPU dynamic-tanh SE: bounded Beta(1,2) variances on [0.25,2.5]."""
import json
import time
from functools import lru_cache
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j


def normal_rule(n):
    x,w=np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.)*x,w/np.sqrt(np.pi)


def noise_rule(order=200):
    x,w=np.polynomial.legendre.leggauss(order)
    b=.5*(x+1.)
    # db=dx/2; Beta(1,2) density=2*(1-b).
    return .25+2.25*b,w*(1.-b)


@lru_cache(maxsize=16)
def make_channel(quad_order=64, mask_quad_order=64, prox_grid_size=241, prox_newton_steps=35):
    z,w=map(j.asarray,normal_rule(quad_order))
    xi,wm=map(j.asarray,normal_rule(mask_quad_order))
    latent,noise=z[:,None],z[None,:]
    ww=w[:,None]*w[None,:]
    def terms(h,Q,C):
        root=j.sqrt(.1875*j.maximum(C,1e-12))
        y=.25*h[...,None]+root*xi
        t=.75*h[...,None]-root*xi
        s=j.tanh(y); sp=1-s*s; spp=-2*s*sp
        avg=lambda x:j.sum(wm*x,axis=-1)
        loss=avg(.75*Q*s*s-2*s*t)
        dh=avg(2*(.1875*Q*s*sp-.25*t*sp-.75*s))
        dhh=avg(2*(.25**2*.75*Q*(sp*sp+s*spp)-2*.1875*sp-.25**2*t*spp))
        dQ=avg(.75*s*s)
        dC=j.sqrt(.1875/j.maximum(C,1e-12))*avg(xi*(.75*Q*s*sp-t*sp+s))
        return loss,dh,dhh,dQ,dC
    @jax.jit
    def channel(M,Q,C,D):
        h0=M*latent+j.sqrt(j.maximum(C,0))*noise
        scale=j.sqrt(j.maximum(D,1e-12)+j.maximum(C,0)+j.mean(h0*h0)+1)
        candidates=h0[...,None]+j.linspace(-8.,8.,prox_grid_size)*scale
        obj=.5*(candidates-h0[...,None])**2/D+terms(candidates,Q,C)[0]
        h=j.take_along_axis(candidates,j.argmin(obj,axis=-1)[...,None],axis=-1)[...,0]
        def newton(_,h):
            loss,dh,dhh,_,_=terms(h,Q,C)
            grad=(h-h0)/D+dh; hess=1/D+dhh
            step=j.clip(j.where(j.abs(hess)>1e-11,grad/hess,0),-2*scale,2*scale)
            candidate=h-step
            new=.5*(candidate-h0)**2/D+terms(candidate,Q,C)[0]
            old=.5*(h-h0)**2/D+loss
            return j.where(j.isfinite(new)&(new<=old),candidate,h)
        h=jax.lax.fori_loop(0,prox_newton_steps,newton,h)
        force=(h-h0)/D
        _,_,_,dQ,dC=terms(h,Q,C)
        dC=dC-force*noise/(2*j.sqrt(j.maximum(C,1e-12)))
        return j.stack([j.sum(ww*latent*force),j.sum(ww*force**2),j.sum(ww*dQ),j.sum(ww*dC)])
    return channel



REFERENCE_CHECK = {'settings': {'quad_order': 64,
              'mask_quad_order': 64,
              'prox_grid_size': 241,
              'prox_newton_steps': 35},
 'cases': [{'macros': [0.15, 4.5, 4.8, 2.1],
            'moments': [0.07455014576337463,
                        2.179257349650528,
                        0.4531469740206898,
                        0.030862128662392168]},
           {'macros': [0.05, 0.1, 0.1, 1.0],
            'moments': [0.10249860636063332,
                        0.44059313982577564,
                        0.052369952663342975,
                        -0.6170497045574108]}]}

def solve(config, task_id=0):
    alpha=float(config["alpha"]); damping=float(config["damping"])
    quad_order=int(config.get("quad_order",64)); mask_quad_order=int(config.get("mask_quad_order",64))
    prox_grid_size=int(config.get("prox_grid_size",241)); prox_newton_steps=int(config.get("prox_newton_steps",35))
    noise_nodes=int(config.get("noise_nodes",200))
    init_m=float(config.get("init_m",.01)); init_v=float(config.get("init_v",1))
    lambda_r=float(config.get("lambda_r",1e-4)); tol=float(config.get("tol",1e-8)); max_iter=int(config.get("max_iter",160000))
    if float(config.get("beta",1)) != 1 or float(config.get("rho",.75)) != .75:
        raise ValueError("This specialized channel requires beta=1 and rho=0.75")
    if config.get("gamma_law") != "gamma=0.25+2.25*Beta(1,2)":
        raise ValueError("This noise rule requires gamma=0.25+2.25*Beta(1,2)")
    if not 0 < damping <= 1 or noise_nodes < 2 or min(quad_order,mask_quad_order)<2:
        raise ValueError("Invalid damping or quadrature order")
    channel=make_channel(quad_order,mask_quad_order,prox_grid_size,prox_newton_steps)
    expected=dict(quad_order=quad_order, mask_quad_order=mask_quad_order,
                  prox_grid_size=prox_grid_size, prox_newton_steps=prox_newton_steps)
    if expected == REFERENCE_CHECK["settings"]:
        for case in REFERENCE_CHECK["cases"]:
            actual=np.asarray(channel(*case["macros"]))
            np.testing.assert_allclose(actual,case["moments"],rtol=2e-6,atol=2e-8)
    gn,wn=noise_rule(noise_nodes)
    np.testing.assert_allclose([wn.sum(),wn@gn,wn@(gn*gn)],[1,1,1.28125],rtol=1e-12)
    g,w=j.asarray(gn),j.asarray(wn)
    @jax.jit
    def macros(state):
        m,q,v=state
        return j.stack([j.sum(w*m),j.sum(w*q),j.sum(w*g*q),j.sum(w*g*v)])
    @jax.jit
    def step(state):
        M,Q,C,D=macros(state)
        A,F,U,H=channel(M,Q,C,D)
        hm=alpha*A; hq=alpha*g*F; precision=lambda_r+2*alpha*(U+g*H)
        vn=1/precision; raw=j.stack([vn*hm,vn*vn*(hm*hm+hq),vn])
        support_min=j.minimum(lambda_r+2*alpha*(U+.25*H),
                              lambda_r+2*alpha*(U+2.5*H))
        return raw,support_min,j.max(j.abs(raw-state))
    m=j.full_like(g,init_m);v=j.full_like(g,init_v)
    q=4*g*init_v+init_m**2
    state=j.stack([m,q,v]);initial=np.asarray(macros(state)).tolist()
    history=[];start=time.monotonic();status='max_iter';residual=float('nan')
    for iteration in range(max_iter):
        raw,precision,res=step(state)
        precision,residual=map(float,(precision,res))
        values=np.asarray(macros(state))
        history.append([iteration+1,*values,precision,residual])
        if not np.isfinite(precision) or precision<=0:
            status='failed: non-positive or non-finite coordinate precision';residual=float('nan');break
        if not np.isfinite(residual): status='failed: non-finite update';break
        if residual<tol: status='converged';break
        if iteration+1<max_iter: state=(1-damping)*state+damping*raw
    M,Q,C,D=map(float,macros(state))
    row=dict(task_id=task_id,activation='tanh',mask_count='dynamic',alpha=alpha,beta=1,rho=.75,lambda_r=lambda_r,damping=damping,tol=tol,max_iter=max_iter,n_gamma=noise_nodes,gamma_law=config['gamma_law'],gamma_low=.25,gamma_high=2.5,noise_discretization='full_support_gauss_legendre_beta12',init_m=init_m,init_v=init_v,init_q='4*gamma*init_v+init_m**2',initial_macros=json.dumps(initial),status=status,n_iter=iteration+1,se_raw_residual=residual,min_precision=precision,M=M,Q=Q,C=C,Delta=D,cosine=M/np.sqrt(Q),elapsed_seconds=time.monotonic()-start,backend='jax_float64',quad_order=quad_order,mask_quad_order=mask_quad_order,prox_grid_size=prox_grid_size,prox_newton_steps=prox_newton_steps)
    if status=='converged':
        A,F,U,H=map(float,channel(M,Q,C,D))
        support_min=min(lambda_r+2*alpha*(U+.25*H),lambda_r+2*alpha*(U+2.5*H))
        row.update(A=A,F=F,U=U,H=H,precision_support_infimum=support_min,
                   full_support_admissible=bool(np.isfinite(support_min) and support_min>0))
    return row
