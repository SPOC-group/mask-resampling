"""Original replica-symmetric local-stability solver, extracted from the solver notebook.

Numerical definitions and activation registrations are unchanged. Only the unused
plotting import and notebook display/configuration cells have been omitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist
from numpy.polynomial.hermite import hermgauss

from scipy.special import ndtri
from scipy.stats import qmc

EPS = 1e-12
SQRT_2PI = np.sqrt(2.0 * np.pi)


def generalized_gamma_grid(
    s: float,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
    n_points: int = 200,
    tail_eps: float = 1e-5,
    renormalize_lognormal_mean: bool = True,
):
    """Legacy truncated quantile approximation; never use for production."""
    if s < 0:
        raise ValueError("s must be non-negative")
    if eta_n < 0 or eta_h < 0:
        raise ValueError("eta_n and eta_h must be non-negative")
    if not (0 <= tail_eps < 0.5):
        raise ValueError("tail_eps must satisfy 0 <= tail_eps < 0.5")

    if eta_h == 0 or s == 0:
        gamma0 = eta_n + eta_h
        if gamma0 <= 0:
            raise ValueError("gamma must be strictly positive")
        return np.array([float(gamma0)]), np.array([1.0])

    normal = NormalDist()
    probs = tail_eps + (1 - 2 * tail_eps) * (np.arange(n_points) + 0.5) / n_points
    g = np.array([normal.inv_cdf(float(p)) for p in probs])
    weights = np.ones(n_points, dtype=float) / n_points

    lognormal = np.exp(s * g - 0.5 * s**2)
    if renormalize_lognormal_mean:
        lognormal /= np.sum(weights * lognormal)

    gamma = eta_n + eta_h * lognormal
    if np.any(gamma <= 0):
        raise ValueError("all gamma values must be positive")
    return gamma.astype(float), weights.astype(float)


def generalized_gamma_hermite_grid(
    s: float,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
    order: int = 40,
):
    """Gauss-Hermite discretization of the *true* unbounded variance law.

    Unlike the legacy equal-weight quantile grid, this integrates lognormal
    moments accurately even for large s. The largest gamma nodes can be very
    large; this is desirable because it also exposes violations of the regularity
    condition for an unbounded-support law.
    """
    x,w=hermgauss(int(order))
    g=np.sqrt(2.0)*x
    weights=w/np.sqrt(np.pi)
    gamma=eta_n+eta_h*np.exp(s*g-0.5*s**2)
    return gamma.astype(float),weights.astype(float)


def gamma_kappa2_exact(s, eta_n=0.75, eta_h=0.25):
    return float(eta_n**2 + 2 * eta_n * eta_h + eta_h**2 * np.exp(s**2))


def gamma_support_is_unbounded(s, eta_h):
    return bool(float(s) > 0.0 and float(eta_h) > 0.0)


def full_support_regularity_margin(tau, *, unbounded, gamma_lower, gamma_upper=None):
    """Infimum of 1+tau*gamma over the actual support of P_gamma."""
    tau = float(tau)
    if not np.isfinite(tau):
        return np.nan
    tolerance = 1e-12
    if tau < -tolerance:
        if unbounded:
            return -np.inf
        if gamma_upper is None:
            raise ValueError("bounded support requires gamma_upper")
        return float(1.0 + tau * float(gamma_upper))
    if abs(tau) <= tolerance:
        return 1.0
    return float(1.0 + tau * float(gamma_lower))


def noise_summary(gamma, weights, s=None, eta_n=None, eta_h=None):
    out = {
        "n_gamma": len(gamma),
        "E[gamma]": float(np.sum(weights * gamma)),
        "E[gamma^2] grid": float(np.sum(weights * gamma**2)),
        "E[gamma^-1]": float(np.sum(weights / gamma)),
        "E[gamma^-2]": float(np.sum(weights / gamma**2)),
        "gamma_min_grid": float(np.min(gamma)),
        "gamma_max_grid": float(np.max(gamma)),
    }
    if s is not None:
        exact_kappa2 = gamma_kappa2_exact(s, eta_n, eta_h)
        out["E[gamma^2] exact"] = exact_kappa2
        out["E[gamma^2] relative error"] = (out["E[gamma^2] grid"] - exact_kappa2) / exact_kappa2
    return pd.Series(out)


def alpha_c_bayes(beta, gamma, weights):
    return float(1.0 / (beta**2 * np.sum(weights / gamma**2)))


def R_gamma(tau, gamma, weights):
    den = 1.0 + float(tau) * gamma
    if np.any(den <= 0):
        return np.nan
    num = np.sum(weights * gamma**2 / den**2)
    d0 = np.sum(weights / den)
    return float(num / d0**2)


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    fn: object
    d1: object
    d2: object
    sigma0: float = np.nan
    slope0: float = np.nan
    smooth_at_zero: bool = True


ACTIVATIONS = {}


def register_activation(name, fn, d1, d2, sigma0=np.nan, slope0=np.nan, smooth_at_zero=True):
    key = str(name).lower()
    ACTIVATIONS[key] = ActivationSpec(
        key, fn, d1, d2,
        float(sigma0) if np.isfinite(sigma0) else np.nan,
        float(slope0) if np.isfinite(slope0) else np.nan,
        bool(smooth_at_zero),
    )
    return ACTIVATIONS[key]


def get_activation_spec(activation):
    if isinstance(activation, ActivationSpec):
        return activation
    key = str(activation).lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}; available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]


def _linear(x):
    return np.asarray(x, dtype=float)

def _linear_d1(x):
    return np.ones_like(np.asarray(x, dtype=float))

def _linear_d2(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def _relu(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)

def _relu_d1(x):
    return (np.asarray(x, dtype=float) > 0).astype(float)

def _relu_d2(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def _tanh(x):
    return np.tanh(np.asarray(x, dtype=float))

def _tanh_d1(x):
    t = np.tanh(np.asarray(x, dtype=float))
    return 1 - t*t

def _tanh_d2(x):
    t = np.tanh(np.asarray(x, dtype=float))
    return -2*t*(1-t*t)


def _elu(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))

def _elu_d1(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))

def _elu_d2(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 0.0, np.exp(np.minimum(x, 0.0)))


register_activation("linear", _linear, _linear_d1, _linear_d2, sigma0=0.0, slope0=1.0)
register_activation("relu", _relu, _relu_d1, _relu_d2, sigma0=0.0, smooth_at_zero=False)
register_activation("tanh", _tanh, _tanh_d1, _tanh_d2, sigma0=0.0, slope0=1.0)
register_activation("elu", _elu, _elu_d1, _elu_d2, sigma0=0.0, slope0=1.0)


def gh_standard_normal(order=21):
    x, w = hermgauss(int(order))
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


@lru_cache(maxsize=None)
def _sobol_standard_normals_cached(dim, sobol_power, seed):
    sampler = qmc.Sobol(d=int(dim), scramble=True, seed=int(seed))
    u = sampler.random_base2(m=int(sobol_power))
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return ndtri(u)


def sobol_standard_normals(dim, sobol_power=8, seed=1234):
    return _sobol_standard_normals_cached(int(dim), int(sobol_power), int(seed))


def finite_k_loss_grad_hess(g, Q, rho, activation, need_hess=True):
    spec = get_activation_spec(activation)
    g = np.asarray(g, dtype=float)
    if g.ndim != 2 or g.shape[1] < 2:
        raise ValueError("g must have shape (n_samples,K+1)")

    h = g[:, 0]
    x = g[:, 1:]
    K = x.shape[1]
    a = 1.0 - rho
    b = rho
    c = a*b
    sc = np.sqrt(c)

    y = a*h[:, None] + sc*x
    t = b*h[:, None] - sc*x
    sig = spec.fn(y)
    sp = spec.d1(y)

    ell = np.mean(b*Q*sig**2 - 2*sig*t, axis=1)
    U = 2*(a*t*sp + b*sig - a*b*Q*sig*sp)
    W = 2*sc*(t*sp - sig - b*Q*sig*sp)

    grad = np.empty_like(g)
    grad[:, 0] = -np.mean(U, axis=1)
    grad[:, 1:] = -W/K
    if not need_hess:
        return ell, grad, None

    spp = spec.d2(y)
    h_hh = 2*(a*a*b*Q*(sp**2 + sig*spp) - 2*a*b*sp - a*a*t*spp)
    h_xx = 2*c*(b*Q*(sp**2 + sig*spp) + 2*sp - t*spp)
    h_hx = 2*sc*(a*b*Q*(sp**2 + sig*spp) + (a-b)*sp - a*t*spp)

    n = g.shape[0]
    D = K+1
    H = np.zeros((n,D,D), dtype=float)
    H[:,0,0] = np.mean(h_hh, axis=1)
    H[:,0,1:] = h_hx/K
    H[:,1:,0] = h_hx/K
    idx = np.arange(K)
    H[:,1+idx,1+idx] = h_xx/K
    return ell, grad, H


def finite_k_smooth_prox(g0, Q, Delta, rho, activation, max_steps=35, tol=1e-9):
    """Generic multistart Newton/line-search Moreau proximal for smooth activations."""
    g0 = np.asarray(g0, dtype=float)
    n, D = g0.shape
    starts = [g0.copy(), 0.5*g0, np.zeros_like(g0)]
    best_g, best_obj = None, None
    diag = np.arange(D)

    for start in starts:
        g = start.copy()
        for _ in range(int(max_steps)):
            ell, grad_loss, H_loss = finite_k_loss_grad_hess(g,Q,rho,activation,True)
            grad = (g-g0)/Delta + grad_loss
            H = H_loss.copy()
            H[:,diag,diag] += 1.0/Delta
            try:
                step = np.linalg.solve(H, grad[...,None])[...,0]
            except np.linalg.LinAlgError:
                step = Delta*grad

            directional = np.sum(grad*step, axis=1)
            bad = directional <= 1e-12
            step[bad] = Delta*grad[bad]
            norm = np.linalg.norm(step, axis=1)
            cap = 4*np.sqrt(Delta+1)
            step *= np.minimum(1.0, cap/np.maximum(norm,1e-12))[:,None]

            obj = 0.5*np.sum((g-g0)**2,axis=1)/Delta + ell
            accepted = np.zeros(n,dtype=bool)
            alpha_ls = np.ones(n)
            g_new = g.copy()
            for _ in range(14):
                cand = g - alpha_ls[:,None]*step
                ell_c,_,_ = finite_k_loss_grad_hess(cand,Q,rho,activation,False)
                obj_c = 0.5*np.sum((cand-g0)**2,axis=1)/Delta + ell_c
                ok = (~accepted) & np.isfinite(obj_c) & (obj_c <= obj + 1e-12)
                g_new[ok] = cand[ok]
                accepted |= ok
                alpha_ls[~accepted] *= 0.5
                if accepted.all():
                    break
            change = np.max(np.abs(g_new-g))
            g = g_new
            if change < tol:
                break

        ell,_,_ = finite_k_loss_grad_hess(g,Q,rho,activation,False)
        obj = 0.5*np.sum((g-g0)**2,axis=1)/Delta + ell
        if best_g is None:
            best_g, best_obj = g.copy(), obj.copy()
        else:
            improve = obj < best_obj
            best_g[improve] = g[improve]
            best_obj[improve] = obj[improve]
    return best_g


def _relu_active_quadratic_coeff(x0, Q, Delta, rho, K):
    a = 1-rho; b = rho; c = a*b; sc = np.sqrt(c)
    r0 = sc*float(x0)
    Dcoef = b*float(Q) + 2.0
    lin_h = a/(Delta*c) + 2.0/K
    curv_y = 1.0/(Delta*c) + 2.0*Dcoef/K
    A = a*a/(Delta*c) - lin_h*lin_h/curv_y
    B = a*r0/(Delta*c) - lin_h*(r0/(Delta*c))/curv_y
    C0 = r0*r0/(2*Delta*c) - (r0/(Delta*c))**2/(2*curv_y)
    return A,B,C0,lin_h,curv_y


def _relu_conditional_y(h, x0, Q, Delta, rho, K):
    a = 1-rho; b = rho; c = a*b; sc = np.sqrt(c)
    x0 = np.asarray(x0,dtype=float)
    d = a*float(h) + sc*x0
    y_neg = np.minimum(d,0.0)
    val_neg = (y_neg-d)**2/(2*Delta*c)
    Dcoef = b*float(Q)+2.0
    curv_y = 1.0/(Delta*c) + 2.0*Dcoef/K
    lin = d/(Delta*c) + 2.0*float(h)/K
    y_pos = np.maximum(lin/curv_y,0.0)
    val_pos = (y_pos-d)**2/(2*Delta*c) + (Dcoef*y_pos**2 - 2*float(h)*y_pos)/K
    use_pos = val_pos < val_neg
    return np.where(use_pos,y_pos,y_neg)


def finite_k_relu_prox_single(g0,Q,Delta,rho):
    g0 = np.asarray(g0,dtype=float)
    K = g0.size-1
    a = 1-rho; b = rho; c = a*b; sc = np.sqrt(c)
    h0 = float(g0[0]); x0 = g0[1:]

    A_tot = 1.0/Delta
    B_tot = -h0/Delta
    C_tot = h0*h0/(2*Delta)
    events=[]

    for xr in x0:
        A_act,B_act,C_act,lin_h,curv_y = _relu_active_quadratic_coeff(xr,Q,Delta,rho,K)
        r0 = sc*float(xr)
        h_d = -r0/a
        h_l = -(r0/(Delta*c))/lin_h
        if xr >= 0:
            A_zero = a*a/(Delta*c)
            B_zero = a*r0/(Delta*c)
            C_zero = r0*r0/(2*Delta*c)
            events.append((h_d,A_zero,B_zero,C_zero))
            events.append((h_l,A_act-A_zero,B_act-B_zero,C_act-C_zero))
        else:
            ratio = np.sqrt(curv_y/(Delta*c))
            h_cross = -r0*(1.0/(Delta*c)+ratio)/(lin_h+a*ratio)
            events.append((h_cross,A_act,B_act,C_act))

    events.sort(key=lambda e:e[0])
    best_val=np.inf; best_h=None; left=-np.inf

    def consider(h,A,B,C0):
        nonlocal best_val,best_h
        val=0.5*A*h*h+B*h+C0
        if np.isfinite(val) and val<best_val:
            best_val=float(val); best_h=float(h)

    i=0
    while i<len(events):
        right=float(events[i][0])
        if A_tot>1e-14:
            hs=-B_tot/A_tot
            if left<hs<right:
                consider(hs,A_tot,B_tot,C_tot)
        consider(right,A_tot,B_tot,C_tot)
        j=i
        while j<len(events) and abs(events[j][0]-right)<=1e-12*(1+abs(right)):
            _,dA,dB,dC=events[j]
            A_tot+=dA; B_tot+=dB; C_tot+=dC
            j+=1
        left=right; i=j

    if A_tot < -1e-12 or (abs(A_tot)<=1e-12 and B_tot < -1e-12):
        raise FloatingPointError("unbounded finite-K ReLU Moreau profile")
    if A_tot>1e-14:
        hs=-B_tot/A_tot
        if hs>left:
            consider(hs,A_tot,B_tot,C_tot)
    if best_h is None:
        raise FloatingPointError("could not locate finite ReLU proximal")

    ystar=_relu_conditional_y(best_h,x0,Q,Delta,rho,K)
    xstar=(ystar-a*best_h)/sc
    return np.concatenate([[best_h],xstar])


def finite_k_relu_prox(g0,Q,Delta,rho):
    g0=np.asarray(g0,dtype=float)
    out=np.empty_like(g0)
    for i in range(g0.shape[0]):
        out[i]=finite_k_relu_prox_single(g0[i],Q,Delta,rho)
    return out


def finite_k_linear_zero_channel(Q,C,Delta,rho,K):
    K=int(K)
    a=1-rho; b=rho; c=a*b
    A=c*(a*Q-2.0)
    B=a-b+c*Q
    Gcoef=b*Q+2.0

    S=2*np.array([[A,B*np.sqrt(c/K)], [B*np.sqrt(c/K),c*Gcoef/K]],dtype=float)
    P=np.eye(2)+Delta*S
    eig=np.linalg.eigvalsh(P)
    s_perp=2*c*Gcoef/K
    dperp=1+Delta*s_perp
    if eig[0]<=0 or dperp<=0:
        raise FloatingPointError("unstable finite-K linear Moreau branch")

    R=np.linalg.solve(P,np.eye(2))
    L=-S@R
    rperp=1/dperp
    lperp=-s_perp*rperp

    chi=float(L[0,0])
    E=float(C*np.sum(L**2)+(K-1)*C*lperp**2)

    ry=np.array([a,np.sqrt(c/K)])
    mean_y2=float(C*np.dot(R.T@ry,R.T@ry)+(K-1)*(c/K)*C*rperp**2)
    U=float(b*mean_y2)
    H=float(-0.5*(np.trace(L)+(K-1)*lperp))
    return {"E":E,"U":U,"H":H,"chi":chi,"solver":"linear_exact"}


def dynamic_linear_zero_channel(Q,C,Delta,rho):
    a=1-rho; b=rho; c=a*b
    a_loss=c*(a*Q-2.0)
    b_loss=c*(b*Q+2.0)
    D=1+2*a_loss*Delta
    if D<=0:
        raise FloatingPointError("unstable dynamic linear Moreau branch")
    chi=-2*a_loss/D
    J=C/D**2
    E=chi**2*C
    U=c*(a*J+b*C)
    H=a_loss/D+b_loss
    return {"E":float(E),"U":float(U),"H":float(H),"chi":float(chi),"solver":"linear_exact"}


def dynamic_loss_terms(h,Q,C,rho,activation,mask_quad_order=21):
    spec=get_activation_spec(activation)
    h=np.asarray(h,dtype=float)
    a=1-rho; b=rho; c=a*b
    Ceff=max(float(C),EPS)
    xi,wx=gh_standard_normal(mask_quad_order)
    root=np.sqrt(c*Ceff)

    H=h[...,None]
    y=a*H+root*xi
    t=b*H-root*xi
    sig=spec.fn(y); sp=spec.d1(y); spp=spec.d2(y)

    ell=np.sum(wx*(b*Q*sig**2-2*sig*t),axis=-1)
    dQ=np.sum(wx*(b*sig**2),axis=-1)
    dC=np.sqrt(c/Ceff)*np.sum(wx*xi*(b*Q*sig*sp-t*sp+sig),axis=-1)
    d1=np.sum(wx*2*(a*b*Q*sig*sp-a*t*sp-b*sig),axis=-1)
    d2=np.sum(wx*2*(a*a*b*Q*(sp**2+sig*spp)-2*a*b*sp-a*a*t*spp),axis=-1)
    return ell,d1,d2,dQ,dC


def dynamic_prox(h0,Q,C,Delta,rho,activation,mask_quad_order=21,grid_size=161,newton_steps=25):
    h0=np.asarray(h0,dtype=float)
    scale=np.sqrt(max(Delta,EPS)+max(C,0.0)+float(np.mean(h0**2))+1.0)
    offsets=np.linspace(-8*scale,8*scale,int(grid_size))
    H=h0[...,None]+offsets
    ell,_,_,_,_=dynamic_loss_terms(H,Q,C,rho,activation,mask_quad_order)
    obj=0.5*(H-h0[...,None])**2/Delta+ell
    idx=np.argmin(obj,axis=-1)
    h=np.take_along_axis(H,idx[...,None],axis=-1)[...,0]

    for _ in range(int(newton_steps)):
        ell,d1,d2,_,_=dynamic_loss_terms(h,Q,C,rho,activation,mask_quad_order)
        grad=(h-h0)/Delta+d1
        hess=1.0/Delta+d2
        step=np.where(np.abs(hess)>1e-11,grad/hess,0.0)
        step=np.clip(step,-2*scale,2*scale)
        cand=h-step
        ellc,_,_,_,_=dynamic_loss_terms(cand,Q,C,rho,activation,mask_quad_order)
        objc=0.5*(cand-h0)**2/Delta+ellc
        objh=0.5*(h-h0)**2/Delta+ell
        h=np.where(np.isfinite(objc)&(objc<=objh),cand,h)
    return h


def dynamic_generic_zero_channel(
    Q,C,Delta,rho,activation,
    outer_quad_order=21,
    mask_quad_order=21,
    prox_grid_size=161,
    prox_newton_steps=25,
):
    if str(activation).lower()=="linear":
        return dynamic_linear_zero_channel(Q,C,Delta,rho)

    z,w=gh_standard_normal(outer_quad_order)
    sqrtC=np.sqrt(max(C,EPS))
    h0=sqrtC*z
    hstar=dynamic_prox(
        h0,Q,C,Delta,rho,activation,
        mask_quad_order=mask_quad_order,
        grid_size=prox_grid_size,
        newton_steps=prox_newton_steps,
    )
    F=(hstar-h0)/Delta
    _,_,_,dQ,dC=dynamic_loss_terms(hstar,Q,C,rho,activation,mask_quad_order)

    E=float(np.sum(w*F**2))
    U=float(np.sum(w*dQ))
    H=float(np.sum(w*(dC-F*z/(2*sqrtC))))
    chi=float(np.sum(w*z*F)/sqrtC)
    return {"E":E,"U":U,"H":H,"chi":chi,"solver":"dynamic_generic"}


def finite_k_generic_zero_channel(
    Q,C,Delta,rho,K,activation,
    sobol_power=8,
    qmc_seed=1234,
    prox_steps=35,
    prox_tol=1e-9,
):
    K=int(K)
    spec=get_activation_spec(activation)
    if spec.name=="linear":
        return finite_k_linear_zero_channel(Q,C,Delta,rho,K)

    sqrtC=np.sqrt(max(C,EPS))
    Z=sobol_standard_normals(K+1,sobol_power,qmc_seed)
    g0=sqrtC*Z

    if spec.name=="relu":
        gstar=finite_k_relu_prox(g0,Q,Delta,rho)
        solver="relu_exact_piecewise"
    else:
        gstar=finite_k_smooth_prox(g0,Q,Delta,rho,spec,max_steps=prox_steps,tol=prox_tol)
        solver="smooth_generic"

    F=(gstar-g0)/Delta
    a=1-rho; c=rho*a
    y=a*gstar[:,[0]]+np.sqrt(c)*gstar[:,1:]
    sig=spec.fn(y)

    E=float(np.mean(np.sum(F**2,axis=1)))
    U=float(np.mean(rho*np.mean(sig**2,axis=1)))
    H=float(np.mean(-np.sum(F*Z,axis=1)/(2*sqrtC)))
    chi=float(np.mean(Z[:,0]*F[:,0])/sqrtC)
    return {"E":E,"U":U,"H":H,"chi":chi,"solver":solver,"n_qmc":len(Z)}


def zero_output_channel(Q,C,Delta,rho,K,activation,numerics):
    if np.isinf(K):
        return dynamic_generic_zero_channel(
            Q,C,Delta,rho,activation,
            outer_quad_order=numerics.get("dynamic_outer_quad",21),
            mask_quad_order=numerics.get("dynamic_mask_quad",21),
            prox_grid_size=numerics.get("dynamic_grid",161),
            prox_newton_steps=numerics.get("dynamic_newton",25),
        )
    return finite_k_generic_zero_channel(
        Q,C,Delta,rho,int(K),activation,
        sobol_power=numerics.get("sobol_power",8),
        qmc_seed=numerics.get("qmc_seed",1234),
        prox_steps=numerics.get("finite_prox_steps",35),
        prox_tol=numerics.get("finite_prox_tol",1e-9),
    )


def initial_zero_state(alpha,rho,gamma,weights):
    """Simple initializer inherited from the old SE notebook, but with M=0 exactly."""
    k2=float(np.sum(weights*gamma**2))
    branch=2*alpha*k2-1
    if branch>0:
        Delta0=1/(2*np.sqrt(max(rho*(1-rho)*branch,1e-10)))
    else:
        Delta0=1.0
    v=np.full_like(gamma,Delta0,dtype=float)
    q=4*gamma*Delta0
    return q,v


def solve_zero_saddle(
    alpha,rho,K,activation,gamma,weights,
    numerics=None,
    init_state=None,
    ridge=0.0,
    gamma_support_unbounded=False,
    gamma_lower_bound=None,
    damping=0.30,
    max_iter=4000,
    tol=1e-8,
):
    """Solve the M=0 branch and return the local gain per unit beta.

    The threshold at signal beta satisfies gain_per_beta = 1/beta.
    """
    if numerics is None:
        numerics={}
    if not (0<rho<1):
        raise ValueError("rho must satisfy 0<rho<1")
    if alpha<=0:
        raise ValueError("alpha must be positive")

    if init_state is None:
        q,v=initial_zero_state(alpha,rho,gamma,weights)
    else:
        q=np.asarray(init_state[0],dtype=float).copy()
        v=np.asarray(init_state[1],dtype=float).copy()

    status="max_iter"
    residual=np.inf
    channel={}

    for it in range(int(max_iter)):
        Q=float(np.sum(weights*q))
        C=float(np.sum(weights*gamma*q))
        Delta=float(np.sum(weights*gamma*v))
        if min(Q,C,Delta)<=0 or not np.all(np.isfinite([Q,C,Delta])):
            status="failed: non-positive macro"
            break

        try:
            channel=zero_output_channel(Q,C,Delta,rho,K,activation,numerics)
        except Exception as err:
            status=f"failed output channel: {err}"
            break

        U_iter=float(channel["U"])
        H_iter=float(channel["H"])
        support_tol=1e-12*(1.0+abs(U_iter)+abs(H_iter))
        if gamma_support_unbounded and H_iter < -support_tol:
            status=("failed: unbounded gamma support is incompatible with "
                    f"negative H={H_iter:.6e}")
            break

        hat_v=ridge+2*alpha*(channel["U"]+gamma*channel["H"])
        if np.any(~np.isfinite(hat_v)) or np.min(hat_v)<=0:
            status="failed: non-positive input curvature"
            break

        v_raw=1/hat_v
        q_raw=v_raw**2*(alpha*gamma*channel["E"])

        rq=np.max(np.abs(q_raw-q)/(1+np.abs(q)))
        rv=np.max(np.abs(v_raw-v)/(1+np.abs(v)))
        residual=float(max(rq,rv))

        q=(1-damping)*q+damping*q_raw
        v=(1-damping)*v+damping*v_raw

        if residual<tol:
            status="converged"
            break

    # Final diagnostics at the actual final state.
    Q=float(np.sum(weights*q))
    C=float(np.sum(weights*gamma*q))
    Delta=float(np.sum(weights*gamma*v))
    Ev=float(np.sum(weights*v))
    try:
        channel=zero_output_channel(Q,C,Delta,rho,K,activation,numerics)
    except Exception as err:
        if status=="converged":
            status=f"failed final channel: {err}"

    chi=float(channel.get("chi",np.nan))
    gain_per_beta=float(alpha*chi*Ev) if np.isfinite(chi) else np.nan
    U=float(channel.get("U",np.nan))
    H=float(channel.get("H",np.nan))
    E=float(channel.get("E",np.nan))
    tau=float(H/U) if np.isfinite(U) and abs(U)>EPS else np.nan
    eta2=float(E/(C*chi**2)) if C>0 and np.isfinite(chi) and abs(chi)>EPS else np.nan
    if gamma_lower_bound is None:
        gamma_lower_bound=float(np.min(gamma))
    support_margin=full_support_regularity_margin(
        tau, unbounded=bool(gamma_support_unbounded),
        gamma_lower=gamma_lower_bound, gamma_upper=float(np.max(gamma))
    )
    if status=="converged" and not (support_margin > 0):
        status="failed: full-support regularity condition violated"

    return {
        "alpha":float(alpha), "rho":float(rho), "K":K,
        "activation":get_activation_spec(activation).name,
        "status":status, "n_iter":it+1, "residual":residual,
        "Q":Q, "C":C, "Delta":Delta, "Ev":Ev,
        "E":E, "U":U, "H":H, "chi":chi,
        "tau":tau, "eta2":eta2,
        "gain_per_beta":gain_per_beta,
        "regularity_grid_min":float(np.min(1+tau*gamma)) if np.isfinite(tau) else np.nan,
        "regularity_support_infimum":support_margin,
        "gamma_support_unbounded":bool(gamma_support_unbounded),
        "state":(q,v),
        "solver":channel.get("solver","unknown"),
    }


def weak_signal_alpha_reference(beta,rho,K,activation,kappa2):
    spec=get_activation_spec(activation)
    if spec.name=="relu":
        D=(rho**2*(1-rho)**2*(1+2/np.pi*np.arcsin(1-rho))
           +rho*(1-rho)/(2*np.pi)*np.sqrt(rho/(2-rho)))
        base=D/(rho**2*(1-rho)**2)
        ratio=base if np.isinf(K) else base+(1/int(K))*(1/(rho*(1-rho))-base)
        return float(kappa2/beta**2*ratio)

    if (np.isfinite(spec.sigma0) and abs(spec.sigma0)<1e-12 and
        np.isfinite(spec.slope0) and abs(spec.slope0)>0 and spec.smooth_at_zero):
        corr=1.0 if np.isinf(K) else 1+(1-2*rho*(1-rho))/(2*int(K)*rho*(1-rho))
        return float(kappa2/beta**2*corr)
    return np.nan


def default_alpha_bounds(beta,rho,K,activation,gamma,weights):
    k2=float(np.sum(weights*gamma**2))
    ref=weak_signal_alpha_reference(beta,rho,K,activation,k2)
    if not np.isfinite(ref):
        ref=k2/beta**2
    return max(ref/30,1e-3), ref*30


def scan_zero_branch(alpha_grid,rho,K,activation,gamma,weights,numerics=None,**solver_kwargs):
    rows=[]
    state=None
    for alpha in np.asarray(alpha_grid,dtype=float):
        res=solve_zero_saddle(
            alpha,rho,K,activation,gamma,weights,
            numerics=numerics,init_state=state,**solver_kwargs
        )
        if res["status"]=="converged":
            state=res["state"]
        row={k:v for k,v in res.items() if k!="state"}
        rows.append(row)
    return pd.DataFrame(rows)


def critical_alpha(
    beta,rho,K,activation,gamma,weights,
    numerics=None,
    alpha_bounds=None,
    n_scan=28,
    bisect_steps=18,
    scan_descending=False,
    **solver_kwargs,
):
    if beta<=0:
        raise ValueError("beta must be positive")
    if alpha_bounds is None:
        alpha_bounds=default_alpha_bounds(beta,rho,K,activation,gamma,weights)
    amin,amax=map(float,alpha_bounds)
    if not (0<amin<amax):
        raise ValueError("alpha_bounds must satisfy 0<min<max")

    alpha_grid=np.geomspace(amin,amax,int(n_scan))
    if scan_descending:
        alpha_grid=alpha_grid[::-1]
    curve=scan_zero_branch(
        alpha_grid,rho,K,activation,gamma,weights,
        numerics=numerics,**solver_kwargs
    )
    target=1.0/beta

    valid=curve[curve["status"]=="converged"].copy()
    if len(valid)<2:
        return {"status":"no usable zero branch","alpha_c":np.nan,"curve":curve}

    vals=valid["gain_per_beta"].to_numpy()-target
    alphas=valid["alpha"].to_numpy()
    brackets=[]
    for i in range(len(valid)-1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1]<=0:
            brackets.append((i,i+1))
    if not brackets:
        return {
            "status":"no crossing in alpha_bounds",
            "alpha_c":np.nan,
            "curve":curve,
            "gain_min":float(np.nanmin(valid["gain_per_beta"])),
            "gain_max":float(np.nanmax(valid["gain_per_beta"])),
            "target_gain":target,
        }

    i,j=brackets[0]
    aL=float(alphas[i]); aR=float(alphas[j])
    # Re-solve endpoints independently to retain the states for warm starts.
    rL=solve_zero_saddle(aL,rho,K,activation,gamma,weights,numerics=numerics,**solver_kwargs)
    rR=solve_zero_saddle(aR,rho,K,activation,gamma,weights,numerics=numerics,init_state=rL["state"],**solver_kwargs)
    fL=rL["gain_per_beta"]-target
    fR=rR["gain_per_beta"]-target

    for _ in range(int(bisect_steps)):
        amid=np.sqrt(aL*aR)
        init=rL["state"] if abs(np.log(amid/aL))<=abs(np.log(aR/amid)) else rR["state"]
        rm=solve_zero_saddle(amid,rho,K,activation,gamma,weights,numerics=numerics,init_state=init,**solver_kwargs)
        if rm["status"]!="converged" or not np.isfinite(rm["gain_per_beta"]):
            # If midpoint fails, stop rather than silently jumping branches.
            break
        fm=rm["gain_per_beta"]-target
        if fL*fm<=0:
            aR,rR,fR=amid,rm,fm
        else:
            aL,rL,fL=amid,rm,fm

    rc=rL if abs(fL)<=abs(fR) else rR
    alpha_c=float(rc["alpha"])
    tau=rc["tau"]
    eta2=rc["eta2"]
    rg=R_gamma(tau,gamma,weights)
    alpha_fact=float(eta2*rg/beta**2) if np.isfinite(eta2) and np.isfinite(rg) else np.nan
    k2=float(np.sum(weights*gamma**2))
    weak=weak_signal_alpha_reference(beta,rho,K,activation,k2)

    return {
        "status":"converged",
        "alpha_c":alpha_c,
        "beta":float(beta),"rho":float(rho),"K":K,
        "activation":get_activation_spec(activation).name,
        "gain":float(beta*rc["gain_per_beta"]),
        "chi_c":float(rc["chi"]),"Ev_c":float(rc["Ev"]),
        "tau_c":float(tau),"eta2_c":float(eta2),
        "Rgamma_c":float(rg),
        "alpha_factorized":alpha_fact,
        "factorization_relerr":float(abs(alpha_fact-alpha_c)/alpha_c) if np.isfinite(alpha_fact) else np.nan,
        "alpha_weak_reference":float(weak) if np.isfinite(weak) else np.nan,
        "alpha_BO":alpha_c_bayes(beta,gamma,weights),
        "regularity_grid_min":float(rc["regularity_grid_min"]),
        "regularity_support_infimum":float(rc["regularity_support_infimum"]),
        "gamma_support_unbounded":bool(rc["gamma_support_unbounded"]),
        "zero_saddle":rc,
        "curve":curve,
    }


def threshold_row(result):
    keys=[
        "activation","K","rho","beta","alpha_c","alpha_weak_reference","alpha_BO",
        "tau_c","eta2_c","chi_c","Ev_c","alpha_factorized","factorization_relerr",
        "regularity_grid_min","regularity_support_infimum",
        "gamma_support_unbounded","status"
    ]
    return {k:result.get(k,np.nan) for k in keys}


def thresholds_from_curve(curve, betas):
    valid=curve[curve["status"]=="converged"].sort_values("alpha")
    a=valid["alpha"].to_numpy(float)
    g=valid["gain_per_beta"].to_numpy(float)
    rows=[]
    for beta in np.atleast_1d(betas).astype(float):
        target=1/beta
        alpha=np.nan
        for i in range(len(a)-1):
            f0=g[i]-target; f1=g[i+1]-target
            if np.isfinite(f0) and np.isfinite(f1) and f0*f1<=0:
                # Interpolate linearly in log(alpha) across the local bracket.
                if abs(g[i+1]-g[i])<EPS:
                    alpha=np.sqrt(a[i]*a[i+1])
                else:
                    t=(target-g[i])/(g[i+1]-g[i])
                    alpha=np.exp((1-t)*np.log(a[i])+t*np.log(a[i+1]))
                break
        rows.append({"beta":beta,"alpha_c_interp":alpha,"target_gain_per_beta":target})
    return pd.DataFrame(rows)

