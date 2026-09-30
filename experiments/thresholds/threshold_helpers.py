"""Original quadrature adapter and numerical settings."""
import numpy as np
from gamma_distributions import bounded_heterogenous_quadrature, bounded_heterogenous_metadata

def law_grid(name,order,target_k2):
    if name=='homogeneous':
        gamma=np.array([1.]); weights=np.array([1.])
        meta=dict(gamma_formula='gamma=1',gamma_low=1.,gamma_high=1.,
                  gamma_mean_exact=1.,gamma_second_moment_exact=1.,quadrature='point_mass')
    else:
        gamma,weights=bounded_heterogenous_quadrature(order)
        meta=bounded_heterogenous_metadata()
        meta['quadrature']='full_support_gauss_legendre'
        if name=='bounded_matched_lognormal_moments':
            scale=np.sqrt((target_k2-1)/(.28125))
            gamma=1+scale*(gamma-1)
            meta.update(gamma_formula=f'1 + {scale:.17g}*(0.25+2.25*Beta(1,2)-1)',
                        gamma_low=1-.75*scale,gamma_high=1+1.5*scale,
                        gamma_second_moment_exact=target_k2,affine_scale=float(scale))
    meta.update(gamma_dist=name,quadrature_order=len(gamma),truncated=False,
                gamma_mean_grid=float(weights@gamma),
                gamma_second_moment_grid=float(weights@(gamma**2)),
                gamma_third_moment_grid=float(weights@(gamma**3)))
    assert np.isclose(meta['gamma_mean_grid'],1.,atol=1e-12)
    assert np.isclose(meta['gamma_second_moment_grid'],meta['gamma_second_moment_exact'],atol=1e-12)
    return gamma,weights,meta


def production_numerics() -> dict:
    return {
        "sobol_power": 10,
        "qmc_seed": 1234,
        "finite_prox_steps": 45,
        "finite_prox_tol": 1e-10,
        "dynamic_outer_quad": 31,
        "dynamic_mask_quad": 31,
        "dynamic_grid": 241,
        "dynamic_newton": 35,
    }
