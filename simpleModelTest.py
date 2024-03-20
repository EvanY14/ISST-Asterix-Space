import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

alpha, sigma = 1, 1

beta = [1., 2.5]

size = 100

X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

Y = alpha + beta[0] * X1 + beta[1] * X2 + sigma * rng.normal(size=size)

import pymc as pm

basic_model = pm.Model()

with basic_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = alpha + beta[0] * X1 + beta[1] * X2

    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    idata = pm.sample()

    az.plot_trace(idata, combined=True, show=True)
    print(az.summary(idata, round_to=2))