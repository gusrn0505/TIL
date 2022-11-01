import numpy as np
import pandas as pd
import statsmodels.stats.weighstats as sms
data = pd.read_csv("data/taxi.txt", sep = '\t', index_col=0)

dat = data/1000
print(dat.describe())
dat_A = dat['BrandA']
dat_B = dat['BrandB']