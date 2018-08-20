
import numpy as np
import pandas as pd
from scipy import stats

def p_value(a,b, two_sided = False):
    #same as stats.ttest_ind(a, b)
    var_a = a.var(ddof=1) # unbiased estimator, divide by N-1 instead of N
    var_b = b.var(ddof=1)
    #we assume equal data length
    N = len(a)
    s = np.sqrt( (var_a + var_b) / 2 ) # balanced standard deviation
    t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N)) # t-statistic
    dof = 2*N - 2 # degrees of freedom
    p = 1 - stats.t.cdf(np.abs(t), df=dof) # one-sided test p-value
    p = 2*p if two_sided else p
    print("t:\t", t, "p:\t", p)
    return t, 2*p

def welch_p_value(a,b, two_sided = False):
    # welch's t-test
    #same as stats.ttest_ind(a, b, equal_var=False)
    N1 = len(a)
    s1_sq = a.var()
    N2 = len(b)
    s2_sq = b.var()
    t = (a.mean() - b.mean()) / np.sqrt(s1_sq / N1 + s2_sq / N2)

    nu1 = N1 - 1
    nu2 = N2 - 1
    dof = (s1_sq / N1 + s2_sq / N2)**2 / ( (s1_sq*s1_sq) / (N1*N1 * nu1) + (s2_sq*s2_sq) / (N2*N2 * nu2) )
    p = (1 - stats.t.cdf(np.abs(t), df=dof))
    p = 2*p if two_sided else p
    print("Welch t-test")
    print("t:\t", t, "p:\t", p)
    return t, p

def chi2_p_value(a, b, dof=1):
    #chi2 test. Requires binary categorical data in 0-1 format.
    A_pos = a.sum()
    A_neg = a.size - a.sum()
    B_pos = b.sum()
    B_neg = b.size - b.sum()

    T = np.array([[A_pos, A_neg], [B_pos, B_neg]])
    # same as scipy.stats.chi2_contingency(T, correction=False)
    det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
    p = 1 - stats.chi2.cdf(x=c2, df=dof)
    print("Chi2 test")
    print("p:\t", p)
    return p
