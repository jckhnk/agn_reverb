import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def ccf_single(x, y, k):

	"""
	Computes the cross-correlation function of x and y at lag k.
	"""

	assert len(x) is len(y)
	n = len(x)
	s = ( (x[:n-k] - x.mean()) * (y[k:] - y.mean()) ).sum()
	return s / (n * x.std() * y.std())


def ccf(x, y, lag_max=10):

	"""
	Computes the cross-correlation function of x and y over the
	range of lag values [-lag_max, lag_max]
	"""

	assert len(x) is len(y)
	lag_max = min(lag_max, len(x))
	vals = [ccf_single(x, y, k) for k in xrange(lag_max + 1)]
	vals.reverse()
	vals += [ccf_single(y, x, k) for k in xrange(1, lag_max + 1)]
	lags = range(-lag_max, lag_max + 1)
	return lags, vals


arr = np.genfromtxt('zw229-spitzer-kepler-140311.txt', skiprows=1)
time, spz, kep = arr[:,0], arr[:,2], arr[:,4]

plt.subplot(211, axisbg='lightgray')
plt.plot(time, spz/spz.mean(), color='red', label='Spitzer', lw=2)
plt.plot(time, kep/kep.mean(), color='blue', label='Kepler', lw=2)
plt.ylabel('Normalized Flux')
plt.xlabel('HJD-55000')
plt.xlim(min(time)*0.99, max(time)*1.01)
plt.legend()
plt.subplot(212, axisbg='lightgray')
lags, ccf_vals = ccf(kep, spz, lag_max=10)
lag_days = np.array(lags) * np.median(np.diff(time))
plt.bar(lag_days, ccf_vals, width=0.5, color='gray', align='center', linewidth=0)
xlim = plt.gca().get_xlim()
ci = 0.95
sig_lev = norm.ppf((1+ci)/2)/np.sqrt(len(time))
plt.plot(xlim, [sig_lev]*2, 'b--', label='95% significance level')
plt.plot(xlim, [-sig_lev]*2, 'b--')
plt.plot(xlim, [0]*2, 'k-')
plt.xlabel('Lag [days]')
plt.ylabel('CCF')
plt.xlim(min(lag_days)-1, max(lag_days)+1)
plt.ylim(round(min(ccf_vals)*10)/10-0.1, round(max(ccf_vals)*10)/10+0.1)
plt.legend()
plt.savefig('spz_kep_ccf.png')
plt.close()
