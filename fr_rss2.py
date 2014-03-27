import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
from scipy.stats import norm

def nan_ccf_single(x, y, k):

	"""
	Computes the cross-correlation function of x and y at lag k.
	"""

	assert len(x) is len(y)
	n = len(x)
	s = np.nansum( (x[:n-k] - np.nanmean(x)) * (y[k:] - np.nanmean(y)) )
	return s / (n * np.nanstd(x) * np.nanstd(y))


# def nan_ccf(tx, x, ty, y, lag_max=10):

# 	"""
# 	Computes the cross-correlation function of x and y over the
# 	range of lag values [-lag_max, lag_max]. Assumes the second
# 	time series is infinitely longer than the other in both time 
# 	directions (at least longer by lag_max in both time directions). 
# 	Uses linear interpolation to put subsets of the second
# 	time series on the same time grid, with the target time grid
# 	defined by the input grid shifted by lag * median(diff(tx)).
# 	"""

# 	vals = []
# 	for i in range(lag_max+1):
# 		tx_trgt = tx - i * bn.nanmedian(np.diff(tx))
# 		y_interp = np.interp(tx_trgt, ty, y)
# 		vals.append(nan_ccf_single(x, y_interp, 0))
# 	vals.reverse()
# 	for i in range(1, lag_max+1):
# 		tx_trgt = tx + i * bn.nanmedian(np.diff(tx))
# 		y_interp = np.interp(tx_trgt, ty, y)
# 		vals.append(nan_ccf_single(x, y_interp, 0))
# 	lags = range(-lag_max, lag_max + 1)
# 	return lags, vals


def get_bootstrap(ts):

	"""
	Samples the input array uniformly at random with replacement. 

	Returns the resulting bootstrap sample with non-sampled array
	elements masked with np.nan values, and an array containing the
	number of times each element was sampled.
	"""


	n_data = len(ts)
	# select n_data random indexs with replacement
	idx = np.random.randint(0, n_data, size=n_data)
	# count the number of times each point is selected in the bootstrap
	idx_n = np.array([np.sum(idx==i) if i in idx else np.nan 
		for i in range(len(idx))])
	# get complement of set of these integers and use to set flux to np.nan
	idx_nan = np.ones(n_data).astype('bool')
	idx_nan[idx] = False
	ts[idx_nan] = np.nan
	return ts, idx_n


def run_mc(t1, t2, ts1, ts2, ts1_sigma, ts2_sigma, 
	n_trials=1000, lag_max=10, sigma_scale=1):

	"""
	Runs the Monte Carlo experiment for the input time series arrays.
	
	t1, t2 are the time arrays (x-coord) corresponding to the flux arrays
	ts1, ts2 (y-coord). 
	
	ts1_sigma and ts2_sigma arrays contain the uncertainties corresponding
	to the flux measurements in ts1, ts2.

	sigma_scale is the scale factor by which to multiply the measurement 
	uncertainties.

	Assumes ts2 has at least lag_max data points preceding
	the beginning of ts1.

	Returns three arrays of size n_trials, one containing the lag values
	corresponding to the maximum value in the CCF of each trial, and the
	other two containing the bootstrap samples of the two time series.
	"""

	assert (len(t1) == len(ts1)) & (len(ts1) == len(ts1_sigma))
	assert (len(t2) == len(ts2)) & (len(ts2) == len(ts2_sigma))
	max_cor_lag_vals = []
	n_data = len(ts1)
	ts1_samples = []
	ts2_samples = []
	ts2_interps, ts2_sigma_interps = [], []
	for i in range(lag_max+1):
		t1_trgt = t1 - i * np.median(np.diff(t1))
		ts2_interps.append(np.interp(t1_trgt, t2, ts2))
		ts2_sigma_interps.append(np.interp(t1_trgt, t2, ts2_sigma))
	for n in xrange(n_trials):
		ccf_vals = []
		for i in range(lag_max+1):
			ts2_interp = ts2_interps[i]
			ts2_sigma_interp = ts2_sigma_interps[i]
			# get bootstrap sample of ts1
			ts1_tmp, idx_n = get_bootstrap(np.array(ts1))
			# now add gaussian noise with sigma = uncertainty of each observation
			# scaled down by sqrt of the number of times in bootstrap sample
			noise = np.random.randn(n_data)*sigma_scale*ts1_sigma/np.sqrt(idx_n)
			ts1_tmp += noise
			# get bootstrap sample of ts2
			ts2_tmp, idx_n = get_bootstrap(np.array(ts2_interp))
			# now add gaussian noise with sigma = uncertainty of each observation
			# scaled down by sqrt of the number of times in bootstrap sample
			noise = np.random.randn(n_data)*sigma_scale*ts2_sigma_interp/np.sqrt(idx_n)
			ts2_tmp += noise
			ccf_vals.append(nan_ccf_single(ts1_tmp, ts2_tmp, 0))
		max_cor_lag_vals.append(np.argmax(ccf_vals))
		if (n+1) % 100 is 0:
			print "Completed {} trials".format(n+1)
	return max_cor_lag_vals


def run_mc_epoch(epoch, n_trials=1000, lag_max=12, sigma_scale=1,
	save_plot=False):

	"""
	Convenience function.

	Calls run_mc on the data for the given epoch and plots the result.
	"""

	e = epoch_spz == epoch

	max_cor_lag_vals = run_mc(time_spz[e], 
		time_kep, flux_spz[e], flux_kep, sigma_spz[e], sigma_kep,
		n_trials=n_trials, lag_max=lag_max, sigma_scale=sigma_scale)

	plt.subplot(111, axisbg='lightgray')
	tau = np.array(max_cor_lag_vals) * np.median(np.diff(time_spz[e]))
	plt.hist(tau, normed=True, bins=len(set(tau)), 
		color='gray', histtype='stepfilled')
	plt.xlabel('Centroid [days]')
	plt.ylabel('Probability Density')
	mu, sd = norm.fit(tau)
	xlim = plt.gca().get_xlim()
	xfit = np.linspace(xlim[0], xlim[1], 100)
	yfit = norm.pdf(xfit, loc=mu, scale=sd)
	plt.plot(xfit, yfit, 'r-', lw=2)
	plt.title(r'$\tau_{cent} = $'+str(round(mu,1))+\
		r'$\pm$'+str(round(sd,1))+' days', fontsize=20)

	if save_plot:
		name = 'tau_centroid_epoch{}_sigma_scale{}_n{}.pdf'
		plt.savefig(name.format(epoch, sigma_scale, n_trials))
		plt.close()
	plt.show()

	return max_cor_lag_vals


def discrete_hist(x):

	"""
	Prints a simple discrete histogram.
	"""

	return [(i, np.sum(np.array(x, copy=False)==i)) 
		for i in sorted(set(x))]


kep = np.genfromtxt('zw229-kepler-140317.txt', skiprows=1)
spz = np.genfromtxt('zw229-spitzer-140320.txt', skiprows=1)
time_kep = kep[:,0]
time_spz = spz[:,0]
flux_kep = kep[:,1]
flux_spz = spz[:,1]
sigma_kep = kep[:,2]
sigma_spz = spz[:,2]
epoch_spz = spz[:,3]

# e1 = epoch_spz == 1
# t1, t2, ts1, ts2, ts1_sigma, ts2_sigma = time_spz[e1], \
# time_kep, flux_spz[e1], flux_kep, sigma_spz[e1], sigma_kep

max_cor_lag_vals = run_mc_epoch(1, save_plot=True)
discrete_hist(max_cor_lag_vals)

# max_cor_lag_vals, kep_samples, spz_samples = run_mc_epoch(1)
# discrete_hist(max_cor_lag_vals)
# max_cor_lag_vals, kep_samples, spz_samples = run_mc_epoch(2)
# discrete_hist(max_cor_lag_vals)
# max_cor_lag_vals, kep_samples, spz_samples = run_mc_epoch(3)
# discrete_hist(max_cor_lag_vals)

for epoch in range(1,4):
	for sigma_scale in [1, 5, 10, 20]:
		lags = run_mc_epoch(epoch, 
			sigma_scale=sigma_scale, save_plot=True, lag_max=15)
