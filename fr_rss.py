import numpy as np
import matplotlib.pyplot as plt


def nan_ccf_single(x, y, k):

	"""
	Computes the cross-correlation function of x and y at lag k.
	"""

	assert len(x) is len(y)
	n = len(x)
	s = np.nansum( (x[:n-k] - np.nanmean(x)) * (y[k:] - np.nanmean(y)) )
	return s / (n * np.nanstd(x) * np.nanstd(y))


def nan_ccf(x, y, lag_max=10):

	"""
	Computes the cross-correlation function of x and y over the
	range of lag values [-lag_max, lag_max]
	"""

	assert len(x) is len(y)
	lag_max = min(lag_max, len(x))
	vals = [nan_ccf_single(x, y, k) for k in xrange(lag_max + 1)]
	vals.reverse()
	vals += [nan_ccf_single(y, x, k) for k in xrange(1, lag_max + 1)]
	lags = range(-lag_max, lag_max + 1)
	return lags, vals


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

	Returns three arrays of size n_trials, one containing the lag values
	corresponding to the maximum value in the CCF of each trial, and the
	other two containing the bootstrap samples of the two time series.
	"""

	assert (len(t1) == len(ts1)) & (len(ts1) == len(ts1_sigma))
	assert (len(t2) == len(ts2)) & (len(ts2) == len(ts2_sigma))
	max_cor_lag_vals = []
	n_data = min(len(ts1), len(ts2))
	ts1_samples = []
	ts2_samples = []
	if len(ts1) < len(ts2):
		ts2 = np.interp(t1, t2, ts2)
		ts2_sigma = np.interp(t1, t2, ts2_sigma)
	else:
		ts1 = np.interp(t2, t1, ts1)
		ts1_sigma = np.interp(t2, t1, ts1_sigma)
	for i in xrange(n_trials):
		ts1_tmp = np.array(ts1)
		ts2_tmp = np.array(ts2)
		# get bootstrap sample
		ts1_tmp, idx_n = get_bootstrap(ts1_tmp)
		# now add gaussian noise with sigma = uncertainty of each observation
		# scaled down by sqrt of the number of times in bootstrap sample
		ts1_tmp += np.random.randn(n_data)*sigma_scale*ts1_sigma/np.sqrt(idx_n)
		# get bootstrap sample
		ts2_tmp, idx_n = get_bootstrap(ts2_tmp)
		# now add gaussian noise with sigma = uncertainty of each observation
		# scaled down by sqrt of the number of times in bootstrap sample
		ts2_tmp += np.random.randn(n_data)*sigma_scale*ts2_sigma/np.sqrt(idx_n)
		lags, ccf_vals = nan_ccf(ts1_tmp, ts2_tmp, lag_max)
		# keep the lag value with the maximum CCF value
		max_cor_lag_vals.append(lags[np.nanargmax(ccf_vals)])
		# keep the individual bootstrap samples for later inspection
		ts1_samples.append(ts1_tmp)
		ts2_samples.append(ts2_tmp)
		if (i+1) % 100 is 0:
			print "Completed {} trials".format(i+1)
	return max_cor_lag_vals, ts1_samples, ts2_samples


def run_mc_epoch(epoch, n_trials=1000, lag_max=12, sigma_scale=1,
	save_plot=False):

	"""
	Convenience function.

	Calls run_mc on the data for the given epoch and plots the result.
	"""

	e = spz_epoch == epoch

	max_cor_lag_vals, kep_samples, spz_samples = run_mc(time_kep, time_spz[e],
		kep_flux, spz_flux[e], kep_sigma, spz_sigma[e],
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

	return max_cor_lag_vals, kep_samples, spz_samples


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
kep_flux = kep[:,1]
kep_sigma = kep[:,2]
spz_flux = spz[:,1]
spz_sigma = spz[:,2]
spz_epoch = spz[:,3]

max_cor_lag_vals, kep_samples, spz_samples = run_mc_epoch(1, save_plot=True)
discrete_hist(max_cor_lag_vals)
# [(-8, 12),
#  (-7, 47),
#  (-6, 146),
#  (-5, 245),
#  (-4, 248),
#  (-3, 183),
#  (-2, 79),
#  (-1, 27),
#  (0, 13)]

# max_cor_lag_vals, kep_samples, spz_samples = run_mc_epoch(1)
# discrete_hist(max_cor_lag_vals)
# max_cor_lag_vals, kep_samples, spz_samples = run_mc_epoch(2)
# discrete_hist(max_cor_lag_vals)
# max_cor_lag_vals, kep_samples, spz_samples = run_mc_epoch(3)
# discrete_hist(max_cor_lag_vals)

# for epoch in range(1,4):
# 	for sigma_scale in [1, 5, 10, 20]:
# 		lags, keps, spzs = run_mc_epoch(epoch, 
# 			sigma_scale=sigma_scale, save_plot=True, lag_max=15)
