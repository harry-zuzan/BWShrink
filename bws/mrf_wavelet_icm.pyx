import pywt
import numpy 
cimport numpy

from libc.math cimport exp, sqrt
from libc.math cimport M_PI

#DEF NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


def shrink_mrf2_redescend(numpy.ndarray[numpy.float64_t,ndim=2] observed,
			double prior_edge_prec, double prior_diag_prec,
			numpy.ndarray[numpy.float64_t,ndim=1] likelihood_prec,
			str wavelet, double cval, int max_iter=30, double converge=1e-6):

	N1,N2 = observed.shape[0], observed.shape[1]

	cdef numpy.ndarray[numpy.float64_t,ndim=2] resids = \
			numpy.zeros((N1,N2), numpy.double)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] redescended = \
			numpy.zeros((N1,N2), numpy.double)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] shrunk = \
			shrink_mrf2(observed, prior_edge_prec, prior_diag_prec,
				likelihood_prec, wavelet, converge)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] shrunk_old = shrunk.copy()

	cdef int iter = 0

	while 1:
		resids = observed - shrunk
#		redescended = shrunk + redescend_residuals_logistic(resids, cval)
		redescended = shrunk + redescend_residuals_normal(resids, 2.5*cval)

		shrunk = shrink_mrf2(redescended, prior_edge_prec, prior_diag_prec,
					likelihood_prec, wavelet, converge)

		iter += 1

		print "iteration number", iter
		if not iter < max_iter: break
		diff =  abs(shrunk - shrunk_old).max()
#		if abs(shrunk - shrunk_old).max() < converge: break
		print 'diff =', diff
		if diff < converge: break

		if iter > 3: shrunk += 0.30*(shrunk - shrunk_old)

		shrunk_old = shrunk.copy()

	return shrunk



def shrink_mrf2(observed, prior_edge_prec, prior_diag_prec,
			likelihood_prec_vec, wavelet, converge=1e-6):

	Wavelet = pywt.Wavelet(wavelet)
	coeffs = pywt.wavedec2(observed,Wavelet)

	lprec = likelihood_prec_vec

	prior_prec_sum = 4.0*(abs(prior_edge_prec) + abs(prior_diag_prec))
	eprec_std = prior_edge_prec/prior_prec_sum
	dprec_std = prior_diag_prec/prior_prec_sum


	for k in range(len(lprec)):
		lprec_k = lprec[k]

		coeffs[-(k+1)][0][:] = shrink_mrf2_icm(coeffs[-(k+1)][0],
								eprec_std, dprec_std, lprec[k])
		coeffs[-(k+1)][1][:] = shrink_mrf2_icm(coeffs[-(k+1)][1],
								eprec_std, dprec_std, lprec[k])
		coeffs[-(k+1)][2][:] = shrink_mrf2_icm(coeffs[-(k+1)][2],
								eprec_std, dprec_std, lprec[k])

	arr2d_shrunk = pywt.waverec2(coeffs,Wavelet)

	return arr2d_shrunk




def shrink_mrf1_icm(numpy.ndarray[numpy.float64_t,ndim=1] vec,
			double prior_prec, double likelihood_prec, double converge=1e-6):
	cdef numpy.ndarray[numpy.float64_t,ndim=1] vec1 = vec.copy()
	cdef numpy.ndarray[numpy.float64_t,ndim=1] old_vec = vec.copy()

	cdef int N = len(vec)

	cdef int i
	cdef double x

	# herehere declare these as doubles
	cdef double prec1 = likelihood_prec + abs(prior_prec)
	cdef prec2 = likelihood_prec + 2.0*abs(prior_prec)

	count = 0
	while 1:
		old_vec = vec1.copy()

		x = prior_prec*vec1[1] + likelihood_prec*vec[0]
		vec1[0] = x/prec1

		for i from 0 < i < N-1:
			x = prior_prec*(vec1[i-1] + vec1[i+1]) + likelihood_prec*vec[i]
			vec1[i] = x/prec2

		x = prior_prec*vec1[N-2] + likelihood_prec*vec[N-1]
		vec1[N-1] = x/prec1

		diff_max = abs(vec1 - old_vec).max()
		
		if diff_max < converge: break
		count += 1

	return vec1


def shrink_mrf2_icm(numpy.ndarray[numpy.float64_t,ndim=2] observed,
			double prior_edge_prec, double prior_diag_prec,
			double likelihood_prec, double converge=1e-6):

	cdef int N = observed.shape[0]
	cdef int P = observed.shape[1]
	cdef numpy.ndarray[numpy.float64_t,ndim=2] shrunk = observed.copy()
	cdef numpy.ndarray[numpy.float64_t,ndim=2] old_shrunk = observed.copy()

	cdef int i, j

	# herehere maybe give x a more meaningful name
	cdef double prec, x
	
	# just to make the code a bit more compact
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec

	
	while 1:
		old_shrunk = shrunk.copy()

		# corners
		prec = 2.0*abs(eprec) + abs(dprec) + lprec

		# top left corner
		x = eprec*(shrunk[0,1] + shrunk[1,0]) + dprec*shrunk[1,1]
		x += lprec*observed[0,0]
		shrunk[0,0] = x/prec

		# top right corner
		x = eprec*(shrunk[0,P-2] + shrunk[1,P-1]) + dprec*shrunk[1,P-2]
		x += lprec*observed[0,P-1]
		shrunk[0,P-1] = x/prec

		# bottom left corner
		x = eprec*(shrunk[N-2,0] + shrunk[N-1,1]) + dprec*shrunk[N-2,1]
		x += lprec*observed[N-1,0]
		shrunk[N-1,0] = x/prec

		# bottom right corner
		x = eprec*(shrunk[N-2,P-1] + shrunk[N-1,P-2]) + dprec*shrunk[N-2,P-2]
		x += lprec*observed[N-1,P-1]
		shrunk[N-1,P-1] = x/prec


		# edges
		prec = 3.0*abs(eprec) + 2.0*abs(dprec) + lprec

		# top side
		for j from 0 < j < P-1:
			x = eprec*(shrunk[0,j-1] + shrunk[0,j+1] + shrunk[1,j])
			x += dprec*(shrunk[1,j-1] + shrunk[1,j+1])
			x += lprec*observed[0,j]
			shrunk[0,j] = x/prec

		# bottom side
		for j from 0 < j < P-1:
			x = eprec*(shrunk[N-1,j-1] + shrunk[N-1,j+1] + shrunk[N-2,j])
			x += dprec*(shrunk[N-2,j-1] + shrunk[N-2,j+1])
			x += observed[N-1,j]
			shrunk[N-1,j] = x/prec


		# left side
		for i from 0 < i < N-1:
			x = eprec*(shrunk[i-1,0] + shrunk[i+1,0] + shrunk[i,1])
			x += dprec*(shrunk[i-1,1] + shrunk[i+1,1])
			x += lprec*observed[i,0]
			shrunk[i,0] = x/prec


		# right side
		for i from 0 < i < N-1:
			x = eprec*(shrunk[i-1,P-1] + shrunk[i+1,P-1] + shrunk[i,P-2])
			x += dprec*(shrunk[i-1,P-2] + shrunk[i+1,P-2])
			x += lprec*observed[i,P-1]
			shrunk[i,P-1] = x/prec

		# middle
		prec = 4.0*abs(eprec) + 4.0*abs(dprec) + lprec


		for i from 0 < i < N-1:
			for j from 0 < j < P-1:
				x = eprec*(shrunk[i-1,j] + shrunk[i+1,j])
				x += eprec*(shrunk[i,j-1] + shrunk[i,j+1])
				x += dprec*(shrunk[i-1,j-1] + shrunk[i-1,j+1])
				x += dprec*(shrunk[i+1,j-1] + shrunk[i+1,j+1])
				x += lprec*observed[i,j]
				shrunk[i,j] = x/prec

		diff_max = abs(shrunk - old_shrunk).max()
		
		if diff_max < converge: break

	return shrunk


#cdef numpy.ndarray[numpy.float64_t,ndim=2] redescend_residuals(
def redescend_residuals_logistic(
		numpy.ndarray[numpy.float64_t,ndim=2] resids, double cval):

	return get_redescend_weights_logistic(resids, cval)*resids


def redescend_residuals_normal(
		numpy.ndarray[numpy.float64_t,ndim=2] resids, double cval):

	return get_redescend_weights_normal(resids, cval)*resids


#cdef numpy.ndarray[numpy.float64_t,ndim=2] get_redescend_weights(
def get_redescend_weights_logistic(
		numpy.ndarray[numpy.float64_t,ndim=2] resids,double cval):

	cdef double sval = resids.std()*sqrt(3.0)/M_PI

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec1 = \
			prob_logistic_vec_2d(resids,sval)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec2 = \
			prob_logistic_vec_2d(resids,cval*sval)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] likelihood_weights = \
		prob_vec1/(prob_vec1 + prob_vec2)

	cdef double prob1 = prob_logistic_scalar(0.0, sval)
	cdef double prob2 = prob_logistic_scalar(0.0, cval*sval)

	likelihood_weights /= prob1/(prob1 + prob2)

	return likelihood_weights


#cdef numpy.ndarray[numpy.float64_t,ndim=2] get_redescend_weights(
def get_redescend_weights_normal(
		numpy.ndarray[numpy.float64_t,ndim=2] resids, double cval):

	cdef double stdev = resids.std()

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec1 = \
			prob_normal_vec_2d(resids,stdev)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec2 = \
			prob_normal_vec_2d(resids,cval*stdev)

	cdef numpy.ndarray[numpy.float64_t,ndim=2] likelihood_weights = \
		prob_vec1/(prob_vec1 + prob_vec2)

	cdef double prob1 = prob_normal_scalar(0.0, stdev)
	cdef double prob2 = prob_normal_scalar(0.0, cval*stdev)

	likelihood_weights /= prob1/(prob1 + prob2)

	return likelihood_weights




cdef double prob_logistic_scalar(double val, double s):
	cdef double numerator = exp(-val/s)
	cdef double denominator = 1.0 + numerator
	denominator *= s*denominator

	return numerator/denominator


cdef numpy.ndarray[numpy.float64_t,ndim=1] prob_logistic_vec_1d(
		numpy.ndarray[numpy.float64_t,ndim=1] vec, double s):

	cdef numpy.ndarray[numpy.float64_t,ndim=1] numerator = exp(-vec/s)
	cdef numpy.ndarray[numpy.float64_t,ndim=1] denominator = \
		1.0 + numerator
	denominator *= s*denominator

	return numerator/denominator


cdef numpy.ndarray[numpy.float64_t,ndim=2] \
	prob_logistic_vec_2d(numpy.ndarray[numpy.float64_t,ndim=2] vec, double s):
	cdef numpy.ndarray[numpy.float64_t,ndim=2] numerator = numpy.exp(-vec/s)
	cdef numpy.ndarray[numpy.float64_t,ndim=2] denominator = \
		1.0 + numerator
	denominator *= s*denominator

	return numerator/denominator


cdef double prob_normal_scalar(double val, double s):
	cdef double prob = numpy.exp(-0.5*(val/s)**2)

	return prob/(s*sqrt(2.0*M_PI))


cdef numpy.ndarray[numpy.float64_t,ndim=2] \
	prob_normal_vec_2d(numpy.ndarray[numpy.float64_t,ndim=2] vec, double s):

	cdef numpy.ndarray[numpy.float64_t,ndim=2] prob_vec = \
			numpy.exp(-0.5*(vec/s)**2)

	return prob_vec/(s*sqrt(2.0*M_PI))

