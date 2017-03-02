import pywt
#from copy import copy

import numpy 
cimport numpy

#from libc.math cimport exp, sqrt
#from libc.math cimport M_PI

from redescendl import redescend_normal1, redescend_normal2



def shrink_mrf1_redescend(numpy.ndarray[numpy.float64_t,ndim=1] observed,
			double prior_prec,
			numpy.ndarray[numpy.float64_t,ndim=1] likelihood_prec,
			str wavelet, double cval, int stagger=0,
			int max_iter=30, double converge=1e-6):

	N = observed.shape[0]

	cdef numpy.ndarray[numpy.float64_t,ndim=1] resids = \
			numpy.zeros((N,), numpy.double)

	cdef numpy.ndarray[numpy.float64_t,ndim=1] redescended = \
			numpy.zeros((N,), numpy.double)

	cdef numpy.ndarray[numpy.float64_t,ndim=1] shrunk = \
			shrink_mrf1(observed, prior_prec, likelihood_prec,
			wavelet, converge)

	cdef numpy.ndarray[numpy.float64_t,ndim=1] shrunk_old = shrunk.copy()

	cdef int iter_count = 0

	cdef double diff

	shrink_func = shrink_mrf1
	if stagger: shrink_func = shrink_mrf1_stagger

	while 1:
		resids = observed - shrunk
		redescended = shrunk + redescend_normal1(resids, cval)

		shrunk = shrink_mrf1(redescended, prior_prec, likelihood_prec,
					wavelet, converge)

		iter_count += 1

		if not iter_count < max_iter: break
		diff =  abs(shrunk - shrunk_old).max()
		if diff < converge: break

		if iter_count > 3: shrunk += 0.35*(shrunk - shrunk_old)

		shrunk_old = shrunk.copy()

	return (shrunk,iter_count)



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

	cdef int iter_count = 0

	cdef double diff

	while 1:
		resids = observed - shrunk
		redescended = shrunk + redescend_normal2(resids, cval)

		shrunk = shrink_mrf2(redescended, prior_edge_prec, prior_diag_prec,
					likelihood_prec, wavelet, converge)

		iter_count += 1

		if not iter_count < max_iter: break
		diff =  abs(shrunk - shrunk_old).max()
		if diff < converge: break

		if iter_count > 3: shrunk += 0.35*(shrunk - shrunk_old)

		shrunk_old = shrunk.copy()

	return (shrunk,iter_count)


def shrink_mrf1(observed, prior_prec, likelihood_prec_vec,
		wavelet, converge=1e-6):

	Wavelet = pywt.Wavelet(wavelet)
	coeffs = pywt.wavedec(observed,Wavelet)

	lprec = likelihood_prec_vec

	# sum of the prior precisions is either -1.0 or 1.0
	# need a better name than precision to convey negative correlations
	# in the precision matrix
	prec_std = 0.5*prior_prec/abs(prior_prec)


	for k in range(len(lprec)):
		lprec_k = lprec[k]

		coeffs[-(k+1)][:] = shrink_mrf1_icm(coeffs[-(k+1)],
								prec_std, lprec[k], converge)

	arr1d_shrunk = pywt.waverec(coeffs,Wavelet)

	return arr1d_shrunk



def shrink_mrf1_stagger(observed,prior_prec,lprec_vec,wavelet,converge=1e-6):
	Wavelet = pywt.Wavelet(wavelet)
	prec_std = 0.5*prior_prec/abs(prior_prec)

	lprec = lprec_vec[-1]

	observed1 = observed[:-1]
	observed2 = observed[1:]
		
	avg1,detail1 = pywt.dwt(observed1,Wavelet)
	avg2,detail2 = pywt.dwt(observed2,Wavelet)

	detail1[:] = shrink_mrf1_icm(detail1, prec_std, lprec, converge)
	detail2[:] = shrink_mrf1_icm(detail2, prec_std, lprec, converge)

	# recursively shrink the averages
	if len(lprec_vec) > 1:
		avg1 = shrink_mrf1_stagger(avg1, prior_prec, lprec_vec[:-1],
				wavelet, converge)

		avg2 = shrink_mrf1_stagger(avg2, prior_prec, lprec_vec[:-1],
				wavelet, converge)

	shrunk1 = pywt.idwt(avg1,detail1,Wavelet)[:len(observed1)]
	shrunk2 = pywt.idwt(avg2,detail2,Wavelet)[:len(observed2)]

	shrunk = numpy.zeros_like(observed)
	shrunk[:-1] += 0.5*shrunk1
	shrunk[1:] += 0.5*shrunk2
	shrunk[0] *= 2.0
	shrunk[-1] *= 2.0

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




def shrink_mrf3_icm(numpy.ndarray[numpy.float64_t,ndim=3] observed,
			double prior_side_prec, double prior_edge_prec,
			double prior_diag_prec, double likelihood_prec,
			double converge=1e-6):

	cdef int M = observed.shape[0]
	cdef int N = observed.shape[1]
	cdef int P = observed.shape[2]
	cdef numpy.ndarray[numpy.float64_t,ndim=3] shrunk = observed.copy()
	cdef numpy.ndarray[numpy.float64_t,ndim=3] old_shrunk = observed.copy()

	cdef int i, j, k

	# maybe give x a more meaningful name or document
	cdef double prec, x, xt, xb
	
	# just to make the code a bit more compact
	cdef double sprec = prior_side_prec
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec

	
	while 1:
		old_shrunk = shrunk.copy()

		# corners
		prec = 3.0*abs(sprec) + 3.0*abs(eprec) + dprec + lprec

		# top left corner on the lower face is at voxel 0,0,0
		x =  sprec*(shrunk[1,0,0] + shrunk[0,1,0] + shrunk[0,0,1])
		x += eprec*(shrunk[1,0,1] + shrunk[0,1,1] + shrunk[1,1,0])
		x += dprec*shrunk[1,1,1]
		x += lprec*observed[0,0,0]
		shrunk[0,0,0] = x/prec

		# top left corner on the upper face is at voxel 0,0,P-1
		x =  sprec*(shrunk[1,0,P-1] + shrunk[0,1,P-1] + shrunk[0,0,P-2])
		x += eprec*(shrunk[1,0,P-2] + shrunk[0,1,P-2] + shrunk[1,1,P-1])
		x += dprec*shrunk[1,1,P-2]
		x += lprec*observed[0,0,P-1]
		shrunk[0,0,P-1] = x/prec

		#------------------------------------------------

		# top right corner on the lower face is at voxel 0,N-1,0
		x =  sprec*(shrunk[1,N-1,0] + shrunk[0,N-2,0] + shrunk[0,N-1,1])
		x += eprec*(shrunk[1,N-1,1] + shrunk[0,N-2,1] + shrunk[1,N-2,0])
		x += dprec*shrunk[1,N-2,1]
		x += lprec*observed[0,N-1,0]
		shrunk[0,N-1,0] = x/prec

		# top right corner on the upper face is at voxel 0,N-1,P-1
		x =  sprec*(shrunk[1,N-1,P-1] + shrunk[0,N-2,P-1] + shrunk[0,N-1,P-2])
		x += eprec*(shrunk[1,N-1,P-2] + shrunk[0,N-2,P-2] + shrunk[1,N-2,P-1])
		x += dprec*shrunk[1,N-2,P-2]
		x += lprec*observed[0,N-1,P-1]
		shrunk[0,N-1,P-1] = x/prec

		#------------------------------------------------

		# bottom left corner on the lower face is at voxel M-1,0,0
		x =  sprec*(shrunk[M-2,0,0] + shrunk[M-1,1,0] + shrunk[M-1,0,1])
		x += eprec*(shrunk[M-2,0,1] + shrunk[M-1,1,1] + shrunk[M-2,1,0])
		x += dprec*shrunk[M-2,1,1]
		x += lprec*observed[M-1,0,0]
		shrunk[M-1,0,0] = x/prec

		# bottom left corner on the upper face is at voxel M-1,0,P-1
		x =  sprec*(shrunk[M-2,0,P-1] + shrunk[M-1,1,P-1] + shrunk[M-1,0,P-2])
		x += eprec*(shrunk[M-2,0,P-2] + shrunk[M-1,1,P-2] + shrunk[M-2,1,P-1])
		x += dprec*shrunk[M-2,1,P-2]
		x += lprec*observed[M-1,0,P-1]
		shrunk[M-1,0,P-1] = x/prec

		#------------------------------------------------
		# last corner to do is the bottom right

		# bottom right corner on the lower face is at voxel M-1,N-1,0
		x =  sprec*(shrunk[M-2,N-1,0] + shrunk[M-1,N-2,0] + shrunk[M-1,N-1,1])
		x += eprec*(shrunk[M-2,N-1,1] + shrunk[M-1,N-2,1] + shrunk[M-2,N-2,0])
		x += dprec*shrunk[M-2,N-2,1]
		x += lprec*observed[M-1,N-1,0]
		shrunk[M-1,N-1,0] = x/prec

		# bottom right corner on the upper face is at voxel M-1,N-1,P-1
		x =  sprec*(shrunk[M-2,N-1,P-1] + shrunk[M-1,N-2,P-1]
						+ shrunk[M-1,N-1,P-2])
		x += eprec*(shrunk[M-2,N-1,P-2] + shrunk[M-1,N-2,P-2]
						+ shrunk[M-2,N-2,P-1])
		x += dprec*shrunk[M-2,N-2,P-2]
		x += lprec*observed[M-1,N-1,P-1]
		shrunk[M-1,N-1,P-1] = x/prec

		#------------------------------------------------
		#------------------------------------------------

		# edges
		prec = 4.0*abs(sprec) + 5.0*abs(eprec) +2*abs(dprec) + lprec


		# left side edge along the bottom face
		for 0 < i < M-1:
			xb =  sprec*(shrunk[i-1,0,0] + shrunk[i+1,0,0])
			xb += sprec*(shrunk[i,1,0] + shrunk[i,0,1])

			xb += eprec*(shrunk[i-1,0,1] + shrunk[i+1,0,1])
			xb += eprec*(shrunk[i-1,1,0] + shrunk[i+1,1,0])
			xb += eprec*shrunk[i,1,1]

			xb += dprec*(shrunk[i-1,1,1] + shrunk[i+1,1,1])
			xb += lprec*observed[i,0,0]

			shrunk[i,0,0] = xb/prec

		# left side edge along the top face
			xt =  sprec*(shrunk[i-1,0,P-1] + shrunk[i+1,0,P-1])
			xt += sprec*(shrunk[i,1,P-1] + shrunk[i,0,P-2])

			xt += eprec*(shrunk[i-1,0,P-2] + shrunk[i+1,0,P-2])
			xt += eprec*(shrunk[i-1,1,P-1] + shrunk[i+1,1,P-1])
			xt += eprec*shrunk[i,1,P-2]

			xt += dprec*(shrunk[i-1,1,P-2] + shrunk[i+1,1,P-2])
			xt += lprec*observed[i,0,P-1]

			shrunk[i,0,P-1] = xt/prec

		# right side edge along the bottom face
		for 0 < i < M-1:
			xb =  sprec*(shrunk[i-1,N-1,0] + shrunk[i+1,N-1,0])
			xb += sprec*(shrunk[i,N-2,0] + shrunk[i,N-1,1])

			xb += eprec*(shrunk[i-1,N-1,1] + shrunk[i+1,N-1,1])
			xb += eprec*(shrunk[i-1,N-2,0] + shrunk[i+1,N-2,0])
			xb += eprec*shrunk[i,N-2,1]

			xb += dprec*(shrunk[i-1,N-2,1] + shrunk[i+1,N-2,1])
			xb += lprec*observed[i,N-1,0]

			shrunk[i,N-1,0] = xb/prec

		# right side edge along the top face
			xt =  sprec*(shrunk[i-1,N-1,P-1] + shrunk[i+1,N-1,P-1])
			xt += sprec*(shrunk[i,N-2,P-1] + shrunk[i,N-1,P-2])

			xt += eprec*(shrunk[i-1,N-1,P-2] + shrunk[i+1,N-1,P-2])
			xt += eprec*(shrunk[i-1,N-2,P-1] + shrunk[i+1,N-2,P-1])
			xt += eprec*shrunk[i,N-2,P-2]

			xt += dprec*(shrunk[i-1,N-2,P-2] + shrunk[i+1,N-2,P-2])
			xt += lprec*observed[i,N-1,0]

			shrunk[i,N-1,P-1] = xt/prec


		# top side edge along the bottom face
		for 0 < i < N-1:
			xb =  sprec*(shrunk[0,i-1,0] + shrunk[0,i+1,0])
			xb += sprec*(shrunk[1,i,0] + shrunk[0,i,1])

			xb += eprec*(shrunk[0,i-1,1] + shrunk[0,i+1,1])
			xb += eprec*(shrunk[1,i-1,0] + shrunk[1,i+1,0])
			xb += eprec*shrunk[1,i,1]

			xb += dprec*(shrunk[1,i-1,1] + shrunk[1,i+1,1])
			xb += lprec*observed[0,i,0]

			shrunk[0,i,0] = xb/prec


		# top side edge along the top face
			xt =  sprec*(shrunk[0,i-1,P-1] + shrunk[0,i+1,P-1])
			xt += sprec*(shrunk[1,i,P-1] + shrunk[0,i,P-2])

			xt += eprec*(shrunk[0,i-1,P-2] + shrunk[0,i+1,P-2])
			xt += eprec*(shrunk[1,i-1,P-1] + shrunk[1,i+1,P-1])
			xt += eprec*shrunk[1,i,P-2]

			xt += dprec*(shrunk[1,i-1,P-2] + shrunk[1,i+1,P-2])
			xt += lprec*observed[0,i,P-1]

			shrunk[0,i,P-1] = xt/prec


		# bottom side edge along the bottom face
		for 0 < i < N-1:
			xb =  sprec*(shrunk[M-1,i-1,0] + shrunk[M-1,i+1,0])
			xb += sprec*(shrunk[M-2,i,0] + shrunk[M-1,i,1])

			xb += eprec*(shrunk[M-1,i-1,1] + shrunk[M-1,i+1,1])
			xb += eprec*(shrunk[M-2,i-1,0] + shrunk[M-2,i+1,0])
			xb += eprec*shrunk[M-2,i,1]

			xb += dprec*(shrunk[M-2,i-1,1] + shrunk[M-2,i+1,1])
			xb += lprec*observed[M-1,i,0]

			shrunk[M-1,i,0] = xb/prec


		# bottom side edge along the top face
		for 0 < i < N-1:
			pass
			xt =  sprec*(shrunk[M-1,i-1,P-1] + shrunk[M-1,i+1,P-1])
			xt += sprec*(shrunk[M-2,i,P-1] + shrunk[M-1,i,P-2])

			xt += eprec*(shrunk[M-1,i-1,P-2] + shrunk[M-1,i+1,P-2])
			xt += eprec*(shrunk[M-2,i-1,P-1] + shrunk[M-2,i+1,P-1])
			xt += eprec*shrunk[M-2,i,P-2]

			xt += dprec*(shrunk[M-2,i-1,P-2] + shrunk[M-2,i+1,P-2])
			xt += lprec*observed[M-1,i,P-1]

			shrunk[M-1,i,P-1] = xt/prec


		# front side edge along the left face (front means at origin)
		for 0 < i < P-1:
			xb =  sprec*(shrunk[0,0,i-1] + shrunk[0,0,i+1])
			xb += sprec*(shrunk[1,0,i] + shrunk[0,1,i])

			xb += eprec*(shrunk[0,1,i-1] + shrunk[0,1,i+1])
			xb += eprec*(shrunk[1,0,i-1] + shrunk[1,0,i+1])
			xb += eprec*shrunk[1,1,i]

			xb += dprec*(shrunk[1,1,i-1] + shrunk[1,1,i+1])
			xb += lprec*observed[0,0,i]

			shrunk[0,0,i] = xb/prec


		# front side edge along the right face (front means at origin)
			xt =  sprec*(shrunk[0,N-1,i-1] + shrunk[0,N-1,i+1])
			xt += sprec*(shrunk[1,N-1,i] + shrunk[0,N-2,i])

			xt += eprec*(shrunk[0,N-2,i-1] + shrunk[0,N-2,i+1])
			xt += eprec*(shrunk[1,N-1,i-1] + shrunk[1,N-1,i+1])
			xt += eprec*shrunk[1,N-2,i]

			xt += dprec*(shrunk[1,N-2,i-1] + shrunk[1,N-2,i+1])
			xt += lprec*observed[0,N-1,i]

			shrunk[0,N-1,i] = xt/prec


		# back side edge along the left face (back means away from origin)
		for 0 < i < P-1:
			xb =  sprec*(shrunk[M-1,0,i-1] + shrunk[M-1,0,i+1])
			xb += sprec*(shrunk[M-2,0,i] + shrunk[M-1,1,i])

			xb += eprec*(shrunk[M-1,1,i-1] + shrunk[M-1,1,i+1])
			xb += eprec*(shrunk[M-2,0,i-1] + shrunk[M-2,0,i+1])
			xb += eprec*shrunk[M-2,1,i]

			xb += dprec*(shrunk[M-2,1,i-1] + shrunk[M-2,1,i+1])
			xb += lprec*observed[M-1,0,i]

			shrunk[M-1,0,i] = xb/prec


		# back side edge along the right face (back means away from origin)
		for 0 < i < P-1:
			xt =  sprec*(shrunk[M-1,N-1,i-1] + shrunk[M-1,N-1,i+1])
			xt += sprec*(shrunk[M-2,N-1,i] + shrunk[M-1,N-2,i])

			xt += eprec*(shrunk[M-1,N-2,i-1] + shrunk[M-1,N-2,i+1])
			xt += eprec*(shrunk[M-2,N-1,i-1] + shrunk[M-2,N-1,i+1])
			xt += eprec*shrunk[M-2,N-2,i]

			xt += dprec*(shrunk[M-2,N-2,i-1] + shrunk[M-2,N-2,i+1])
			xt += lprec*observed[M-1,N-1,i]

			shrunk[M-1,N-1,i] = xt/prec

		#------------------------------------------------
		#------------------------------------------------

		# middle
		prec = 6.0*abs(sprec) + 8*(abs(eprec) + abs(dprec)) + lprec


		for i from 0 < i < M-1:
			for j from 0 < j < N-1:
				for k from 0 < k < P-1:
					x =  sprec*(shrunk[i,j,k-1] + shrunk[i,j,k+1])
					x += sprec*(shrunk[i,j-1,k] + shrunk[i,j+1,k])
					x += sprec*(shrunk[i-1,j,k] + shrunk[i+1,j,k])

					x += eprec*(shrunk[i,j-1,k-1] + shrunk[i,j+1,k-1])
					x += eprec*(shrunk[i,j-1,k+1] + shrunk[i,j+1,k+1])
					x += eprec*(shrunk[i-1,j,k-1] + shrunk[i+1,j,k-1])
					x += eprec*(shrunk[i-1,j,k+1] + shrunk[i+1,j,k+1])

					x += dprec*(shrunk[i-1,j-1,k-1] + shrunk[i-1,j+1,k-1])
					x += dprec*(shrunk[i+1,j-1,k-1] + shrunk[i+1,j+1,k-1])
					x += dprec*(shrunk[i-1,j-1,k+1] + shrunk[i-1,j+1,k+1])
					x += dprec*(shrunk[i+1,j-1,k+1] + shrunk[i+1,j+1,k+1])

					x += lprec*observed[i,j,k]

					shrunk[i,j,k] = x/prec


		diff_max = abs(shrunk - old_shrunk).max()
		
		if diff_max < converge: break

	return shrunk

