#import numpy 
cimport numpy
import cython
from cython.parallel import prange

# should be cimport to run in parallel
#from libc.math import exp as exp_c
from libc.math cimport fabs as fabs_c


# This isn't parallel because it doesn't condition on sets from the
# global Markov property as easily as do 2D and 3D arrays.  The number
# of processors/threads will need to be identified ahead of time to
# condition on end points and then mid-points.
@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf1_icm(double[:] vec,
			double pprec, double lprec, double converged=1e-6):

	cdef double[:] vec1 = vec.copy()
	cdef double[:] old_vec = vec.copy()
	cdef double[:] diffs = numpy.zeros_like(vec)

	cdef int N = vec.shape[0]

	cdef int i
	cdef start_idx
	cdef double val

	cdef double prec1 = lprec + fabs_c(pprec)
	cdef double prec2 = lprec + 2.0*fabs_c(pprec)

	while 1:
		old_vec = vec1.copy()

		vec1[0] = (pprec*vec1[1] + lprec*vec[0])/prec1
		diffs[0] = fabs_c(vec1[0] - old_vec[0])

		for i in range(1,N-1):
			val = pprec*(vec1[i-1] + vec1[i+1]) + lprec*vec[i]
			vec1[i] = val/prec2
			diffs[i] = fabs_c(vec1[i] - old_vec[i])

		vec1[N-1] = (pprec*vec1[N-2] + lprec*vec[N-1])/prec1
		diffs[N-1] = fabs_c(vec1[N-1] - old_vec[N-1])
		
		# convergence is quadratic the sanity check needs to be on how
		# small the value of converged is
		if diffs.max() < converged: break

	return vec1



@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf2_icm(double[:,:] observed,
			double prior_edge_prec, double prior_diag_prec,
			double likelihood_prec, double converged=1e-6):

	cdef int N = observed.shape[0]
	cdef int P = observed.shape[1]
	cdef double[:,:] shrunk = observed.copy()
	cdef double[:,:] old_shrunk = observed.copy()
	cdef double[:,:] diffs = numpy.zeros_like(observed)

	cdef int i, j
	cdef int start_idx

	cdef double prec, val
	
	# just to make the code a bit more compact
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec

	
	while 1:
		old_shrunk = shrunk.copy()

		# corners
		prec = 2.0*fabs_c(eprec) + fabs_c(dprec) + lprec

		# top left corner
		val = eprec*(shrunk[0,1] + shrunk[1,0]) + dprec*shrunk[1,1]
		val = val + lprec*observed[0,0]
		shrunk[0,0] = val/prec
		diffs[0,0] = fabs_c(shrunk[0,0] - old_shrunk[0,0])
		

		# top right corner
		val = eprec*(shrunk[0,P-2] + shrunk[1,P-1]) + dprec*shrunk[1,P-2]
		val = val + lprec*observed[0,P-1]
		shrunk[0,P-1] = val/prec
		diffs[0,P-1] = fabs_c(shrunk[0,P-1] - old_shrunk[0,P-1])

		# bottom left corner
		val = eprec*(shrunk[N-2,0] + shrunk[N-1,1]) + dprec*shrunk[N-2,1]
		val = val + lprec*observed[N-1,0]
		shrunk[N-1,0] = val/prec
		diffs[N-1,0] = fabs_c(shrunk[N-1,0] - old_shrunk[N-1,0])

		# bottom right corner
		val = eprec*(shrunk[N-2,P-1] + shrunk[N-1,P-2]) + dprec*shrunk[N-2,P-2]
		val = val + lprec*observed[N-1,P-1]
		shrunk[N-1,P-1] = val/prec
		diffs[N-1,P-1] = fabs_c(shrunk[N-1,P-1] - old_shrunk[N-1,P-1])


		# edges
		prec = 3.0*fabs_c(eprec) + 2.0*fabs_c(dprec) + lprec

		# across top then bottom
		for j in range(1,P-1):
			val = eprec*(shrunk[0,j-1] + shrunk[0,j+1] + shrunk[1,j])
			val = val +dprec*(shrunk[1,j-1] + shrunk[1,j+1])
			val = val + lprec*observed[0,j]
			shrunk[0,j] = val/prec
			diffs[0,j] = fabs_c(shrunk[0,j] - old_shrunk[0,j])

			val = eprec*(shrunk[N-1,j-1] + shrunk[N-1,j+1] + shrunk[N-2,j])
			val = val + dprec*(shrunk[N-2,j-1] + shrunk[N-2,j+1])
			val = val + lprec*observed[N-1,j]
			shrunk[N-1,j] = val/prec
			diffs[N-1,j] = fabs_c(shrunk[N-1,j] - old_shrunk[N-1,j])


		# down left then right
		for i in range(1,N-1):
			val = eprec*(shrunk[i-1,0] + shrunk[i+1,0] + shrunk[i,1])
			val = val + dprec*(shrunk[i-1,1] + shrunk[i+1,1])
			val = val + lprec*observed[i,0]
			shrunk[i,0] = val/prec
			diffs[i,0] = fabs_c(shrunk[i,0] - old_shrunk[i,0])

			val = eprec*(shrunk[i-1,P-1] + shrunk[i+1,P-1] + shrunk[i,P-2])
			val = val + dprec*(shrunk[i-1,P-2] + shrunk[i+1,P-2])
			val = val + lprec*observed[i,P-1]
			shrunk[i,P-1] = val/prec
			diffs[i,P-1] = fabs_c(shrunk[i,P-1] - old_shrunk[i,P-1])


		# middle precision
		prec = 4.0*fabs_c(eprec) + 4.0*fabs_c(dprec) + lprec

		for start_idx in range(1,3):
			for i in prange(start_idx,N-1,2,nogil=True):
				for j in range(1,P-1):
					pass
					val = eprec*(shrunk[i-1,j] + shrunk[i+1,j])
					val = val + eprec*(shrunk[i,j-1] + shrunk[i,j+1])
					val = val + dprec*(shrunk[i-1,j-1] + shrunk[i-1,j+1])
					val = val + dprec*(shrunk[i+1,j-1] + shrunk[i+1,j+1])
					val = val + lprec*observed[i,j]
					shrunk[i,j] = val/prec
					diffs[i,j] = fabs_c(shrunk[i,j] - old_shrunk[i,j])

		
		if numpy.max() < converged: break

	return shrunk




@cython.boundscheck(False)
@cython.cdivision(True)
def shrink_mrf3_icm(double[:,:,:] observed,
			double prior_side_prec, double prior_edge_prec,
			double prior_diag_prec, double likelihood_prec,
			double converged=1e-6):

	cdef int M = observed.shape[0]
	cdef int N = observed.shape[1]
	cdef int P = observed.shape[2]
	cdef double[:,:,:] shrunk = observed.copy()
	cdef double[:,:,:] diffs = observed.copy()

	cdef int i, j, k

	# maybe give x a more meaningful name or document
#	cdef double prec, x, xt, xb
	cdef double prec, val
	
	# just to make the code a bit more compact
	cdef double sprec = prior_side_prec
	cdef double eprec = prior_edge_prec
	cdef double dprec = prior_diag_prec
	cdef double lprec = likelihood_prec

	
	while 1:
		# this also doubles to keep trackof the absolute difference
		diffs = shrunk.copy()

		# corners
		prec = 3.0*abs(sprec) + 3.0*abs(eprec) + dprec + lprec

		# top left corner on the lower face is at voxel 0,0,0
		val = sprec*(shrunk[1,0,0] + shrunk[0,1,0] + shrunk[0,0,1])
		val = val + eprec*(shrunk[1,0,1] + shrunk[0,1,1] + shrunk[1,1,0])
		val = val + dprec*shrunk[1,1,1]
		val = val + lprec*observed[0,0,0]
		shrunk[0,0,0] = val/prec
		diffs[0,0,0] = fabs_c(shrunk[0,0,0] - diffs[0,0,0])

		# top left corner on the upper face is at voxel 0,0,P-1
		val = sprec*(shrunk[1,0,P-1] + shrunk[0,1,P-1] + shrunk[0,0,P-2])
		val = val + eprec*(shrunk[1,0,P-2] + shrunk[0,1,P-2] + shrunk[1,1,P-1])
		val = val + dprec*shrunk[1,1,P-2]
		val = val + lprec*observed[0,0,P-1]
		shrunk[0,0,P-1] = val/prec
		diffs[0,0,P-1] = fabs_c(shrunk[0,0,P-1] - diffs[0,0,P-1])

		#------------------------------------------------

		# top right corner on the lower face is at voxel 0,N-1,0
		val = sprec*(shrunk[1,N-1,0] + shrunk[0,N-2,0] + shrunk[0,N-1,1])
		val = val + eprec*(shrunk[1,N-1,1] + shrunk[0,N-2,1] + shrunk[1,N-2,0])
		val = val + dprec*shrunk[1,N-2,1]
		val = val + lprec*observed[0,N-1,0]
		shrunk[0,N-1,0] = val/prec
		diffs[0,N-1,0] = fabs_c(shrunk[0,N-1,0] - diffs[0,N-1,0])

		# top right corner on the upper face is at voxel 0,N-1,P-1
		val = sprec*(shrunk[1,N-1,P-1] + shrunk[0,N-2,P-1] + shrunk[0,N-1,P-2])
		val = val + eprec*(shrunk[1,N-1,P-2] + shrunk[0,N-2,P-2] +
						shrunk[1,N-2,P-1])
		val = val + dprec*shrunk[1,N-2,P-2]
		val = val + lprec*observed[0,N-1,P-1]
		shrunk[0,N-1,P-1] = val/prec
		diffs[0,N-1,P-1] = fabs_c(shrunk[0,N-1,P-1] - diffs[0,N-1,P-1])

		#------------------------------------------------

		# bottom left corner on the lower face is at voxel M-1,0,0
		val = sprec*(shrunk[M-2,0,0] + shrunk[M-1,1,0] + shrunk[M-1,0,1])
		val = val + eprec*(shrunk[M-2,0,1] + shrunk[M-1,1,1] + shrunk[M-2,1,0])
		val = val + dprec*shrunk[M-2,1,1]
		val = val + lprec*observed[M-1,0,0]
		shrunk[M-1,0,0] = val/prec
		diffs[M-1,0,0] = fabs_c(shrunk[M-1,0,0] - diffs[M-1,0,0])

		# bottom left corner on the upper face is at voxel M-1,0,P-1
		val = sprec*(shrunk[M-2,0,P-1] + shrunk[M-1,1,P-1] + shrunk[M-1,0,P-2])
		val = val + eprec*(shrunk[M-2,0,P-2] + shrunk[M-1,1,P-2] +
						shrunk[M-2,1,P-1])
		val = val + dprec*shrunk[M-2,1,P-2]
		val = val + lprec*observed[M-1,0,P-1]
		shrunk[M-1,0,P-1] = val/prec
		diffs[M-1,0,P-1] = fabs_c(shrunk[M-1,0,P-1] - diffs[M-1,0,P-1])

		#------------------------------------------------
		# last corner to do is the bottom right

		# bottom right corner on the lower face is at voxel M-1,N-1,0
		val = sprec*(shrunk[M-2,N-1,0] + shrunk[M-1,N-2,0] + shrunk[M-1,N-1,1])
		val = val + eprec*(shrunk[M-2,N-1,1] + shrunk[M-1,N-2,1] +
						eprec*shrunk[M-2,N-2,0])
		val = val + dprec*shrunk[M-2,N-2,1]
		val = val + lprec*observed[M-1,N-1,0]
		shrunk[M-1,N-1,0] = val/prec
		diffs[M-1,N-1,0] = fabs_c(shrunk[M-1,N-1,0] - diffs[M-1,N-1,0])

		# bottom right corner on the upper face is at voxel M-1,N-1,P-1
		val = sprec*(shrunk[M-2,N-1,P-1] + shrunk[M-1,N-2,P-1]
						+ shrunk[M-1,N-1,P-2])
		val = val + eprec*(shrunk[M-2,N-1,P-2] + shrunk[M-1,N-2,P-2]
						+ shrunk[M-2,N-2,P-1])
		val = val + dprec*shrunk[M-2,N-2,P-2]
		val = val + lprec*observed[M-1,N-1,P-1]
		shrunk[M-1,N-1,P-1] = val/prec
		diffs[M-1,N-1,P-1] = fabs_c(shrunk[M-1,N-1,P-1] - diffs[M-1,N-1,P-1])

		#------------------------------------------------
		#------------------------------------------------

		# edges
		prec = 4.0*abs(sprec) + 5.0*abs(eprec) +2*abs(dprec) + lprec


		# left side edge along the bottom face
		for i in range(1,M-1):
			val = sprec*(shrunk[i-1,0,0] + shrunk[i+1,0,0])
			val = val + sprec*(shrunk[i,1,0] + shrunk[i,0,1])

			val = val + eprec*(shrunk[i-1,0,1] + shrunk[i+1,0,1])
			val = val + eprec*(shrunk[i-1,1,0] + shrunk[i+1,1,0])
			val = val + eprec*shrunk[i,1,1]

			val = val + dprec*(shrunk[i-1,1,1] + shrunk[i+1,1,1])
			val = val + lprec*observed[i,0,0]

			shrunk[i,0,0] = val/prec
			diffs[i,0,0] = fabs_c(shrunk[i,0,0] - diffs[i,0,0])

#		# left side edge along the top face
		for i in range(1,M-1):
			val = sprec*(shrunk[i-1,0,P-1] + shrunk[i+1,0,P-1])
			val = val + sprec*(shrunk[i,1,P-1] + shrunk[i,0,P-2])

			val = val + eprec*(shrunk[i-1,0,P-2] + shrunk[i+1,0,P-2])
			val = val + eprec*(shrunk[i-1,1,P-1] + shrunk[i+1,1,P-1])
			val = val + eprec*shrunk[i,1,P-2]

			val = val + dprec*(shrunk[i-1,1,P-2] + shrunk[i+1,1,P-2])
			val = val + lprec*observed[i,0,P-1]

			shrunk[i,0,P-1] = val/prec
			diffs[i,0,P-1] = fabs_c(shrunk[i,0,P-1] - diffs[i,0,P-1])


		# right side edge along the bottom face
		for i in range(1,M-1):
			val = sprec*(shrunk[i-1,N-1,0] + shrunk[i+1,N-1,0])
			val = val + sprec*(shrunk[i,N-2,0] + shrunk[i,N-1,1])

			val = val + eprec*(shrunk[i-1,N-1,1] + shrunk[i+1,N-1,1])
			val = val + eprec*(shrunk[i-1,N-2,0] + shrunk[i+1,N-2,0])
			val = val + eprec*shrunk[i,N-2,1]

			val = val + dprec*(shrunk[i-1,N-2,1] + shrunk[i+1,N-2,1])
			val = val + lprec*observed[i,N-1,0]

			shrunk[i,N-1,0] = val/prec
			diffs[i,N-1,0] = fabs_c(shrunk[i,N-1,0] - diffs[i,N-1,0])

		# right side edge along the top face
		for i in range(1,M-1):
			val = sprec*(shrunk[i-1,N-1,P-1] + shrunk[i+1,N-1,P-1])
			val = val + sprec*(shrunk[i,N-2,P-1] + shrunk[i,N-1,P-2])

			val = val + eprec*(shrunk[i-1,N-1,P-2] + shrunk[i+1,N-1,P-2])
			val = val + eprec*(shrunk[i-1,N-2,P-1] + shrunk[i+1,N-2,P-1])
			val = val + eprec*shrunk[i,N-2,P-2]

			val = val + dprec*(shrunk[i-1,N-2,P-2] + shrunk[i+1,N-2,P-2])
			val = val + lprec*observed[i,N-1,P-1]

			shrunk[i,N-1,P-1] = val/prec
			diffs[i,N-1,P-1] = fabs_c(shrunk[i,N-1,P-1] - diffs[i,N-1,P-1])


		# top side edge along the bottom face
		for i in range(1,N-1):
			val = sprec*(shrunk[0,i-1,0] + shrunk[0,i+1,0])
			val = val + sprec*(shrunk[1,i,0] + shrunk[0,i,1])

			val = val + eprec*(shrunk[0,i-1,1] + shrunk[0,i+1,1])
			val = val + eprec*(shrunk[1,i-1,0] + shrunk[1,i+1,0])
			val = val + eprec*shrunk[1,i,1]

			val = val + dprec*(shrunk[1,i-1,1] + shrunk[1,i+1,1])
			val = val + lprec*observed[0,i,0]

			shrunk[0,i,0] = val/prec
			diffs[0,i,0] = fabs_c(shrunk[0,i,0] - diffs[0,i,0])


		# top side edge along the top face
		for i in range(1,N-1):
			val = sprec*(shrunk[0,i-1,P-1] + shrunk[0,i+1,P-1])
			val = val + sprec*(shrunk[1,i,P-1] + shrunk[0,i,P-2])

			val = val + eprec*(shrunk[0,i-1,P-2] + shrunk[0,i+1,P-2])
			val = val + eprec*(shrunk[1,i-1,P-1] + shrunk[1,i+1,P-1])
			val = val + eprec*shrunk[1,i,P-2]

			val = val + dprec*(shrunk[1,i-1,P-2] + shrunk[1,i+1,P-2])
			val = val + lprec*observed[0,i,P-1]

			shrunk[0,i,P-1] = val/prec
			diffs[0,i,P-1] = fabs_c(shrunk[0,i,P-1] - diffs[0,i,P-1])


		# bottom side edge along the bottom face
		for i in range(1,N-1):
			val = sprec*(shrunk[M-1,i-1,0] + shrunk[M-1,i+1,0])
			val = val + sprec*(shrunk[M-2,i,0] + shrunk[M-1,i,1])

			val = val + eprec*(shrunk[M-1,i-1,1] + shrunk[M-1,i+1,1])
			val = val + eprec*(shrunk[M-2,i-1,0] + shrunk[M-2,i+1,0])
			val = val + eprec*shrunk[M-2,i,1]

			val = val + dprec*(shrunk[M-2,i-1,1] + shrunk[M-2,i+1,1])
			val = val + lprec*observed[M-1,i,0]

			shrunk[M-1,i,0] = val/prec
			diffs[M-1,i,0] = fabs_c(shrunk[M-1,i,0] - diffs[M-1,i,0])


		# bottom side edge along the top face
		for i in range(1,N-1):
			val = sprec*(shrunk[M-1,i-1,P-1] + shrunk[M-1,i+1,P-1])
			val = val + sprec*(shrunk[M-2,i,P-1] + shrunk[M-1,i,P-2])

			val = val + eprec*(shrunk[M-1,i-1,P-2] + shrunk[M-1,i+1,P-2])
			val = val + eprec*(shrunk[M-2,i-1,P-1] + shrunk[M-2,i+1,P-1])
			val = val + eprec*shrunk[M-2,i,P-2]

			val = val + dprec*(shrunk[M-2,i-1,P-2] + shrunk[M-2,i+1,P-2])
			val = val + lprec*observed[M-1,i,P-1]

			shrunk[M-1,i,P-1] = val/prec
			diffs[M-1,i,P-1] = fabs_c(shrunk[M-1,i,P-1] - diffs[M-1,i,P-1])


		# front side edge along the left face (front means at origin)
		for i in range(1,P-1):
			val = sprec*(shrunk[0,0,i-1] + shrunk[0,0,i+1])
			val = val + sprec*(shrunk[1,0,i] + shrunk[0,1,i])

			val = val + eprec*(shrunk[0,1,i-1] + shrunk[0,1,i+1])
			val = val + eprec*(shrunk[1,0,i-1] + shrunk[1,0,i+1])
			val = val + eprec*shrunk[1,1,i]

			val = val + dprec*(shrunk[1,1,i-1] + shrunk[1,1,i+1])
			val = val + lprec*observed[0,0,i]

			shrunk[0,0,i] = val/prec
			diffs[0,0,i] = fabs_c(shrunk[0,0,i] - diffs[0,0,i])


		# front side edge along the right face (front means at origin)
		for i in range(1,P-1):
			val = sprec*(shrunk[0,N-1,i-1] + shrunk[0,N-1,i+1])
			val = val + sprec*(shrunk[1,N-1,i] + shrunk[0,N-2,i])

			val = val + eprec*(shrunk[0,N-2,i-1] + shrunk[0,N-2,i+1])
			val = val + eprec*(shrunk[1,N-1,i-1] + shrunk[1,N-1,i+1])
			val = val + eprec*shrunk[1,N-2,i]

			val = val + dprec*(shrunk[1,N-2,i-1] + shrunk[1,N-2,i+1])
			val = val + lprec*observed[0,N-1,i]

			shrunk[0,N-1,i] = val/prec
			diffs[0,N-1,i] = fabs_c(shrunk[0,N-1,i] - diffs[0,N-1,i])



		# back side edge along the left face (back means away from origin)
		for i in range(1,P-1):
			val = sprec*(shrunk[M-1,0,i-1] + shrunk[M-1,0,i+1])
			val = val + sprec*(shrunk[M-2,0,i] + shrunk[M-1,1,i])

			val = val + eprec*(shrunk[M-1,1,i-1] + shrunk[M-1,1,i+1])
			val = val + eprec*(shrunk[M-2,0,i-1] + shrunk[M-2,0,i+1])
			val = val + eprec*shrunk[M-2,1,i]

			val = val + dprec*(shrunk[M-2,1,i-1] + shrunk[M-2,1,i+1])
			val = val + lprec*observed[M-1,0,i]

			shrunk[M-1,0,i] = val/prec
			diffs[M-1,0,i] = fabs_c(shrunk[M-1,0,i] - diffs[M-1,0,i])


#		# back side edge along the right face (back means away from origin)
		for i in range(1,P-1):
			val = sprec*(shrunk[M-1,N-1,i-1] + shrunk[M-1,N-1,i+1])
			val = val + sprec*(shrunk[M-2,N-1,i] + shrunk[M-1,N-2,i])

			val = val + eprec*(shrunk[M-1,N-2,i-1] + shrunk[M-1,N-2,i+1])
			val = val + eprec*(shrunk[M-2,N-1,i-1] + shrunk[M-2,N-1,i+1])
			val = val + eprec*shrunk[M-2,N-2,i]

			val = val + dprec*(shrunk[M-2,N-2,i-1] + shrunk[M-2,N-2,i+1])
			val = val + lprec*observed[M-1,N-1,i]

			shrunk[M-1,N-1,i] = val/prec
			diffs[M-1,N-1,i] = fabs_c(shrunk[M-1,N-1,i] - diffs[M-1,N-1,i])


		#------------------------------------------------

		prec = 5.0*abs(sprec) + 8.0*abs(eprec) + 4.0*abs(dprec) + lprec

		# bottom face
		for i in range(1,M-1):
			for j in range(1,N-1):
				val = sprec*(shrunk[i-1,j,0] + shrunk[i+1,j,0])
				val = val + sprec*(shrunk[i,j-1,0] + shrunk[i,j+1,0])
				val = val + sprec*shrunk[i,j,1]

				val = val + eprec*(shrunk[i-1,j-1,0] + shrunk[i+1,j-1,0])
				val = val + eprec*(shrunk[i-1,j+1,0] + shrunk[i+1,j+1,0])
				val = val + eprec*(shrunk[i-1,j,1] + shrunk[i+1,j,1])
				val = val + eprec*(shrunk[i,j-1,1] + shrunk[i,j+1,1])

				val = val + dprec*(shrunk[i-1,j-1,1] + shrunk[i+1,j-1,1])
				val = val + dprec*(shrunk[i-1,j+1,1] + shrunk[i+1,j+1,1])

				val = val + lprec*observed[i,j,0]

				shrunk[i,j,0] = val/prec
				diffs[i,j,0] = fabs_c(shrunk[i,j,0] - diffs[i,j,0])


		# top face
		for i in range(1,M-1):
			for j in range(1,N-1):
				val = sprec*(shrunk[i-1,j,P-1] + shrunk[i+1,j,P-1])
				val = val + sprec*(shrunk[i,j-1,P-1] + shrunk[i,j+1,P-1])
				val = val + sprec*shrunk[i,j,P-2]

				val = val + eprec*(shrunk[i-1,j-1,P-1] + shrunk[i+1,j-1,P-1])
				val = val + eprec*(shrunk[i-1,j+1,P-1] + shrunk[i+1,j+1,P-1])
				val = val + eprec*(shrunk[i-1,j,P-2] + shrunk[i+1,j,P-2])
				val = val + eprec*(shrunk[i,j-1,P-2] + shrunk[i,j+1,P-2])

				val = val + dprec*(shrunk[i-1,j-1,P-2] + shrunk[i+1,j-1,P-2])
				val = val + dprec*(shrunk[i-1,j+1,P-2] + shrunk[i+1,j+1,P-2])

				val = val + lprec*observed[i,j,P-1]

				shrunk[i,j,P-1] = val/prec
				diffs[i,j,P-1] = fabs_c(shrunk[i,j,P-1] - diffs[i,j,P-1])


		# left face
		for i in range(1,M-1):
			for k in range(1,P-1):
				val = sprec*(shrunk[i-1,0,k] + shrunk[i+1,0,k])
				val = val + sprec*(shrunk[i,0,k-1] + shrunk[i,0,k+1])
				val = val + sprec*shrunk[i,1,k]

				val = val + eprec*(shrunk[i-1,0,k-1] + shrunk[i+1,0,k-1])
				val = val + eprec*(shrunk[i-1,0,k+1] + shrunk[i+1,0,k+1])
				val = val + eprec*(shrunk[i-1,1,k] + shrunk[i+1,1,k])
				val = val + eprec*(shrunk[i,1,k-1] + shrunk[i,1,k+1])

				val = val + dprec*(shrunk[i-1,1,k-1] + shrunk[i+1,1,k-1])
				val = val + dprec*(shrunk[i-1,1,k+1] + shrunk[i+1,1,k+1])

				val = val + lprec*observed[i,0,k]

				shrunk[i,0,k] = val/prec
				diffs[i,0,k] = fabs_c(shrunk[i,0,k] - diffs[i,0,k])


		# right face
		for i in range(1,M-1):
			for k in range(1,P-1):
				val = sprec*(shrunk[i-1,N-1,k] + shrunk[i+1,N-1,k])
				val = val + sprec*(shrunk[i,N-1,k-1] + shrunk[i,N-1,k+1])
				val = val + sprec*shrunk[i,N-2,k]

				val = val + eprec*(shrunk[i-1,N-1,k-1] + shrunk[i+1,N-1,k-1])
				val = val + eprec*(shrunk[i-1,N-1,k+1] + shrunk[i+1,N-1,k+1])
				val = val + eprec*(shrunk[i-1,N-2,k] + shrunk[i+1,N-2,k])
				val = val + eprec*(shrunk[i,N-2,k-1] + shrunk[i,N-2,k+1])

				val = val + dprec*(shrunk[i-1,N-2,k-1] + shrunk[i+1,N-2,k-1])
				val = val + dprec*(shrunk[i-1,N-2,k+1] + shrunk[i+1,N-2,k+1])

				val = val + lprec*observed[i,N-1,k]

				shrunk[i,N-1,k] = val/prec
				diffs[i,N-1,k] = fabs_c(shrunk[i,N-1,k] - diffs[i,N-1,k])


		# back face
		for j in range(1,N-1):
			for k in range(1,P-1):
				val = sprec*(shrunk[0,j-1,k] + shrunk[0,j+1,k])
				val = val + sprec*(shrunk[0,j,k-1] + shrunk[0,j,k+1])
				val = val + sprec*shrunk[1,j,k]

				val = val + eprec*(shrunk[0,j-1,k-1] + shrunk[0,j+1,k-1])
				val = val + eprec*(shrunk[0,j-1,k+1] + shrunk[0,j+1,k+1])
				val = val + eprec*(shrunk[1,j-1,k] + shrunk[1,j+1,k])
				val = val + eprec*(shrunk[1,j,k-1] + shrunk[1,j,k+1])

				val = val + dprec*(shrunk[1,j-1,k-1] + shrunk[1,j+1,k-1])
				val = val + dprec*(shrunk[1,j-1,k+1] + shrunk[1,j+1,k+1])

				val = val + lprec*observed[0,j,k]

				shrunk[0,j,k] = val/prec
				diffs[0,j,k] = fabs_c(shrunk[0,j,k] - diffs[0,j,k])



#herehere
		#------------------------------------------------

		# middle
		prec = 6.0*abs(sprec) + 12*abs(eprec) + 8*abs(dprec) + lprec


		for i in range(1,M-1):
			for j in range(1,N-1):
				for k in range(1,P-1):
					val = sprec*(shrunk[i,j,k-1] + shrunk[i,j,k+1])
					val = val + sprec*(shrunk[i,j-1,k] + shrunk[i,j+1,k])
					val = val + sprec*(shrunk[i-1,j,k] + shrunk[i+1,j,k])

					val = val + eprec*(shrunk[i,j-1,k-1] + shrunk[i,j+1,k-1])
					val = val + eprec*(shrunk[i-1,j,k-1] + shrunk[i+1,j,k-1])

					val = val + eprec*(shrunk[i-1,j-1,k] + shrunk[i-1,j+1,k])
					val = val + eprec*(shrunk[i+1,j-1,k] + shrunk[i+1,j+1,k])

					val = val + eprec*(shrunk[i,j-1,k+1] + shrunk[i,j+1,k+1])
					val = val + eprec*(shrunk[i-1,j,k+1] + shrunk[i+1,j,k+1])

					val = val + dprec*(shrunk[i-1,j-1,k-1]+shrunk[i-1,j+1,k-1])
					val = val + dprec*(shrunk[i+1,j-1,k-1]+shrunk[i+1,j+1,k-1])
					val = val + dprec*(shrunk[i-1,j-1,k+1]+shrunk[i-1,j+1,k+1])
					val = val + dprec*(shrunk[i+1,j-1,k+1]+shrunk[i+1,j+1,k+1])

					val = val + lprec*observed[i,j,k]

					shrunk[i,j,k] = val/prec
					diffs[i,j,k] = fabs_c(shrunk[i,j,k] - diffs[i,j,k])

		if numpy.abs(diffs).max() < converged: break

	return shrunk

