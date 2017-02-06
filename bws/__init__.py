
from mrf_wavelet_icm import shrink_mrf2_redescend as bws2d_redescend
from mrf_wavelet_icm import shrink_mrf2 as bws2d

from mrf_wavelet_icm import shrink_mrf1_redescend as bws1d_redescend
from mrf_wavelet_icm import shrink_mrf1 as bws1d
from mrf_wavelet_icm import shrink_mrf1_stagger as bws1d_stagger
from mrf_wavelet_icm import shrink_mrf1_redescend
import test1D
#from mrf_wavelet_icm import shrink_mrf1_icm as bws1d


__all__ = ["bws1d","bws1d_stagger","bws2d","bws2d_redescend","test1D"]

del mrf_wavelet_icm
