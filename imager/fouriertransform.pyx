cdef extern from "complex.h":
     pass

import numpy as np
cimport numpy as np


ctypedef np.int_t INT
ctypedef np.float64_t FLOAT
ctypedef np.complex128_t COMPLEX

def ft_vis(COMPLEX[:] vis_fi, FLOAT[:, :] qvector, FLOAT[:, :] blvector, INT[:] blredundancy):

    cdef int qi, bi
    cdef int rd
    cdef FLOAT[:] q, bl

    ft_vis = np.zeros(qvector.shape[0], dtype=np.complex128)

    for qi in range(qvector.shape[0]):
        q = qvector[qi]
        for bi in range(blvector.shape[0]):
            bl = blvector[bi]
            rd = blredundancy[bi]
            ft_vis[qi] += rd * vis_fi[bi] * np.exp(-complex(0.0, 1.0) * (q[0] * bl[0] + q[1] * bl[1]))

    ft_vis /= np.sum(blredundancy)

    return ft_vis.real
