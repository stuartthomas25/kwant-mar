'''
Current in Biased JJ with MAR

Averin, D., & Bardas, A. (1995). ac Josephson effect in a single quantum channel. Physical Review Letters, 75(9), 1831–1834. https://doi.org/10.1103/PhysRevLett.75.1831

Setiawan, F., Cole, W. S., Sau, J. D., & Das Sarma, S. (2017). Transport in superconductor-normal metal-superconductor tunneling structures: Spinful p -wave and spin-orbit-coupled topological wires. Physical Review B, 95(17), 1–16. https://doi.org/10.1103/PhysRevB.95.174515

Authors:
Stuart Thomas
(snthomas01@email.wm.edu)

2019
'''

import sys
import os
import numpy as np
import tinyarray as ta
import pickle
import dill
from functools import lru_cache, partial
from matplotlib import pyplot as plt
import itertools
from scipy.optimize import brute
import vquad
import kwant
from stus_tools import *
import threading
import asyncio
from copy import deepcopy
import adaptive
import quickle

from scipy.sparse import csr_matrix
from scipy.linalg import sqrtm, block_diag
from scipy.sparse.linalg import spsolve, lsqr
from scipy.interpolate import interp1d
from scipy.stats import linregress

from numpy import linalg as nla

import ipywidgets
from IPython.display import display

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# these cause numpy to not parallelize, so the multiprocessing can parallelize more efficiently
os.environ['MKL_NUM_THREADS'] = '1'
#TODO: move these out of this module


# define Pauli-matrices for convenience
x, y, z = (1, 2, 3)


class PauliMatrices:
    def __init__(self, dim=1):
        self.__dim = dim
        self.__s = [
            ta.array([[1, 0], [0, 1]]),
            ta.array([[0, 1], [1, 0]]),
            ta.array([[0, -1j], [1j, 0]]),
            ta.array([[1, 0], [0, -1]])
        ]

    def __kron(self, *args):
        # define kronecker product function for >2 matrices, using associativity of kronecker product.
        if len(args) == 1:
            return args[0]
        else:
            ret = np.kron(args[0], args[1])
            for a in args[2:]:
                ret = np.kron(ret, a)
            return ret

    def __getitem__(self, key):
        if type(key) is not tuple:
            key = (key,)
        return ta.array(self.__kron(*(self.__s[i] for i in key)))


tau = PauliMatrices()


# for some backwards compatibilty
s_0 = tau[0]
s_x = tau[x]
s_y = tau[y]
s_z = tau[z]


e_mat = 0.5*(tau[0]+tau[z])
h_mat = 0.5*(tau[0]-tau[z])


def d(i, j): return 1 if i == j else 0  # kronecker delta


def kron(*args):
    # define kronecker product function for >2 matrices, using associativity of kronecker product.
    if len(args) == 1:
        return args[0]
    else:
        ret = np.kron(args[0], args[1])
        for a in args[2:]:
            ret = np.kron(ret, a)
        return ret


# creates a vertical basis vector ([[0],[0],[1],[0],...[0]]) with n components and a 1 element at index i
def e_(n, i):
    ret = np.zeros((n, 1))
    ret[i] = 1
    return np.asmatrix(ret, dtype=np.int)


spin = True
if spin:
    identity = tau[0, 0]  # tau[0] for spinless case
    sigma_0 = np.identity(2)
else:
    identity = tau[0]
    sigma_0 = np.identity(1)
lms = 4*identity.shape[0]  # level matrix size


@lru_cache(maxsize=8)
def M(interface, region, propdir, identity=identity):
    return kron(e_(2, interface), e_(2, region).T, e_(2, propdir).T, identity)

################
#  SystemInfo  #
################


L, R, N = (0, 1, 2)


class SystemInfo:
    kwant = __import__('kwant')

    def __init__(self, L=None, N=None, R=None, E_scale=1, band_min=0):
        self.use_Z = False
        self.systs = [L, R, N]
        self.Es = None
        self.interpolators = [None, None, None]
        self.E_scale = E_scale
        self.band_min = band_min
        self._Ebounds = None
        self.Z = None
        self.test_flag = False

        self.r_func = None
        self.t_func = None

        # self.async_results = []
        # self._progress_bar = None
        # self._pbthread = None

    @staticmethod
    def _rotate_smat(mat, wf_in, wf_out):
        '''
        rotate a scattering/transission/reflection matrix into a consistent basis.

        Parameters
        -----------
        - mat : ndarray (nxn)
            the scattering matrix, produced by kwant.solvers.common.SMatrix
        - wf_in : ndarray (mxn)
            matrix with incoming wavefunctions as columns, as produced by kwant.physics.PropogatingModes.wave_functions
        - wf_out : ndarray (mxn)
            matrix with outgoing wavefunctions as columns, as produced by kwant.physics.PropogatingModes.wave_functions

        Returns
        -----------
        - ndarray (mxm)
            resulting matrix in the basis given by wf_in and wf_out.
        Notes
        -----------
        If m<n, (n-m) orthogonal modes are added with with complete reflection so the resultant matrix is mxm.

        If the the Hamiltonian has some degeneracy, this must be address by a conservation law when building the system. 
        Otherwise, these degenerate states may form unorthogonal propogating modes.

        '''
        if mat.size == 0:
            raise Exception("Matrix is empty")
        
        def normalize_columns(M):
            return M / np.tile(np.linalg.norm(M, axis=0), (M.shape[0], 1))
        
        def is_unitary(M):
            return np.allclose(dagger(M)@M, np.eye(M.shape[1]))
        
        wf_in = normalize_columns(wf_in)
        wf_out = normalize_columns(wf_out) 
        if not (is_unitary(wf_in) and is_unitary(wf_out)):
            raise Exception("Propogating modes must be orthogonal. This may be the caused by a degeneracy unaddressed by a kwant conservation law.")

        # extend modes to span all Nambu space, even with modes that do not actually exists with a QR decomposition.
        # For "modes" that do not actually exist in the leads, we implement complete reflection.
        # This helps facilitate the matrix solution in calculate_current().
        # num_pmodes = wf_in.shape[0]-wf_in.shape[1]
        # if num_pmodes > 0:
        #     wf_in = np.hstack((wf_in, 
        #             np.linalg.qr(wf_in,   mode='complete')[0][:, num_pmodes:]))
        #     wf_out = np.hstack((wf_out, 
        #             np.linalg.qr(np.conj(wf_out), mode='complete')[0][:, num_pmodes:]))
        #     mat = block_diag(mat, np.identity(num_pmodes))

        #r = wf_out @ mat @ np.linalg.inv(wf_in)
        #return r

        r = wf_out @ mat @ dagger(wf_in)
        t = sqrtm(np.identity(wf_out.shape[0]) - wf_out @ dagger(wf_out)) #if a particle has no modes in the lead, return 100% reflection, not 0%
        return r + t

    def _refl_LR(self, d, E, params, complete=True):
        '''
        create a reflection matrix by rotating scattering matrices into a consistent basis. Note that while the kwant basis is arbitrary, kwant perserves independent modes for conservation laws.
        - kwant_smat: kwant.SMatrix object
        - d: either -1 or 1 for left/right scattering interface

        '''
        if self.interpolated(d):
            if len(params) > 0:
                print(
                    'warning: params passed to an interpolated function, values will be ignored')
            if self._Ebounds[0] <= E <= self._Ebounds[1]:
                return self.interpolators[d](E)
            else:
                raise Exception(
                    f"Energy {E} is outside bounds ({self._Ebounds})")

        if self.systs[d] is None:
            raise Exception("Right system is not defined")

        try:
            ksmat = kwant.smatrix(self.systs[d], E, params=params)
        except ValueError:
            ksmat = kwant.smatrix(self.systs[d], E + 1e-4, params=params)  # this is to counter a bug in kwant

        lead_num = 1 if d == L else 0
        modes = ksmat.lead_info[lead_num].wave_functions
        wf_in, wf_out = np.split(modes, 2, axis=1)
        krmat = ksmat.submatrix(lead_num, lead_num)

        return SystemInfo._rotate_smat(krmat, wf_in, wf_out)

    def refl_L(self, E, params={}):
        return self._refl_LR(L, E, params)

    def refl_R(self, E, params={}):
        return self._refl_LR(R, E, params)

    def smat_N(self, E, params={}):
        if self.interpolated('N'):
            if len(params) > 0:
                print(
                    'warning: params passed to an interpolated function, values will be ignored')
            if self._Ebounds[0] <= E <= self._Ebounds[1]:
                return self.interpolators[N](E)
            else:
                raise Exception(
                    f"Energy {E} is outside bounds ({self._Ebounds})")

        if self.systs[N] is None:
            raise Exception("Normal system is not defined")

        try:
            ksmat = kwant.smatrix(self.systs[N], E, params=params)
        except ValueError:
            ksmat = kwant.smatrix(self.systs[N], E + 1e-4, params=params)  # this is to counter a bug in kwant

        left_modes = ksmat.lead_info[0].wave_functions
        right_modes = ksmat.lead_info[1].wave_functions
        # renormalize modes #TODO: why do I need this??
        left_modes /= np.tile(np.linalg.norm(left_modes,
                                             axis=0), (left_modes.shape[0], 1))
        right_modes /= np.tile(np.linalg.norm(right_modes,
                                              axis=0), (right_modes.shape[0], 1))
        wf_in_l, wf_out_l = np.split(left_modes,  2, axis=1)
        wf_in_r, wf_out_r = np.split(right_modes, 2, axis=1)

        wf_in = block_diag(wf_in_l,  wf_in_r)
        wf_out = block_diag(wf_out_l, wf_out_r)

        return SystemInfo._rotate_smat(ksmat.data, wf_in, wf_out)

    def get_interpolation_bounds(self, minE0, maxV):
        """
        Calculates interpolation bounds to be used in SMatrices.interpolate() for use in calculate_current().

        Parameters:
        - minE0 (float or int):    the lower bound of the energy integral, usually taken to be Fermi energy in the middle region
        - maxV (float or int):     the maximum voltage that will be used in calculate_current

        Returns:
        - (tuple of floats): bounds to be used in SMatrices.interpolate()

        """

        def fmax(args): return -args[0]-args[1] * \
            _calculate_nmax(args[1]/self.E_scale)
        def fmin(args): return args[0]+args[1] * \
            _calculate_nmax(args[1]/self.E_scale)
        # find max/min of E0+neV, add/sub 1 E_scale for good measure (and because brute without finish can be a little off)
        # 0.2*E_scale is subtracted from minE0 to ensure a numerical artifact is captured and later cancelled.
        maxE = -brute(fmax, ((minE0-0.2*self.E_scale-maxV, 0),
                             (-maxV, maxV)), finish=None, full_output=True)[1]+1
        minE = brute(fmin, ((minE0-0.2*self.E_scale-maxV, 0),
                            (-maxV, maxV)), finish=None, full_output=True)[1]-1
        return minE, maxE

    @staticmethod
    def _validate_key(key):
        for i in key:
            if not i in 'LNR':
                raise Exception(f"invalid key '{i}'")
    # Interpolation
    @staticmethod
    def complexify(new_a):
        return new_a[:,:,0] + 1j*new_a[:,:,1]
    @staticmethod
    def decomplexify(a):
        new_a = np.empty(a.shape+(2,), dtype=np.float)
        new_a[:,:,0] = np.real(a)
        new_a[:,:,1] = np.imag(a)
        return new_a
        
    def get_learners(self, key, bounds, dview):
        SystemInfo._validate_key(key)
        self._Ebounds = bounds
        def decomp(f):
            def wrapper(E): 
                return SystemInfo.decomplexify(f(E)) #adaptive does not work with complex matrices
            return wrapper

        funcs = {'L': decomp(self.refl_L), 'N': decomp(self.smat_N), 'R': decomp(self.refl_R)}
        hashed_funcs = {k:quickle.quickle(v, dview) for k,v in funcs.items()}
        return [adaptive.Learner1D(hashed_funcs[k], bounds=bounds) for k in key]

    def update_interpolators(self, key, learners):
        SystemInfo._validate_key(key)
        for k, l in zip(key, learners):
            energies = list(l.data.keys())
            mats = [SystemInfo.complexify(mat) for mat in l.data.values()]
            self.interpolators[eval(k)] = interp1d(energies, mats, axis=0)
        
    def interpolated(self, key):
        if type(key) is int:
            return not self.interpolators[key] is None
        else:
            SystemInfo._validate_key(key)
            for i in key:
                return not self.interpolators[eval(i)] is None
    
    def interpolate(self, key, bounds, params={}, resolution=100, executor=None, progress_bar=True, threads=1):
        # key must be string with 'L', 'N', and/or 'R' (e.g. 'LR')

        # os.environ['OMP_NUM_THREADS'] = str(threads)
        # os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
        # # these cause numpy to not parallelize, so the multiprocessing can parallelize more efficiently
        # os.environ['MKL_NUM_THREADS'] = str(threads)

        self.Es = np.arange(*bounds, self.E_scale/resolution)
        self._Ebounds = (np.amin(self.Es), np.amax(self.Es))
        if executor is not None:
            lview = executor.load_balanced_view()
            map_func = lview.map
        else:
            map_func = map


        SystemInfo._validate_key(key)

        #since async_results can't pass objects like progress bars and threads, create a clone of self without those

        pseudo_self = deepcopy(self)
        funcs = {'L': pseudo_self.refl_L, 'N': pseudo_self.smat_N, 'R': pseudo_self.refl_R}

        for i in key:
            async_result = map_func(partial(funcs[i], params=params), self.Es)
            self.async_results.append(async_result)
            loop = asyncio.get_event_loop()
            task = loop.create_task(self._finish_interpolation(
                i, async_result))

        if progress_bar and executor is not None:
            self._progress_bar = ipywidgets.IntProgress()
            self._progress_bar.max = len(self.Es) * len(key)
            display(self._progress_bar)
            self._pbthread = threading.Thread(
                target=self._iter_progress_bar)
            self._pbthread.start()

    def _iter_progress_bar(self):
        while not all([res.ready() for res in self.async_results]):
            self._progress_bar.value = sum([res.progress for res in self.async_results])
            time.sleep(.1)
        self._progress_bar.close()

    async def _finish_interpolation(self, key, async_result):
        if type(async_result) is map:
            data = np.array(list(async_result))
            self.interpolators[eval(key)] = interp1d(self.Es, data, axis=0)
        else:
            while not async_result.done():
                await asyncio.sleep(0.1)
            if async_result.successful():
                #print(f'completed in {self.async_result.wall_time:.1f} secs')
                #print(f'speedup: {self.async_result.serial_time / self.async_result.wall_time:.1f}')

                data = np.array(async_result.result())
                self.interpolators[eval(key)] = interp1d(self.Es, data, axis=0)
            else:
                self.async_result.result()

    def __getstate__(self):
        pickle_attrs = dict(async_results = [], _pbthread = [], _progress_bar = [])
        return {k : (v if k not in pickle_attrs else pickle_attrs[k]) for k, v in self.__dict__.items()}
        

#######################
# Current Calculation #
#######################


def _calculate_nmax(V, NMAX=6):
    if V == 0.0:
        return 0
    greater_voltage = V
    return int(np.exp(-(greater_voltage+0.7)**10)*NMAX+7)


def _S_N(n, n_, r, t, edge=0, particles=(True, True)):  # Scattering matrix in normal region
    sp = sigma_0.shape[0]
    S = np.zeros((4*sp,)*2, dtype=np.complex)
    if abs(n_-n) < 2:
        if not type(r) is tuple and not type(t) is tuple:
            r = (r,)
            t = (t,)
            
        for i in range(len(r)):
            _r, _t = (r[i], t[i])
            S_e = np.matrix([[        _r  *   d(n, n_),             _t  * d(n, n_-1)],
                             [np.conj(_t) * d(n, n_+1),    -np.conj(_r) *   d(n, n_)]])
            S_h = dagger(S_e)
            if edge == -1:
                S_e[1, 1] = -1
                S_h[0, 0] = -1
            elif edge == 1:
                S_e[0, 0] = -1
                S_h[1, 1] = -1
            for j in range(sp//len(r)): #spin states per value of i
                for b,S_n,k in zip(particles, [S_e,S_h], range(2)):
                    if b:
                        S[j+i+(sp*k)::2*sp, j+i+(sp*k)::2*sp] = S_n

    return S

def _kwant_S_N(m,n,E,V,sysinfo,edge=0):
    sp = sigma_0.shape[0]
    S = np.zeros((4*sp,)*2, dtype=np.complex)
    if abs(n-m)>1:
        return S
    
    eL = slice(0*sp,1*sp)
    hL = slice(1*sp,2*sp)
    eR = slice(2*sp,3*sp)
    hR = slice(3*sp,4*sp)

    def apply_slice(full_smat,*args):
        for arg in args:
            S[arg]=full_smat[arg]
    # def check_probabilities(S, precision=2):
    #     if not np.all(np.round(np.sum(np.abs(S), axis=0), precision)==1)
    #     except Exception as e:
    #         print(np.sum(np.abs(S), axis=0))
    #         raise e

    if m==n:
        ep = E + m * V
        full_smat_inc = sysinfo.smat_N(E + (m+0.5) * V) if edge !=  1 else -np.identity(S.shape[0])
        full_smat_dec = sysinfo.smat_N(E + (m-0.5) * V) if edge != -1 else -np.identity(S.shape[0])

        apply_slice(full_smat_inc, (eL,eL), (hR,hR))
        apply_slice(full_smat_dec, (eR,eR), (hL,hL))
    else:
        full_smat = sysinfo.smat_N(E + (m+n)/2 * V)
        if m<n:
            slices = [(eL,eR), (hR,hL)]
            apply_slice(full_smat, (eL,eR), (hR,hL))
        else: #m>n
            apply_slice(full_smat, (eR,eL), (hL,hR))

    return S


@lru_cache(maxsize=32)
def _big_S_N(E, V, sysinfo, nmax):
    Z = sysinfo.Z
    if sysinfo.use_Z:
        if sysinfo.r_func is None or sysinfo.t_func is None:
            r = lambda E: Z/np.sqrt(1+Z**2)
            t = lambda E: 1/np.sqrt(1+Z**2)
        else:
            r = sysinfo.r_func
            t = sysinfo.t_func
        def S_N(m,n,edge=0): 
            raise Exception("This does not make physical sense")
            return _S_N(m, n, r(E+V/2), t(E+V/2), edge, particles=(True,False)) + \
                   _S_N(m, n, r(E-V/2), t(E-V/2), edge, particles=(False,True))
    else:
        def S_N(m,n,edge=0): 
            return _kwant_S_N(m,n,E,V,sysinfo,edge)

    e_levels = nmax*2+1
    size = e_levels*4*identity.shape[0]
    S = np.asmatrix(np.zeros((size, size)), dtype=np.complex)
    if nmax == 0:
        return (M(L, L, L)+M(R, R, R)).T @ S_N(0, 0) @ (M(L, L, R)+M(R, R, L)) \
            + (M(L, L, L)+M(R, R, R)).T @ (S_N(1, 0) + S_N(0, 1, 0.)) @ (M(L, L, R)+M(R, R, L)) #TODO: Use adjust energies for the scattering matrices in terms of voltage
    for i, j in np.ndindex(e_levels, e_levels):
        if i == j == 0:
            e=-1 #lowest energy
        elif i == j == e_levels-1:
            e=1 #highest energy
        else:
            e=0

        #THIS IS A BODGE
        #TODO: MAKE THIS PRETTIER AND MORE EFFICIENT, also peep a few lines above
        if sysinfo.test_flag:
            normal_S = S_N(i-nmax, j-nmax, edge=e)
        else:
            normal_S_e = S_N(i-nmax, j-nmax, edge=e)
            normal_S_h = S_N(i-nmax, j-nmax, edge=e)
            normal_S = np.zeros_like(normal_S_e)
            sp=2
            for k in range(sp): #spin states
                normal_S[k::2*sp, k::2*sp] = normal_S_e[k::2*sp, k::2*sp] 
                normal_S[k+sp::2*sp, k+sp::2*sp] = normal_S_h[k+sp::2*sp, k+sp::2*sp]

        if np.count_nonzero(normal_S) != 0:
            mat = (M(L, L, L)+M(R, R, R)).T @ normal_S @ (M(L, L, R)+M(R, R, L))
            S[i*lms:(i+1)*lms, j*lms:(j+1)*lms] += mat
    return S


def _S_(E, V, nmax, sysinfo, params):
    i_size = identity.shape[0]
    e_levels = nmax*2+1
    S = np.copy(_big_S_N(E, V, sysinfo, nmax))
    for n in range(e_levels):
        r_L = sysinfo.refl_L(E+(n-nmax)*V, params)
        r_R = sysinfo.refl_R(E+(n-nmax)*V, params)
        def slice_gen(_n, _i): return slice(
            _n*lms+_i*i_size, _n*lms+(_i+1)*i_size)
        S[slice_gen(n, 1), slice_gen(n, 0)] += r_L
        S[slice_gen(n, 2), slice_gen(n, 3)] += r_R
    return S


# for testing purposes
test_array = []


def _calc_integrand(E, V, nu, sysinfo, params, E_scale, ret_J=False, components=None, check_convergence=False):  # absoute energies
    nmax = _calculate_nmax(V/E_scale)
    try:
        S = _S_(E, V, nmax, sysinfo, params)
    except Exception as e:
        print(
            f"Exception occured at E={E:.2f}, V={V:.2f}, Z={sysinfo.Z:.2f}, E_scale={E_scale:.2f}")
        raise e
    if check_convergence:
        l, v = nla.eig(S)
        if not np.all(np.round(np.abs(l), 13)<=1):
            raise Exception("S-matrix does not converge!!")
    # represents the matrix rho_z tau_z in the Setiawan current equation.
    O_mat = -1*kron(e_mat, tau[z, z], sigma_0)
    A = csr_matrix(np.identity(S.shape[0])-S)
    e_levels = nmax*2+1
    i_size = identity.shape[0]
    if components is None:
        components = range(i_size)
    block_size = 4*i_size
    ret = 0.0
    for i in components:
        in_vec = e_(i_size, i)
        if nu == L:
            r = sysinfo.refl_L(E, params)  # get R->,L<- block of matrix
        else:
            r = sysinfo.refl_R(E, params)
        t = sqrtm(identity - r @ dagger(r))  # see appendix of Setiawan
        in_vec_N = t @ in_vec

        J_init = np.zeros((S.shape[0],), dtype=np.complex)
        if nu == L:
            J_init[block_size*nmax+i_size:block_size *
                   nmax+i_size*2] = in_vec_N.flatten()
        else:
            J_init[block_size*nmax+2*i_size:block_size *
                   nmax+3*i_size] = in_vec_N.flatten()

        with Muzzle():
            J = np.matrix(lsqr(A, J_init)[0]).T

        # sum all components, giving opposite signs to e/h and left/right propogation
        inner = np.real((J.H @ block_diag(*(O_mat,)*e_levels) @ J)[0, 0])

        # for testing purposes
        global test_array
        test_array.append([E, inner/sigma_0.shape[0], i+4*nu])

        ret += inner/sigma_0.shape[0]
    if ret_J:
        return ret, J
    else:
        return ret


def _atol(atol, Z):
    return atol / (1+Z**2)**2


def calculate_current(V, sysinfo, params={}, min_E=None, E_scale=1, tol=1e-2, ret_integrand=False, check_convergence=False):  # TODO add temperature
    """
    Calculate current in biased Josephson Junction.

    Parameters:
    - V (float or int):              voltage
    - Z (float or int):              strength of barrier in normal region
    - sysinfo (SystemInfo):              SystemInfo object containing scattering info for the system
    - (opt) min_E (float or int):          lower bound of integration, by default, 10*E_scale
    - (opt) E_scale (float or int):  a scaling factor which determines the number of energy levels to use, usually taken to be the smallest superconducting gap
    - (opt) atol (float or int):     absolute tolerance of the integral over energy
    - (opt) ret_integrand (bool):    a debugging feature

    Returns:
    - (float): total current through system
    """
    if min_E is None:
        min_E = E_scale*10
    global test_array
    test_array = []
    def integrand_L(E): return _calc_integrand(
        E, V, L, sysinfo, params, E_scale,check_convergence=check_convergence)
    def integrand_R(E): return _calc_integrand(
        E, V, R, sysinfo, params, E_scale,check_convergence=check_convergence)
    a = min_E - 0.2*E_scale  # 0.2*E_scale subtracted to capture small numerical feature
    b = 0.0
    cur = vquad.vquad(np.vectorize(integrand_L), a, b, atol=_atol(tol, sysinfo.Z))[0]\
        + vquad.vquad(np.vectorize(integrand_R), a-V,
                      b, atol=_atol(tol, sysinfo.Z))[0]
    if ret_integrand:
        return cur, test_array
    else:
        return cur


def find_GN(Vs, Is, dV=0.3):
    """
    To be used after calculate_current() to calculate GN, or the normal conductance. Assumes that dI/dV converges to GN at high voltages (Setiawan).
    Uses a linear regression between max(Vs)-dV and max(Vs).

    Parameters:
    - Vs (list or ndarray):    voltage values
    - Is (list or ndarray):    current values
    - (opt) dV (float or int): width of regression sample

    Returns:
    - (float): normal conductance
    """
    Vs = np.array(Vs)
    arg_dV = np.argmin(np.abs(Vs - (np.amax(Vs) - dV)))
    GN = linregress(Vs[arg_dV:], Is[arg_dV:]).slope
    return GN
