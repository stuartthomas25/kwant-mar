import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import matplotlib
import os
from matplotlib import pyplot as plt
import kwant
import tinyarray as ta
import scipy
from stus_tools import *
from importlib import reload
from kwant_mar import SystemInfo
import adaptive


class PauliMatrices(object):
    ta = __import__('tinyarray')
    np = __import__('numpy')
    def __init__(self,dim=1):
        self.__dim=dim
        self.__s=[
                    self.ta.array([[1, 0], [0, 1]]),
                    self.ta.array([[0, 1], [1, 0]]),
                    self.ta.array([[0, -1j], [1j, 0]]),
                    self.ta.array([[1, 0], [0, -1]])
                 ]
    def __kron(self,*args):
        #define kronecker product function for >2 matrices, using associativity of kronecker product.
        if len(args)==1:
            return args[0]
        else:
            ret = self.np.kron(args[0],args[1])
            for a in args[2:]:
                ret = self.np.kron(ret,a)
            return ret
    def __getitem__(self, key):
        if type(key) is not tuple: key=(key,)
        return self.ta.array(self.__kron(*(self.__s[i] for i in key)))

tau=PauliMatrices()
x,y,z = (1,2,3)


# # Define constants

# In[5]:



# # Make kwant systems

from types import SimpleNamespace

from kwant_mar import L,R,N

class MARSolver:
    def __init__(self, params : SimpleNamespace, E_scale=None):
        self.params = params
        minE0 = -params.mu_sm-params.V_Z_N
        E_scale = params.Delta_L if E_scale is None else E_scale
        self.maxV = 3.0*E_scale
        self.make_systems()
        self.sysinfo = SystemInfo(L = self.systs[L],
                        R = self.systs[R],
                        N = self.systs[N],
                        E_scale = E_scale,
                        band_min = minE0
                        )
        self.sysinfo.use_Z = False
        #self.generate_Z_estimate()

    def make_systems(self):
        p = self.params

        assert p.L_NS%2==0
        numpy = np

        tau = PauliMatrices()

        E_soc_scL = p.alpha_L/(2*p.a)
        E_soc_sm  = p.alpha_sm/(2*p.a)
        E_soc_scR = p.alpha_R/(2*p.a)

        #if spin is degenerate, conserve spin
        #this is necessary to properly rotate scatting matrices
        cons_law = np.diag([-2,-1,1,2]) if p.alpha_sm==0.0 else -tau[z, 0]
        #cons_law = np.diag([-2,-1,1,2])

        sym_l = kwant.TranslationalSymmetry((-p.a, 0))
        sym_r = kwant.TranslationalSymmetry((p.a, 0))

        lat = kwant.lattice.square(p.a, norbs=4)

        #make L system
        def Delta_L_f(x):
            print(x, p.Delta_L*(numpy.tanh(-p.beta*(x-p.L_NS//2+0.5)) + 1)/2)
            return p.Delta_L*(numpy.tanh(-p.beta*(x-p.L_NS//2+0.5)) + 1)/2
        sys_L = kwant.Builder()
        def onsite_sc_L(site):#,V,U,salt):
            return (2.0 * p.t_sc - p.mu_sc) * tau[3,0] + p.V_Z_L * tau[3,3] - Delta_L_f(site.pos[0]) * tau[2,2]
        def onsite_sm_L(site):#,V,U,salt):
            return (2.0 * p.t_sm - p.mu_sm) * tau[3,0] + p.V_Z_N * tau[3,3] - Delta_L_f(site.pos[0]) * tau[2,2]
        def onsite_sc_L_lead(site):
            return (2.0 * p.t_sc - p.mu_sc) * tau[3,0] + p.V_Z_L * tau[3,3] - p.Delta_L * tau[2,2]
        def onsite_sm_L_lead(site):
            V,U,salt = 0.,0.,bytes(1)
            return (2.0 * p.t_sm - p.mu_sm - V) * tau[3,0] + p.V_Z_N * tau[3,3]
        sys_L[(lat(i, 0) for i in range(p.L_NS//2))] = onsite_sc_L
        sys_L[(lat(i, 0) for i in range(p.L_NS//2, p.L_NS))] = onsite_sm_L
        sys_L[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -p.t_sc * tau[3,0] - 1j * E_soc_sm * tau[3,2]

        left_lead_L = kwant.Builder(sym_l)
        left_lead_L[lat(0, 0)] = onsite_sc_L_lead
        left_lead_L[kwant.builder.HoppingKind((1, 0), lat, lat)] = -p.t_sc * tau[3,0] - 1j * E_soc_scL * tau[3,2]

        right_lead_L = kwant.Builder(sym_r, conservation_law=cons_law)
        right_lead_L[lat(0, 0)] = onsite_sm_L_lead
        right_lead_L[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -p.t_sm * tau[3,0] - 1j * E_soc_sm * tau[3,2]

        sys_L.attach_lead(left_lead_L)
        sys_L.attach_lead(right_lead_L)
        #make middle system
        sys_N = kwant.Builder()
        def onsite_sm_N(site):#,V,U,salt):
            #V,U,salt = 0.,0.,bytes(1)
            #phi = V * (1-site.pos[0]/((L_sm-1)*a))
            V =  p.U0 if site.pos[0]==0 or site.pos[0]==(p.L_sm-1)*p.a else -p.well
            return (2.0 * p.t_sm - p.mu_sm + V) * tau[3,0] + p.V_Z_N * tau[3,3]
        def onsite_sm_N_l(site):#,V,U,salt):
            #V,U,salt = 0.,0.,bytes(1)
            return (2.0 * p.t_sm - p.mu_sm) * tau[3,0] + p.V_Z_N * tau[3,3]
        onsite_sm_N_r = (2.0 * p.t_sm - p.mu_sm) * tau[3,0] + p.V_Z_N * tau[3,3]

        for i in range(p.L_sm):
            sys_N[lat(i,0)] = onsite_sm_N
        sys_N[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -p.t_sm * tau[3,0] - 1j * E_soc_sm * tau[3,2]

        left_lead_N = kwant.Builder(sym_l, conservation_law=cons_law)
        left_lead_N[lat(0, 0)] = onsite_sm_N_l
        left_lead_N[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -p.t_sm * tau[3,0] - 1j * E_soc_sm * tau[3,2]

        right_lead_N = kwant.Builder(sym_r, conservation_law=cons_law)
        right_lead_N[lat(0, 0)] = onsite_sm_N_r
        right_lead_N[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -p.t_sm * tau[3,0] - 1j * E_soc_sm * tau[3,2]

        sys_N.attach_lead(left_lead_N)
        sys_N.attach_lead(right_lead_N)

        #make R system
        sys_R = kwant.Builder()
        def Delta_R_f(x):
            return p.Delta_R*(numpy.tanh(p.beta*(x-p.L_NS//2+0.5)) + 1)/2
        def onsite_sc_R(site):
            return (2.0 * p.t_sc - p.mu_sc) * tau[3,0] + p.V_Z_R * tau[3,3] - Delta_R_f(site.pos[0]) * tau[2,2]
        def onsite_sm_R(site):
            return (2.0 * p.t_sm - p.mu_sm) * tau[3,0] + p.V_Z_N * tau[3,3] - Delta_R_f(site.pos[0]) * tau[2,2]
        def onsite_sc_R_lead(site):
            return (2.0 * p.t_sc - p.mu_sc) * tau[3,0] + p.V_Z_R * tau[3,3] - p.Delta_R * tau[2,2]
        def onsite_sm_R_lead(site):
            return (2.0 * p.t_sm - p.mu_sm) * tau[3,0] + p.V_Z_N * tau[3,3]

        sys_R[(lat(i, 0) for i in range(p.L_NS//2))] = onsite_sm_R
        sys_R[(lat(i, 0) for i in range(p.L_NS//2, p.L_NS))] = onsite_sc_R
        sys_R[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -(p.t_sc + p.t_sm) / 2 * tau[3,0] - 1j * (E_soc_scR+E_soc_sm)/2 * tau[3,2]  #edit

        left_lead_R = kwant.Builder(sym_l, conservation_law=cons_law)
        left_lead_R[lat(0, 0)] = onsite_sm_R_lead
        left_lead_R[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -p.t_sm * tau[3,0] - 1j * E_soc_sm * tau[3,2]

        right_lead_R = kwant.Builder(sym_r)
        right_lead_R[lat(0, 0)] = onsite_sc_R_lead
        right_lead_R[kwant.builder.HoppingKind((1, 0), lat, lat)]         = -p.t_sc * tau[3,0] - 1j * E_soc_scR * tau[3,2]

        sys_R.attach_lead(left_lead_R)
        sys_R.attach_lead(right_lead_R)

        self.systs = [sys_L.finalized(), sys_R.finalized(), sys_N.finalized()]
        return sys_L.finalized(), sys_R.finalized(), sys_N.finalized()

    def find_min_gap(self, *leads,params={},test=False):
        '''can have some issues, double check result'''
        mins = []
        for lead in leads:
            bands = kwant.physics.Bands(lead, params=params)
            energy = lambda k: min([abs(ep) for ep in bands(k)])
            k_min = scipy.optimize.brute(energy, [(-1e-2, np.pi-1e-2)], Ns = 6000)[0] #spectra are symetrical
            mins.append(energy(k_min))
        return min(mins)

    def plot_bands(self, lead_syst,params={},fname=None,momenta=np.linspace(-np.pi, np.pi, 6001),ylim=None,lines=[],ax=None):
        if fname: print(fname)
        bands    = kwant.physics.Bands(lead_syst,params=params)
        O = np.diag([0,0,1,1])
        energies = []
        weights = []
        for k in momenta:
            eigs, evecs = bands(k,return_eigenvectors=True)
            energies.append(eigs)
            if evecs.shape[0]==O.shape[0]:
                weights.append((dagger(evecs) @ O @ evecs).diagonal() / (dagger(evecs) @ np.identity(O.shape[0]) @ evecs).diagonal())
            else:
                weights.append(np.ones_like(eigs))


        energies = np.array(energies)
        weights = np.real(np.array(weights))

        if not ax:
            fig, ax = plt.subplots()
        for line in lines:
            if isinstance(line,tuple):
                ax.axhline(line[0],c=line[1])
            else:
                ax.axhline(line)
        for i in range(energies.shape[1]):
            weighted_plot(momenta, energies[:,i],weights[:,i],ax=ax)
        ax.set_xlabel("momentum [(lattice constant)^-1]")
        ax.set_ylabel("energy [meV]")
        if ylim: ax.set_ylim(ylim)

    def plot_geometry(self):
        geo_fig, ((axL, axN, axR)) = plt.subplots(1, 3, sharey=True,figsize=(18,2))
        syst_L, syst_R, syst_N = self.systs
        axL.set_title('Left system')
        axN.set_title('Normal system')
        axR.set_title('Right system')
        kwant.plot(syst_L, ax=axL)
        kwant.plot(syst_N, ax=axN)
        kwant.plot(syst_R, ax=axR)

    def plot_spectra(self, xlim, ylim):
        syst_L, syst_R, syst_N = self.systs
        gap_L = self.find_min_gap(syst_L.leads[0])
        gap_R = self.find_min_gap(syst_R.leads[1])
        min_gap = min(gap_L, gap_R)

        spec_fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(18,10), sharex=True, sharey=True)

        leads = [*syst_L.leads, *syst_N.leads, *syst_R.leads]
        axes  = [ax1, ax4, ax2, ax5, ax3, ax6]

        for lead,ax in zip(leads,axes):
            self.plot_bands(lead,momenta=np.linspace(-xlim,xlim, 6001),ylim=(-ylim,ylim),lines=[(0,'w'),(self.sysinfo.E_scale,'b'),(self.sysinfo.band_min,'r')],ax=ax)

        spec_fig.text(0.07, 0.33, "right lead", color='white',rotation='vertical',size='x-large')
        spec_fig.text(0.07, 0.72, "left lead", color='white',rotation='vertical',size='x-large')

    def plot_potential(self):
        syst_L, syst_R, syst_N = self.systs
        t_sm = self.params.t_sm
        potential_up = -2*t_sm + np.abs(syst_N.hamiltonian_submatrix().diagonal()[::4])
        potential_do = -2*t_sm + np.abs(syst_N.hamiltonian_submatrix().diagonal()[1::4])
        potential_L = -2*t_sm + np.abs(syst_N.leads[0].hamiltonian_submatrix().diagonal()[:1])
        potential_R = -2*t_sm + np.abs(syst_N.leads[1].hamiltonian_submatrix().diagonal()[:1])
        spec_fig, ax = plt.subplots(1, 1)
        buffer = int(potential_up.size * 0.1)
        ax.plot(np.arange(-buffer, potential_up.size+buffer), np.concatenate([np.repeat(potential_L,buffer), potential_up, np.repeat(potential_R,buffer)]), label=r'U$\uparrow$');
        ax.plot(np.arange(-buffer, potential_up.size+buffer), np.concatenate([np.repeat(potential_L,buffer), potential_do, np.repeat(potential_R,buffer)]), label=r'U$\downarrow$');
        ax.set_title("Potential Plot")
        ax.set_ylabel("Potential (meV)")
        ax.set_xlim((-buffer,potential_up.size+buffer))
        ax.legend()

    def plot_sm_conductance(self, Es, ax=None, spin=True, label=r"T"):
        if ax is None:
            f, ax = plt.subplots(1, 1)
        T_up = np.empty_like(Es)
        if spin: T_down = np.empty_like(Es)

        for i,E in enumerate(Es):
            smat = kwant.smatrix(self.systs[N],E)
            T_up[i] = smat.transmission((0,0),(1,0))
            if spin: T_down[i] = smat.transmission((0,1),(1,1))

        if spin:
            ax.plot(Es/self.sysinfo.E_scale, T_up,':', label=label+r" $\uparrow$")
            ax.plot(Es/self.sysinfo.E_scale, T_down,':', label=label+r" $\downarrow$")
        else:
            ax.plot(Es/self.sysinfo.E_scale, T_up,'-', label=label)
        ax.set_title("Transmission Probabilities")
        ax.set_xlabel(r"$E$ ($\Delta$)")
        ax.set_ylabel(r"$|t|^2$")
        ax.set_ylim((-0.1,1.1))
        ax.legend()

    def generate_Z_estimate(self):
        '''calculate approximate Z to determine necessary precision when calculating current'''
        Es = np.linspace(-self.sysinfo.E_scale, self.sysinfo.E_scale, 10)
        T = np.empty_like(Es)
        for i,E in enumerate(Es):
            smat = kwant.smatrix(self.systs[N],E)
            T[i] = smat.transmission((0,0),(1,0))
        T_avg = np.round(np.average(T),5) #round to prevent numerical errors
        if T_avg==0.:
            self.sysinfo.Z = 999.
        else:
            self.sysinfo.Z =  np.sqrt(1/T_avg-1)

    def interpolate(self, client, res=100):
        #self.sysinfo.E_scale = 1.0*kB #TODO: but why though
        dview = client[:]
        interp_bounds = self.sysinfo.get_interpolation_bounds(self.sysinfo.band_min, self.maxV)
        dview.push({'tau' : tau})

        with dview.sync_imports(quiet=True):
            import numpy
        self.sysinfo.interpolate('LNR',
                            interp_bounds,
                            params = {},
                            resolution = res,
                            executor = client,
                            progress_bar = True
                        )

    def _lview_func(self, V, threads=1, phase=0., tol=0.01):
        os.environ['OMP_NUM_THREADS'] = str(threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
        os.environ['MKL_NUM_THREADS'] = str(threads)

        import kwant_mar
        import numpy as np

        cur = kwant_mar.calculate_current(
                                        V,
                                        self.sysinfo,
                                        {},
                                        self.sysinfo.band_min,
                                        self.sysinfo.E_scale,
                                        tol = tol
                                         )
        return cur




def interpolate(solvers, client, goal=0.01):
    dview = client[:]
    dview.push({'tau' : tau})

    learners = []
    for s in solvers:
        si = s.sysinfo
        si.interpolators = [None, None, None]
        bounds = si.get_interpolation_bounds(si.band_min, s.maxV)
        si_learners = si.get_learners('LNR', bounds, client[:])
        learners.extend(si_learners)
    runner = adaptive.Runner(adaptive.BalancingLearner(learners), retries=2, goal=lambda l : l.loss() < goal, executor=client)
    return runner

def update_interpolators(solvers,runner):
    learners = runner.learner.learners
    if len(solvers)*3!=len(learners): raise Exception('runner should match solver')
    for i, s in enumerate(solvers):
        s.sysinfo.update_interpolators('LNR', learners[i*3:(i+1)*3])