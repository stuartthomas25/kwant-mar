from colorsys import hls_to_rgb
import timeit
import sys
import time
import ipywidgets
import numpy as np
from matplotlib import pyplot as plt
#create Timer class to estimate process time
class Timer:
    def __init__(self,length):
        self.length=int(length)
        self.__start=None
        self.i=0
    def elapsed_time(self):
        return timeit.default_timer()-self.__start
    def increment(self):
        clear_output(wait=True)
        if self.i==0: 
            self.__start=timeit.default_timer()
        print("Current progress: ",str(np.round((self.i/self.length)*100,2))+"%")
        print("Current runtime:  ",self.secsToString(self.elapsed_time()))
        if self.i<5:
            print("Expected runtime: ...")
        else:
            time_perc=timeit.default_timer()
            etime=(self.elapsed_time()/self.i)*self.length
            print("Expected runtime: ",self.secsToString(etime))

        self.i+=1
    def secsToString(self,secs):
        return "{}h {}m {}s".format(secs//3600,(secs%3600)//60,int(secs%60))
class LineTimer: 
    def __init__(self,name):
        self.name = str(name)
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print('"{0}" took {1:.4f} sec.'.format(self.name, self.end - self.start))
def print_nice(*objs,mod=True):
    if mod:
        [print(np.round(np.abs(obj),3)) for obj in objs]
    else: [(print('real'),print(np.round(np.real(obj),3)),print('imag'),print(np.round(np.imag(obj),3)),print()) for obj in objs]
     

class Muzzle:
    def __init__(self):
        super().__init__()
        self.__old_stdout = None
    def __enter__(self):
        self.__old_stdout = sys.stdout
        sys.stdout = None
    def __exit__(self,*args):
        sys.stdout = self.__old_stdout
        
        
class HashableNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __hash__(self):
        return hash(repr(self))


def complex_matshow(M,show=True,figsize=(6,6)):
    _M = np.asarray(M.T)
    r = np.abs(_M)
    arg = np.angle(_M) 
    
    
    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**1.0)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    plt.figure(figsize=figsize)
    im = plt.imshow(c)
    if show: plt.show()
    return im



def plot_Smat(S,figsize=(10,10)):
    complex_matshow(S,False,figsize=figsize)
    for i in range(4,S.shape[0],4):
        plt.axhline(i-0.5,c='grey',linewidth=0.3)
        plt.axvline(i-0.5,c='grey',linewidth=0.3)
    for i in range(16,S.shape[0],16):
        plt.axhline(i-0.5,c='w',linewidth=0.5)
        plt.axvline(i-0.5,c='w',linewidth=0.5)
    plt.show()
    
def plot_J(J):
    e_levels = J.size//16
    im = complex_matshow(np.flip(J.reshape((e_levels,16)), 0),False,figsize=(12,8))
    ax = im.axes
    ax.set_yticks([i for i in range(e_levels)])
    ax.set_xticks([i for i in range(16)])
    ax.axes.set_yticklabels([str(e_levels//2-i) for i in range(e_levels)])
    ax.axes.set_xticklabels([r'$e\uparrow$',r'$e\downarrow$',r'$h\uparrow$',r'$h\downarrow$']*4)
    ax.set_ylabel("$n$ (energy level)")

    for i in range(3):
        ax.axvline(3.5+4*i)
    plt.show()
    
def progress_bar(ar):
    w = ipywidgets.IntProgress()
    w.max = len(ar.msg_ids)
    display(w)
    while not ar.ready():
        w.value = ar.progress
        time.sleep(.1)
    w.close()
    print(f'completed in {ar.wall_time:.1f} secs')
    print(f'speedup: {ar.serial_time / ar.wall_time:.1f}')
    
def dagger(a): 
    return np.conj(a.T)
    
def plot_comp_func(xs,f,use_weighted_plot=False, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9,7)) 
        ax = fig.add_subplot(1,1,1)
    if callable(f):
        f_vec = np.vectorize(f)(xs)
    else:
        f_vec = f
    fig = plt.figure(figsize=(9,7))
    if use_weighted_plot:
        weighted_plot(xs,np.abs(f_vec)**2,np.angle(f_vec)%(2*np.pi)/(2*np.pi),ax=ax,cmap='hsv')
    else:
        ax.plot(xs, np.real(f_vec),'r-')
        ax.plot(xs, np.imag(f_vec),'b-')
        ax.plot(xs,  np.abs(f_vec)**2,'k-')
    #plt.ylim((-1.1,1.1))
    #plt.xlim((-5,5))
    ax.grid()

def finished_results(ar, contiguous=False):
    """return a list of the finished subset of results in an AsyncResult"""
    results = []
    for msg_id in ar.msg_ids:
        # grab an AsyncResult for each msg_id
        # omit owner=False on IPython < 3
        sub_r = rc.get_result(msg_id, owner=False)
        # if that sub-result is done, add it to results
        if sub_r.ready():
            r = sub_r.get()
            if isinstance(ar, parallel.AsyncMapResult):
                # unpack list of result in AMR partition
                results.extend(r)
            else:
                results.append(r)
        else:
            if contiguous:
                # if contiguous, stop on the first result that isn't ready
                # otherwise, keep going and see if any later results are ready,
                # in which case the partial result may be a little out of order.
                break
    return results

def weighted_plot(xs,fs,ws,ax=None,cmap='bwr'):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
    norm = plt.Normalize(0., 1.)
    #for i in range(0,eig_arr.shape[1]):
    
    ax.plot(xs,fs,linewidth=3.25,c='w',zorder = -10)
    
    points = np.array([xs, fs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(ws)
    lc.set_linewidth(2.25)
    line = ax.add_collection(lc)
    ax.patch.set_facecolor('black')
    
    return ax
    
##################


import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def profiling_func(func):
    def wrapper(V):
        from cProfile import Profile
        import time
        prof = Profile()
        start_time = time.time()
        init_time = time.strftime('%y%m%d-%H%M%S')
        ret = prof.runcall(func,V)
        prof.dump_stats(f"profiles/profile:t={time.time()-start_time:.2f}:V={V:.2f}:time={init_time}.prof") 
        return ret
    return wrapper