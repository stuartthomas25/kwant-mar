from hashlib import md5
import dill
import os
import shutil

root = '.quickle'

# Old implementation, not pickleable
# 
# def quickle(f, dview):
#     with dview.sync_imports(quiet=True):
#         import dill

#     binary = dill.dumps(f)
#     key = md5(binary).hexdigest()

#     if not os.path.exists(root):
#         os.makedirs(root)

#     path = os.path.join(root, key + '.dat')

#     with open(path, 'wb') as f:
#         f.write(binary)
    
#     def _hf(*args, **kwargs):
#         func = dill.load(open(path,'rb'))
#         return func(*args, **kwargs)

#     return _hf

class quickle(object):
    def __init__(self,f,dview):
        with dview.sync_imports(quiet=True):
            import dill

        if not os.path.exists(root):
            os.makedirs(root)

        binary = dill.dumps(f)
        key = md5(binary).hexdigest()
        self.path = os.path.join(root, key)
        with open(self.path, 'wb') as f:
            f.write(binary)
        
    def __call__(self, *args, **kwargs):
        func = dill.load(open(self.path,'rb'))
        return func(*args, **kwargs)

def clear():
    shutil.rmtree(root) 

# def test_hash(dview):
#     def test():
#         try:
#             __hashlist__
#             return True
#         except:
#             return False
#     return all(dview.apply_sync(test))

# def add_func(dview, obj):
#     add_funcs(dview, [obj])

# def add_funcs(dview, objs):
#     def _ao(objs):
#         if not "__hashlist__" in globals():
#             global __hashlist__ 
#             __hashlist__ = {}
            
#         for obj in objs:
#             __hashlist__[ohash(obj)] = obj

#     _ao.__globals__['ohash'] = ohash
#     dview.apply_sync(_ao, objs)
    