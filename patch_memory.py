import builtins

original_import = builtins.__import__

def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Execute the actual import subsystem
    module = original_import(name, globals, locals, fromlist, level)
    
    # Standardize empty or None elements to prevent non-iterable type errors
    has_target = (name == 'numpy') or (fromlist is not None and 'numpy' in fromlist)
    
    if has_target:
        try:
            import numpy as np
            
            if not hasattr(np, '_original_zeros'):
                np._original_zeros = np.zeros
                np._original_empty = np.empty
                
                # Intercept 3D array mappings specifically
                def safe_zeros(*args, **kwargs):
                    arr = np._original_zeros(*args, **kwargs)
                    if arr.ndim == 3:
                        return np.require(arr, dtype=np.float64, requirements=['C', 'W', 'A'])
                    return arr
                    
                def safe_empty(*args, **kwargs):
                    arr = np._original_empty(*args, **kwargs)
                    if arr.ndim == 3:
                        return np.require(arr, dtype=np.float64, requirements=['C', 'W', 'A'])
                    return arr
                
                # Rebind custom functions into the active runtime namespace
                np.zeros = safe_zeros
                np.empty = safe_empty
        except Exception:
            pass
            
    return module

builtins.__import__ = patched_import