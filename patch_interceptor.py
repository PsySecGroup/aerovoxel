import sys
import importlib.abc
import importlib.util

class PybindInterceptor:
    def __init__(self, original_module):
        self.original_module = original_module

    def process_image_cpp(self, *args, **kwargs):
        print("\n=== INTERCEPTED PYBIND11 CALL PARAMETERS ===", flush=True)
        # Positional arguments mapping based on the pybind11 signature
        arg_names = [
            "image (shape/dtype)", "earth_position", "pointing_direction", "fov",
            "image_width", "image_height", "voxel_grid (shape)", "voxel_grid_extent",
            "max_distance", "num_steps", "celestial_sphere_texture", "center_ra_rad",
            "center_dec_rad", "angular_width_rad", "angular_height_rad",
            "update_celestial_sphere", "perform_background_subtraction"
        ]
        
        for idx, name in enumerate(arg_names):
            if idx < len(args):
                val = args[idx]
                # Format arrays cleanly to avoid text flood
                if hasattr(val, 'shape'):
                    print(f"  {name}: shape={val.shape}, dtype={val.dtype}", flush=True)
                else:
                    print(f"  {name}: {val}", flush=True)
                    
        # Forward execution to the actual C++ binary
        return self.original_module.process_image_cpp(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original_module, name)

class InterceptorMetaPathFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "process_image_cpp":
            # Remove this finder to prevent infinite recursion during real load
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Wrap module in our interceptor proxy
            sys.modules[fullname] = PybindInterceptor(module)
            return spec
        return None

# Inject finder into Python's module resolution pipeline
sys.meta_path.insert(0, InterceptorMetaPathFinder())

# Execute the original, unaltered script natively
importlib.import_module("spacevoxelviewer")