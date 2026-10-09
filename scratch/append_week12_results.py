import numpy as np
import os

new_data_week12 = {
    1: {
        'x': [1.000000, 0.718593],
        'y': 3.611702647110725e-102
    },
    2: {
        'x': [0.668342, 0.919598],
        'y': 0.5020407850490992
    },
    3: {
        'x': [0.012599, 0.974551, 0.512869],
        'y': -0.0357233107235321
    },
    4: {
        'x': [0.384096, 0.428937, 0.398021, 0.337853],
        'y': 0.24332912992100164
    },
    5: {
        'x': [0.994502, 0.991348, 0.009326, 0.990912],
        'y': 4127.641223912825
    },
    6: {
        'x': [0.500492, 0.302059, 0.623995, 0.749921, 0.120259],
        'y': -0.20286530005072956
    },
    7: {
        'x': [0.109984, 0.307595, 0.307183, 0.330717, 0.317244, 0.697533],
        'y': 2.6619084931242267
    },
    8: {
        'x': [0.163699, 0.206374, 0.149274, 0.080089, 0.831871, 0.522949, 0.177955, 0.223963],
        'y': 9.9665579542646
    }
}

print("=== APPENDING WEEK 12 DATA (MODULE 23) ===")
for func_id, val in new_data_week12.items():
    fdir = f"function_{func_id}"
    in_path = os.path.join(fdir, "initial_inputs.npy")
    out_path = os.path.join(fdir, "initial_outputs.npy")
    
    X = np.load(in_path)
    y = np.load(out_path)
    
    new_x = np.array(val['x']).reshape(1, -1)
    new_y = np.array([val['y']])
    
    if len(X) > 0 and np.allclose(X[-1], new_x[0], atol=1e-5):
        print(f"Func {func_id}: Already appended. Shape: {X.shape}, {y.shape}")
    else:
        X_updated = np.vstack([X, new_x])
        y_updated = np.concatenate([y, new_y])
        np.save(in_path, X_updated)
        np.save(out_path, y_updated)
        print(f"Func {func_id}: Updated sample count: {len(X)} -> {len(X_updated)}. New point y={val['y']}")
