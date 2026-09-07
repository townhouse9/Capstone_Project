"""
Black-Box Optimization (BBO) Weekly Pipeline & Template
Imperial College Capstone Project

This script provides an automated, modular workflow for:
1. Ingesting function datasets (.npy files).
2. Comparing Bayesian Optimization (GP) against ML Regression models (Random Forest, Extra Trees, Polynomial Ridge, Gradient Boosting, Neural Network MLP).
3. Computing acquisition functions (EI, UCB) over dense grids (2D) or Latin Hypercube/Monte Carlo candidate samples (3D-8D).
4. Formatting queries according to project brief guidelines (0.xxxxxx-0.yyyyyy-...).
5. Generating comprehensive visualisations for tracking weekly progress across all functions and multi-week trajectories.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Ensure output directory exists
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)


def format_query_string(x):
    """
    Formats a 1D numpy array of coordinates into the required brief format:
    x1-x2-x3-...-xn with 6 decimal places starting with '0.'.
    Example: 0.123456-0.654321
    """
    formatted_coords = [f"{val:.6f}" for val in x]
    return "-".join(formatted_coords)


def load_function_data(func_id, base_dir="."):
    """Loads inputs and outputs numpy arrays for a given function folder."""
    fdir = os.path.join(base_dir, f"function_{func_id}")
    inputs_path = os.path.join(fdir, "initial_inputs.npy")
    outputs_path = os.path.join(fdir, "initial_outputs.npy")

    if not os.path.exists(inputs_path) or not os.path.exists(outputs_path):
        raise FileNotFoundError(f"Missing data files in {fdir}")

    X = np.load(inputs_path)
    y = np.load(outputs_path)
    return X, y


def define_candidate_models():
    """Returns a dictionary of candidate surrogate models to evaluate."""
    return {
        'GP (Matern 2.5)': GaussianProcessRegressor(
            kernel=C(1.0) * Matern(length_scale=0.2, length_scale_bounds=(1e-2, 1e2), nu=2.5),
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        ),
        'GP (Matern 1.5)': GaussianProcessRegressor(
            kernel=C(1.0) * Matern(length_scale=0.2, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        ),
        'GP (RBF)': GaussianProcessRegressor(
            kernel=C(1.0) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1e2)),
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'Polynomial Ridge': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ]),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=42
        ),
        'Neural Net (MLP)': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=600, alpha=1e-3, random_state=42))
        ])
    }


def evaluate_models_cv(X, y, n_splits=5):
    """
    Evaluates all candidate surrogate models using K-Fold Cross-Validation.
    Returns a DataFrame sorted by Mean Squared Error (lower is better).
    """
    models = define_candidate_models()
    n_samples = len(X)
    actual_splits = min(n_splits, n_samples)
    kf = KFold(n_splits=actual_splits, shuffle=True, random_state=42)

    results = []
    for name, model in models.items():
        mse_list = []
        r2_list = []
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            try:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                mse_list.append(mean_squared_error(y_te, preds))
                if np.var(y_te) > 1e-12:
                    r2_list.append(r2_score(y_te, preds))
                else:
                    r2_list.append(0.0)
            except Exception as e:
                mse_list.append(np.nan)
                r2_list.append(np.nan)

        results.append({
            'Model': name,
            'CV_MSE_Mean': np.nanmean(mse_list),
            'CV_MSE_Std': np.nanstd(mse_list),
            'CV_R2_Mean': np.nanmean(r2_list)
        })

    df = pd.DataFrame(results).sort_values('CV_MSE_Mean').reset_index(drop=True)
    return df


def generate_candidate_points(dim, num_samples=30000, res_2d=200):
    """
    Generates candidate search points in [0, 1]^dim.
    Uses dense grid for 2D, and uniform random candidate grid for >=3D.
    """
    if dim == 2:
        x_lin = np.linspace(0.0, 1.0, res_2d)
        X1, X2 = np.meshgrid(x_lin, x_lin)
        candidates = np.vstack([X1.ravel(), X2.ravel()]).T
        grid_shape = (res_2d, res_2d)
        return candidates, grid_shape
    else:
        rng = np.random.RandomState(42)
        candidates = rng.uniform(0.0, 1.0, size=(num_samples, dim))
        return candidates, None


def compute_acquisition(gp_model, candidates, current_best_y, y_range, xi_frac=0.01, beta=2.576):
    """
    Computes Expected Improvement (EI) and Upper Confidence Bound (UCB) for GP model predictions.
    Scales xi relative to the dynamic output range y_range.
    """
    mean, std = gp_model.predict(candidates, return_std=True)
    xi = xi_frac * y_range

    with np.errstate(divide='ignore', invalid='ignore'):
        imp = mean - current_best_y - xi
        Z = imp / (std + 1e-12)
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std <= 1e-12] = 0.0

    ucb = mean + beta * std
    return mean, std, ei, ucb


def process_function(func_id, week_num=7):
    """
    Executes full modeling, acquisition, formatting, and visualization pipeline for one function.
    """
    X, y = load_function_data(func_id)
    dim = X.shape[1]
    n_samples = len(X)
    current_best_idx = np.argmax(y)
    current_best_y = y[current_best_idx]
    current_best_x = X[current_best_idx]
    y_range = max(np.max(y) - np.min(y), 1e-6)

    print(f"\n==================== FUNCTION {func_id} (Dimension: {dim}D, Samples: {n_samples}) ====================")
    print(f"Current Max y: {current_best_y:.6f} at X: {np.round(current_best_x, 6).tolist()}")

    # 1. Model Evaluation via CV
    cv_df = evaluate_models_cv(X, y)
    print("\n--- Cross-Validation Model Comparison ---")
    print(cv_df.to_string(index=False))

    # 2. Primary GP Model fitting for BO
    gp_primary = GaussianProcessRegressor(
        kernel=C(1.0) * Matern(length_scale=0.2, length_scale_bounds=(1e-2, 1e2), nu=2.5),
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=15,
        random_state=42
    )
    gp_primary.fit(X, y)

    # Neural Network / RF surrogates for comparison and gradient/importance insight
    rf_primary = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    rf_primary.fit(X, y)

    # 3. Candidate search & Acquisition
    candidates, grid_shape = generate_candidate_points(dim)
    mean, std, ei, ucb = compute_acquisition(gp_primary, candidates, current_best_y, y_range)

    if np.max(ei) > 1e-12:
        best_cand_idx = np.argmax(ei)
        acq_name = "Expected Improvement (EI)"
    else:
        best_cand_idx = np.argmax(ucb)
        acq_name = "Upper Confidence Bound (UCB)"

    next_query_x = candidates[best_cand_idx]
    next_query_pred_mean = mean[best_cand_idx]
    next_query_pred_std = std[best_cand_idx]
    query_str = format_query_string(next_query_x)

    print(f"\n--- Proposed Query Selection (Week {week_num}) ---")
    print(f"Acquisition Method: {acq_name}")
    print(f"Suggested Next Query X: {np.round(next_query_x, 6).tolist()}")
    print(f"Predicted Output y (GP mean): {next_query_pred_mean:.6f} +/- {next_query_pred_std:.6f}")
    print(f"Formatted Submission String: {query_str}")

    # 4. Generate Visualizations
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Function {func_id} ({dim}D) - Week {week_num} Model Diagnostics & Query Selection", fontsize=16, fontweight='bold')

    if dim == 2:
        res = grid_shape[0]

        # Panel 1: CV Model Comparison
        ax1 = fig.add_subplot(2, 2, 1)
        models_names = cv_df['Model']
        mses = cv_df['CV_MSE_Mean']
        ax1.barh(models_names, mses, color='skyblue', edgecolor='black')
        ax1.set_xlabel("CV Mean Squared Error (Lower is Better)")
        ax1.set_title("1. Model Cross-Validation Performance")
        ax1.invert_yaxis()

        # Panel 2: Predicted Mean Surface & Query Point
        ax2 = fig.add_subplot(2, 2, 2)
        X1 = candidates[:, 0].reshape(res, res)
        X2 = candidates[:, 1].reshape(res, res)
        Z_mean = mean.reshape(res, res)
        c2 = ax2.contourf(X1, X2, Z_mean, levels=30, cmap='viridis')
        fig.colorbar(c2, ax=ax2, label='Predicted Output (y)')
        ax2.scatter(X[:, 0], X[:, 1], color='red', marker='o', s=60, edgecolors='black', label=f'Explored Points (n={n_samples})')
        ax2.scatter(current_best_x[0], current_best_x[1], color='gold', marker='*', s=250, edgecolors='black', label=f'Current Max ({current_best_y:.3f})')
        ax2.scatter(next_query_x[0], next_query_x[1], color='magenta', marker='X', s=250, edgecolors='black', label=f'Next Query (W{week_num})')
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        ax2.set_title("2. GP Mean Landscape & Query Location")
        ax2.legend(loc='upper left', fontsize=8)

        # Panel 3: Acquisition Function Contour (EI)
        ax3 = fig.add_subplot(2, 2, 3)
        Z_ei = ei.reshape(res, res)
        c3 = ax3.contourf(X1, X2, Z_ei, levels=30, cmap='magma')
        fig.colorbar(c3, ax=ax3, label='EI Acquisition Value')
        ax3.scatter(next_query_x[0], next_query_x[1], color='magenta', marker='X', s=250, edgecolors='black', label=f'Next Query (W{week_num})')
        ax3.set_xlabel("Dimension 1")
        ax3.set_ylabel("Dimension 2")
        ax3.set_title("3. Expected Improvement (EI) Acquisition Map")
        ax3.legend(loc='upper left', fontsize=8)

        # Panel 4: GP Model Uncertainty (Std Dev) Contour Map
        ax4 = fig.add_subplot(2, 2, 4)
        Z_std = std.reshape(res, res)
        c4 = ax4.contourf(X1, X2, Z_std, levels=30, cmap='plasma')
        fig.colorbar(c4, ax=ax4, label='Uncertainty (Std Dev)')
        ax4.scatter(X[:, 0], X[:, 1], color='red', marker='o', s=50, edgecolors='black')
        ax4.set_xlabel("Dimension 1")
        ax4.set_ylabel("Dimension 2")
        ax4.set_title("4. GP Model Uncertainty (Epistemic Variance)")

    else:
        # High dimensional layout (3D to 8D)
        # Panel 1: CV Model Comparison
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.barh(cv_df['Model'], cv_df['CV_MSE_Mean'], color='teal', edgecolor='black')
        ax1.set_xlabel("CV Mean Squared Error")
        ax1.set_title("1. Model Cross-Validation Performance")
        ax1.invert_yaxis()

        # Panel 2: 1D Profile Slices through Current Best Point
        ax2 = fig.add_subplot(2, 2, 2)
        n_points_slice = 100
        t_vals = np.linspace(0, 1, n_points_slice)
        colors = plt.cm.tab10(np.linspace(0, 1, dim))

        for d in range(dim):
            slice_coords = np.tile(current_best_x, (n_points_slice, 1))
            slice_coords[:, d] = t_vals
            pred_m, pred_s = gp_primary.predict(slice_coords, return_std=True)
            ax2.plot(t_vals, pred_m, label=f"Dim {d+1}", color=colors[d], linewidth=1.8)

        ax2.axvline(current_best_x[0], color='gray', linestyle='--', alpha=0.5, label='Current Best Location')
        ax2.set_xlabel("Coordinate Value [0, 1]")
        ax2.set_ylabel("Predicted Output y")
        ax2.set_title(f"2. 1D Feature Sensitivity Slices ({dim}D)")
        ax2.legend(loc='upper right', fontsize=8, ncol=2)

        # Panel 3: RF Feature Importances (Surrogate Structure Insight)
        ax3 = fig.add_subplot(2, 2, 3)
        rf_imp = rf_primary.feature_importances_
        dims_labels = [f"Dim {d+1}" for d in range(dim)]
        ax3.bar(dims_labels, rf_imp, color='darkorange', edgecolor='black')
        ax3.set_ylabel("Gini Feature Importance")
        ax3.set_title("3. Feature Importance Profile")

        # Panel 4: Candidate Acquisition Distribution & Top Query Selection
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(ei, bins=50, color='purple', alpha=0.7, edgecolor='black', log=True)
        ax4.axvline(np.max(ei), color='magenta', linestyle='--', linewidth=2, label=f'Selected Query (EI={np.max(ei):.4e})')
        ax4.set_xlabel("Expected Improvement (EI) Score")
        ax4.set_ylabel("Candidate Count (Log Scale)")
        ax4.set_title("4. Acquisition Value Distribution over Candidate Space")
        ax4.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    vis_path = os.path.join(VIS_DIR, f"function_{func_id}_week{week_num}.png")
    plt.savefig(vis_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {vis_path}")

    return {
        'func_id': func_id,
        'dim': dim,
        'n_samples': n_samples,
        'current_best_y': float(current_best_y),
        'current_best_x': current_best_x.tolist(),
        'best_model': cv_df.iloc[0]['Model'],
        'best_model_cv_mse': float(cv_df.iloc[0]['CV_MSE_Mean']),
        'next_query_x': next_query_x.tolist(),
        'predicted_y_mean': float(next_query_pred_mean),
        'predicted_y_std': float(next_query_pred_std),
        'submission_string': query_str,
        'cv_results': cv_df.to_dict(orient='records')
    }


def generate_weekly_summary_dashboard(summary_results, current_week_label="Week 7 (Module 18)"):
    """
    Generates and updates a master 2x2 dashboard figure tracking multi-week optimization progress.
    """
    history_path = "weekly_progress_history.json"
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {}

    history[current_week_label] = {}
    for entry in summary_results:
        fid = f"Func {entry['func_id']}"
        history[current_week_label][fid] = {
            'best_y': entry['current_best_y'],
            'pred_mean': entry['predicted_y_mean'],
            'pred_std': entry['predicted_y_std'],
            'best_model': entry['best_model'],
            'cv_mse': entry['best_model_cv_mse']
        }

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Imperial Capstone Black-Box Optimization - Weekly Master Dashboard", fontsize=18, fontweight='bold', y=0.98)

    # Subplot 1: Current Max vs Predicted Next Peak
    ax1 = fig.add_subplot(2, 2, 1)
    func_ids = [f"Func {d['func_id']}\n({d['dim']}D)" for d in summary_results]
    current_bests = [d['current_best_y'] for d in summary_results]
    pred_means = [d['predicted_y_mean'] for d in summary_results]
    pred_stds = [d['predicted_y_std'] for d in summary_results]

    x_indices = np.arange(len(summary_results))
    width = 0.35

    ax1.bar(x_indices - width/2, current_bests, width, label='Current Best y', color='navy', alpha=0.85, edgecolor='black')
    ax1.bar(x_indices + width/2, pred_means, width, yerr=pred_stds, label='Predicted Next y (GP Mean ± Std)', color='crimson', alpha=0.75, edgecolor='black', capsize=4)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(func_ids, fontsize=9)
    ax1.set_ylabel("Output Value y")
    ax1.set_title("1. Current Maximum vs Predicted Next Query Output", fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_yscale('symlog', linthresh=0.01)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Multi-Week Trajectory Tracking
    ax2 = fig.add_subplot(2, 2, 2)
    weeks = list(history.keys())
    func_labels = [f"Func {i}" for i in range(1, 9)]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for idx, fid in enumerate(func_labels):
        y_vals = []
        for w in weeks:
            if fid in history[w]:
                y_vals.append(history[w][fid]['best_y'])
            else:
                y_vals.append(np.nan)
        ax2.plot(weeks, y_vals, marker='o', linewidth=2.2, label=fid, color=colors[idx])

    ax2.set_ylabel("Max Output Found y")
    ax2.set_title("2. Weekly Optimization Trajectory (Max Found per Round)", fontweight='bold')
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax2.set_yscale('symlog', linthresh=0.01)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Winning Surrogate Model & CV MSE
    ax3 = fig.add_subplot(2, 2, 3)
    best_models = [d['best_model'] for d in summary_results]
    cv_mses = [d['best_model_cv_mse'] for d in summary_results]

    bars = ax3.bar(func_ids, cv_mses, color='teal', edgecolor='black', alpha=0.85)
    ax3.set_yscale('log')
    ax3.set_ylabel("5-Fold CV Mean Squared Error (Log Scale)")
    ax3.set_title("3. Surrogate Model CV Error & Winning Model Type", fontweight='bold')

    for bar, m_name in zip(bars, best_models):
        height = bar.get_height()
        ax3.annotate(f"{m_name}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, rotation=25)

    ax3.grid(True, linestyle='--', alpha=0.5, which='both')

    # Subplot 4: Relative Expected Improvement Potential
    ax4 = fig.add_subplot(2, 2, 4)
    rel_gains = []
    for d in summary_results:
        cb = d['current_best_y']
        pm = d['predicted_y_mean']
        gain = (pm - cb) / (abs(cb) + 1.0)
        rel_gains.append(max(gain, 0.0))

    ax4.bar(func_ids, rel_gains, color='darkorange', edgecolor='black', alpha=0.85)
    ax4.set_ylabel("Normalized Expected Gain Ratio")
    ax4.set_title("4. Expected Improvement (EI) Potential per Function", fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    summary_fig_path = os.path.join(VIS_DIR, "weekly_progress_summary.png")
    plt.savefig(summary_fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nMaster summary dashboard visualization updated at: {summary_fig_path}")


def main():
    CURRENT_WEEK = 7
    CURRENT_WEEK_LABEL = f"Week {CURRENT_WEEK} (Module 18)"
    summary_results = []
    for func_id in range(1, 9):
        res = process_function(func_id, week_num=CURRENT_WEEK)
        summary_results.append(res)

    json_path = f"week{CURRENT_WEEK}_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2)

    # Generate master weekly summary dashboard figure
    generate_weekly_summary_dashboard(summary_results, current_week_label=CURRENT_WEEK_LABEL)

    print("\n" + "=" * 80)
    print(f"                      WEEK {CURRENT_WEEK} PORTAL SUBMISSION SUMMARY                       ")
    print("=" * 80)
    print(f"{'Func':<6} | {'Dim':<4} | {'Current Best y':<16} | {'Submission String (x1-x2-...-xn)'}")
    print("-" * 80)
    for s in summary_results:
        print(f"Func {s['func_id']:<2} | {s['dim']:<4}D | {s['current_best_y']:<16.6f} | {s['submission_string']}")
    print("=" * 80)
    print(f"\nSubmission strings saved to {json_path}. Visualisations generated in visualizations/")


if __name__ == "__main__":
    main()
