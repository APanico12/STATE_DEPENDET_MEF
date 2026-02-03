import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

class RobustCPD:
    """
    Robust Change Point Detector (Riani et al., 2019) with Polynomial Trends.
    Detects structural changes in trend (slope) robust to outliers and non-linear drifts.
    """
    
    def __init__(self, trimming=0.10, alpha=0.01):
        """
        Args:
            trimming (float): Fraction of data to trim (default 0.10 = 10%).
            alpha (float): Significance level for outlier detection (default 0.01 = 99% confidence).
        """
        self.trimming = trimming
        self.alpha = alpha
        self.results = None
        
    def fit(self, y_series, poly_order=0):
        """
        Main method to detect the structural break.
        
        Args:
            y_series: The time series data.
            poly_order (int): Order of the polynomial trend. 
                              0 = Linear (Standard Broken Trend)
                              2 = Quadratic (Controls for U-shape)
                              8 = Holladay (2017) style high-order trend
        """
        y = np.array(y_series)
        n = len(y)
        t = np.arange(n)
        
        # === NORMALIZATION ===
        # We normalize t to range [-1, 1] to ensure numerical stability 
        # when calculating high powers like t^8.
        t_norm = 2 * (t - t.min()) / (t.max() - t.min()) - 1
        
        # 1. Define Trimming Parameters
        h = int(n * (1 - self.trimming))
        
        # 2. Define Search Range
        start_tau = int(n * 0.15)
        end_tau = int(n * 0.85)
        
        # Variables to store the 'Best' result (minimized error)
        best_tau = -1
        min_trimmed_error = np.inf
        best_lts_params = None
        
        # Lists to store the sequence of statistics
        seq_taus = []
        seq_beta2_est = []   
        seq_beta2_tstat = [] 
        
        print(f"Scanning {end_tau - start_tau} candidate dates (Polynomial Order: {poly_order})...")
        
        # ================= STEP A: ROBUST GRID SEARCH =================
        for tau in range(start_tau, end_tau):
            
            # --- Construct Design Matrix ---
            slope_change = np.maximum(0, t - tau)
            
            # Base Columns: [Constant, Linear Time, Slope Change]
            # We preserve this order so Beta_2 is always at index 2
            columns = [np.ones(n), t, slope_change]
            
            # Add Polynomial Terms (if requested)
            # We append them at the end so they act as controls without shifting the index of slope_change
            if poly_order > 1:
                for p in range(2, poly_order + 1):
                    columns.append(t_norm ** p)
            
            X = np.column_stack(columns)
            
            # 1. Run FastLTS (Approximation via C-steps)
            beta_lts, trimmed_error, _ = self._run_lts_c_steps(X, y, h)
            
            # 2. Check if this is the best fit so far
            if trimmed_error < min_trimmed_error:
                min_trimmed_error = trimmed_error
                best_tau = tau
                best_lts_params = beta_lts
            
            # 3. Compute Inference for THIS candidate tau
            beta2_val, beta2_t = self._compute_reweighted_stats(X, y, beta_lts, trimmed_error, h)
            
            seq_taus.append(tau)
            seq_beta2_est.append(beta2_val)
            seq_beta2_tstat.append(beta2_t)

        # ================= STEP B: FINAL INFERENCE (BEST TAU) =================
        # Re-construct best model (need to rebuild X for the winner)
        slope_change_best = np.maximum(0, t - best_tau)
        best_columns = [np.ones(n), t, slope_change_best]
        if poly_order > 1:
            for p in range(2, poly_order + 1):
                best_columns.append(t_norm ** p)
        X_best = np.column_stack(best_columns)
        
        # Calculate Robust Scale & Outliers
        raw_scale = np.sqrt(min_trimmed_error / h)
        cons_factor = self._consistency_factor(h, n)
        robust_scale = raw_scale * cons_factor
        
        final_residuals = y - (X_best @ best_lts_params)
        is_outlier = np.abs(final_residuals / robust_scale) > stats.norm.ppf(1 - self.alpha / 2)
        
        # Final OLS on Clean Data
        weights = (~is_outlier).astype(int)
        X_clean = X_best[weights == 1]
        y_clean = y[weights == 1]
        
        final_model = sm.OLS(y_clean, X_clean).fit()
        
        # Store Results
        # Note: Index 2 is still Beta_2 (Slope Change) because we appended polynomials at the end
        self.results = {
            'break_index': best_tau,
            'robust_scale': robust_scale,
            'outliers_detected': np.sum(is_outlier),
            'beta2_val': final_model.params[2],
            'beta2_t_stat': final_model.tvalues[2],
            'beta2_p_val': final_model.pvalues[2],
            'model_summary': final_model.summary(),
            'sequence_taus': np.array(seq_taus),
            'sequence_beta2_est': np.array(seq_beta2_est),
            'sequence_beta2_tstats': np.array(seq_beta2_tstat)
        }
        
        return self.results

    def _compute_reweighted_stats(self, X, y, beta_lts, trimmed_error, h):
        """Helper to quickly calculate reweighted t-statistic."""
        n = len(y)
        raw_scale = np.sqrt(trimmed_error / h)
        cons_factor = self._consistency_factor(h, n)
        robust_scale = raw_scale * cons_factor
        
        if robust_scale < 1e-9: robust_scale = 1e-9
        residuals = y - (X @ beta_lts)
        is_outlier = np.abs(residuals / robust_scale) > 2.576
        weights = (~is_outlier).astype(int)
        
        try:
            if np.sum(weights) > X.shape[1] + 5: # Ensure enough degrees of freedom
                X_clean = X[weights == 1]
                y_clean = y[weights == 1]
                model = sm.OLS(y_clean, X_clean).fit()
                # Index 2 is consistently the Slope Change parameter
                return model.params[2], model.tvalues[2]
            else:
                return 0.0, 0.0
        except:
            return 0.0, 0.0

    def _run_lts_c_steps(self, X, y, h, n_iter=3):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        for _ in range(n_iter):
            residuals = (y - X @ beta)
            squared_resid = residuals**2
            best_h_indices = np.argsort(squared_resid)[:h]
            X_subset = X[best_h_indices]
            y_subset = y[best_h_indices]
            beta = np.linalg.lstsq(X_subset, y_subset, rcond=None)[0]
        
        final_residuals = (y - X @ beta)
        squared_resid = final_residuals**2
        squared_resid.sort()
        trimmed_error = np.sum(squared_resid[:h])
        return beta, trimmed_error, final_residuals

    def _consistency_factor(self, h, n):
        fraction = h / n
        q = stats.norm.ppf((1 + fraction) / 2)
        return 1 / q

    def summary(self):
        if self.results is None: return "Model not fitted."
        print("="*40)
        print(f"ROBUST STRUCTURAL BREAK DETECTED")
        print("="*40)
        print(f"Estimated Break Date     : {self.results['break_index']}")
        print(f"Slope Change (Beta2)     : {self.results['beta2_val']:.4f}")
        print(f"T-Statistic              : {self.results['beta2_t_stat']:.4f}")
        print("-" * 40)
        if abs(self.results['beta2_t_stat']) > 1.96:
            print(">> CONCLUSION: SIGNIFICANT BREAK DETECTED")
        else:
            print(">> CONCLUSION: NO SIGNIFICANT BREAK")

# ================= EXAMPLE USAGE =================
if __name__ == "__main__":
    # Synthetic Data with a "U-Shape" + Break + Outliers
    np.random.seed(42)
    n = 300
    t = np.arange(n)
    
    # 1. Quadratic Trend (The "U" shape)
    # y = 0.0005 * (t - 150)^2 
    prices = 0.0005 * (t - 150)**2 + 10
    
    # 2. Add Break at t=200
    prices[200:] += 1.0 * (t[200:] - 200)
    
    # 3. Add Noise & Outliers
    prices += np.random.normal(0, 0.5, n)
    prices[50:55] += 20.0 # Outliers

    detector = RobustCPD(trimming=0.25)
    
    # RUN 1: Standard Linear Model
    print("\n--- RUN 1: Linear Model ---")
    res_linear = detector.fit(prices, poly_order=0)
    detector.summary()
    
    # RUN 2: Quadratic Model (Controls for the U-shape)
    print("\n--- RUN 2: Quadratic Model ---")
    res_poly = detector.fit(prices, poly_order=2)
    detector.summary()
    
    # Plot Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(res_linear['sequence_taus'], np.abs(res_linear['sequence_beta2_tstats']), label='Linear Model', linestyle='--')
    plt.plot(res_poly['sequence_taus'], np.abs(res_poly['sequence_beta2_tstats']), label='Quadratic Model (Poly=2)', linewidth=2)
    plt.axhline(1.96, color='red', alpha=0.5)
    plt.title("Comparison: Does the Polynomial Kill the Break?")
    plt.legend()
    plt.show()