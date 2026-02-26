"""
US MEF (Marginal Emission Factor) Analysis with Markov Switching
Methodology: Dummy Variable Seasonality Extraction (Kapoor et al., 2023 adapted)
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm

import matplotlib as mpl
import warnings
from scipy.stats import chi2
from sympy import true


warnings.filterwarnings('ignore')

from statsmodels.formula.api import ols
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import io    
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

class LightMEF:
    """
    US Marginal Emission Factor Analysis with Markov Switching Regimes.
    Uses Dummy Variable Approach for Seasonality Extraction.
    """
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.data = None
        self.data_with_regimes = None
        self.ms_results = None
        self.high_regime = None
        self.low_regime = None
        
    # ==================== STYLE SETTINGS ====================
   
    def set_publication_style(self, use_latex=False):
        """
        Apply a publication-quality Matplotlib style.
        Optimized for figures intended for journals (Nature, Science, PNAS, etc.).

        Parameters
        ----------
        use_latex : bool, optional
            If True, enables LaTeX rendering for text and math (requires local LaTeX install).
        """

        # 1️⃣ BASE STYLE (clean & minimal)
        plt.style.use('seaborn-v0_8-white')

        # 2️⃣ FONT SETTINGS — Helvetica or Arial preferred by most journals
        font_params = {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 20,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'mathtext.fontset': 'stixsans',  # Sans-serif math look
        }

        # 3️⃣ OPTIONAL: LATEX RENDERING (for math-heavy papers)
        if use_latex:
            font_params.update({
                'text.usetex': True,
                'font.family': 'serif',
                'text.latex.preamble': (
                    r'\usepackage{amsmath} \usepackage{helvet} '
                    r'\usepackage{sansmath} \sansmath'
                )
            })

        # 4️⃣ AXES & GRID STYLE
        axes_params = {
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            'axes.grid': False,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }

        # 5️⃣ TICKS — fine-tuned for print readability
        tick_params = {
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
        }

        # 6️⃣ COLORBLIND-FRIENDLY PALETTE (Okabe–Ito)
        cb_palette = [
            '#E69F00', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7', '#000000'
        ]

        # 7️⃣ APPLY ALL STYLE SETTINGS
        mpl.rcParams.update(font_params)
        mpl.rcParams.update(axes_params)
        mpl.rcParams.update(tick_params)
        mpl.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'axes.prop_cycle': plt.cycler('color', cb_palette),
        })

    # ==================== DATA LOADING ====================
   
    def load_and_clean_data(self, start_date='2019-01-01', end_date='2023-01-01'):
        """Load and clean hourly dataset"""
        print("📂 Loading data...")
        
        df = pd.read_excel(self.data_path)
        df.columns = df.columns.str.strip()
        
        # Parse datetime
        df['bid_offer_date'] = pd.to_datetime(df['UTC time']).dt.date
        df['date_hour'] = pd.to_datetime(df['UTC time'])
        
        # Filter date range
        df = df[
            (df['bid_offer_date'] >= pd.to_datetime(start_date).date()) &
            (df['bid_offer_date'] <= pd.to_datetime(end_date).date())
        ].copy()
        
        # Create hourly interval (1-24)
        df['interval'] = df['date_hour'].dt.hour + 1
        
        # Rename columns
        rename_map = {
            'Demand': 'D',
            'Net generation': 'NG',
            'Hour': 'Local_Hour',
            'Sum (NG)': 'hourly_generation',
            'CO2 Emissions Generated': 'hourly_emissions',
            'CO2 Emissions Intensity for Generated Electricity': 'CO2EmissionsIntensity',
            'RNG': 'hourly_generation_renewables',
            'NRNG': 'hourly_generation_nonrenewables',
            'NG: COL': 'Gen_Coal',
            'NG: NG': 'Gen_Gas'
        }
        df = df.rename(columns=rename_map)
        
        # Convert units
        df['hourly_emissions_mlb'] = df['hourly_emissions'] * 2204.6 / 1_000_000
        df['hourly_generation_mkwh'] = df['hourly_generation'] * 1000 / 1_000_000
        df['hourly_generation_renewables_mkwh'] = df['hourly_generation_renewables'] * 1000 / 1_000_000
        df['hourly_generation_nonrenewables_mkwh'] = df['hourly_generation_nonrenewables'] * 1000 / 1_000_000
         # --- Convert units ---
        df['hourly_emissions_mlb'] = df['hourly_emissions'] * 2204.6 / 1_000_000  # metric tons → million lbs
        df['hourly_generation_mkwh'] = df['hourly_generation'] * 1000 / 1_000_000  # GWh → million kWh
        df['hourly_generation_renewables_mkwh'] = df['hourly_generation_renewables'] * 1000 / 1_000_000  # GWh → million kWh
        df['hourly_generation_nonrenewables_mkwh'] = df['hourly_generation_nonrenewables'] * 1000 / 1_000_000  # GWh → million kWh
        # --- Select final columns  ---
        keep_cols = [
            'bid_offer_date', 'interval', 'date_hour', 
            'D', 'NG','Local_Hour',
            'hourly_generation','hourly_generation_renewables','hourly_generation_nonrenewables', 'hourly_emissions',  
            'hourly_emissions_mlb', 'hourly_generation_mkwh','hourly_generation_renewables_mkwh','hourly_generation_nonrenewables_mkwh',
            'Gen_Coal', 'Gen_Gas'
        ]
        
        self.data = df[[col for col in keep_cols if col in df.columns]].copy()
        self.data = df
        self.data['T'] = np.arange(1, len(self.data) + 1)
        
        print(f"✅ Data loaded: {len(self.data)} observations from {start_date} to {end_date}")
        return self.data
    
    # ==================== TIME VARIABLES ====================
    
    def create_time_variables(self):
        """Create time dummies and factor variables"""
        print("🕐 Creating time variables (Season, Month, Day)...")
        
        df = self.data
        df['bid_offer_date'] = pd.to_datetime(df['bid_offer_date'], errors='coerce')
        
        # Extract components
        df['month'] = df['bid_offer_date'].dt.month
        df['year'] = df['bid_offer_date'].dt.year
        df['day'] = df['bid_offer_date'].dt.day_name()
        
        # Season Mapping
        season_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
            5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'
        }
        df['season'] = df['month'].map(season_map)
        
        # Trend
        df['trend'] = np.arange(1, len(df) + 1)
        
        self.data = df
        print("✅ Time variables created")
        return df
    
    # ==================== DUMMY VARIABLE DETRENDING ====================
    
    def dummy_variable_seasonality_extraction(self):
        """
        Removes seasonality using the Dummy Variable Approach (Kapoor et al., 2023).
        Formula: Y = Intercept + C(Day) + C(Month) + C(Season) + C(Hour) + Residuals
        """
        print("\n🔧 Extracting seasonality using Dummy Variables...")
        df = self.data.copy()
        
        # Define formula: Trend + Weekday + Month + Season + Hour
        # Note: 'season' and 'month' are collinear, OLS will handle this by dropping reference levels automatically
        formula_suffix = " ~  C(day) + C(year)*C(month) + C(interval)*C(month) + trend" 
        
        # 1. Emissions
        print("   -> Fitting OLS for Emissions...")
        model_em = ols(f'hourly_emissions_mlb {formula_suffix}', data=df).fit()
        print(model_em.summary())
        # Keep Intercept + Residuals (This is the de-seasonalized series)
        df['hourly_emissions_detrended'] = model_em.resid + model_em.params['Intercept']
        
        # 2. Non-Renewable Generation
        print("   -> Fitting OLS for Non-Renewables...")
        model_nr = ols(f'hourly_generation_nonrenewables_mkwh {formula_suffix}', data=df).fit()
        #print(model_nr.summary())
        df['hourly_generation_nonrenewables_detrended'] = model_nr.resid + model_nr.params['Intercept']
        
        # 3. Renewable Generation
        print("   -> Fitting OLS for Renewables...")
        model_r = ols(f'hourly_generation_renewables_mkwh {formula_suffix}', data=df).fit()
        df['hourly_generation_renewables_detrended'] = model_r.resid + model_r.params['Intercept']
        
        self.data = df
        print("✅ Dummy Variable Extraction completed")
        print(f"   R² Emissions Seasonality Model: {model_em.rsquared:.4f}")
        
        return df
    
    # ==================== MARKOV SWITCHING MODEL ====================
    
    def test_optimal_selection(self): 
        """
        Grid search for optimal Markov Switching Model parameters based on AIC/BIC.
        Iterates over:
          - k_regimes: [2, 3]
          - order: [1, 2, 3]
          - switching_variance: [True, False]
        """
        if 'hourly_emissions_detrended' not in self.data.columns:
            raise ValueError("❌ Run dummy_variable_seasonality_extraction() first!")
        
        print("\n🔄 Running Optimal Selection Grid Search...")
        subset = self.data.copy().reset_index(drop=True)
        
        # Scale variables
        scaler_y = StandardScaler()
        scaler_x = StandardScaler()
        
        y_scaled = scaler_y.fit_transform(subset[['hourly_emissions_detrended']])

        X_unscaled = subset[['hourly_generation_renewables_detrended', 'hourly_generation_nonrenewables_detrended']]
        X_scaled = scaler_x.fit_transform(X_unscaled)
        
        # Storage for results
        results_list = []
        
        # Grid Search Loops
        # k_regimes: 2 to 3
        # order: 1 to 3
        # switching_variance: True or False
        param_combinations = [
            (k, p, var) 
            for k in [2] 
            for p in [x for x in range(1,3)] 
            for var in [False]
        ]
        
        total_runs = len(param_combinations)
        
        for i, (k, p, var) in enumerate(param_combinations, 1):
            print(f"  > Fitting Model {i}/{total_runs}: k={k}, order={p}, switch_var={var}")
            
            try:
                ms_model = MarkovAutoregression(
                    endog=y_scaled,
                    exog=X_scaled,
                    k_regimes=k,
                    order=p,
                    trend='c',
                    switching_trend=True,
                    switching_exog=[True, True], # Assumes 2 exog columns
                    switching_variance=var
                )
            
                #ms_results = ms_model.fit()
                ms_results = ms_model.fit(
                    maxiter=1000,
                    em_iter=50,
                    search_reps=20,      
                    search_iter=10,
                    disp=False
            )   
                
                        
                
                results_list.append({
                    'k_regimes': k,
                    'order': p,
                    'switching_variance': var,
                    'Log_Likelihood': ms_results.llf,
                    'AIC': ms_results.aic,
                    'BIC': ms_results.bic,
                    'HQIC': ms_results.hqic if hasattr(ms_results, 'hqic') else np.nan,
                    'Converged': ms_results.mle_retvals['converged']
                })
            except Exception as e:
                print(f"Model failed for k={k}, order={p}, var={var}. Error: {e}")
                results_list.append({
                    'k_regimes': k,
                    'order': p,
                    'switching_variance': var,
                    'Log_Likelihood': None,
                    'AIC': None,
                    'BIC': None,
                    'HQIC': None,
                    'Converged': False
                })

        # Create Summary DataFrame
        summary_df = pd.DataFrame(results_list)
        
        # Sort by BIC (ascending) to highlight the best model
        summary_df.sort_values(by='BIC', ascending=True, inplace=True)
        
        print("\n✅ Grid Search Complete. Top 3 Models by BIC:")
        print(summary_df.head(3))
        
        return summary_df
    
    def freq_state(self, max_iter=1000, search_reps=20,k=3):
        if 'hourly_emissions_detrended' not in self.data.columns:
            raise ValueError("❌ Run dummy_variable_seasonality_extraction() first!")
        
        print("\n🔄 Fitting Markov Switching Model...")
        subset = self.data.copy().reset_index(drop=True)
        
        # Scale variables
        scaler_y = StandardScaler()
        scaler_x = StandardScaler()
        
        y_scaled = scaler_y.fit_transform(subset[['hourly_emissions_detrended']])

        X_unscaled = subset[['hourly_generation_renewables_detrended', 'hourly_generation_nonrenewables_detrended']]
        X_scaled = scaler_x.fit_transform(X_unscaled)
        
     
        ms_model = MarkovAutoregression(
            endog=y_scaled,
            exog=X_scaled,
            k_regimes=k,
            order=1,
            trend='c',
            switching_trend=True,
            switching_exog= [True, True],
            switching_variance=False
        )
        
        ms_results = ms_model.fit(maxiter=max_iter, search_reps=search_reps, disp=False)
        h_prob = ms_results.smoothed_marginal_probabilities
        # Convert to DataFrame for easier handling
        probs = pd.DataFrame(h_prob, columns=['Regime 0', 'Regime 1', 'Regime 2'])

        # Assign regimes (Highest probability wins)
        regime_assign = probs.idxmax(axis=1)

        # --- 2. DIAGNOSTICS (Your existing logic) ---
        print("--- DIAGNOSTICS ---\n")

        print("1. REGIME FREQUENCY:")
        freq = regime_assign.value_counts(normalize=True).sort_index() * 100
        for regime, pct in freq.items():
            print(f"   {regime}: {pct:.1f}%")

        if len(freq) < 3:
            print(f"   ⚠️ WARNING: Only {len(freq)} regimes were assigned. Some regimes have 0% frequency.")
            min_freq = 0.0
        else:
            min_freq = freq.min()
        status = "✅ PASS" if min_freq > 10 else "❌ FAIL (outlier regime detected)"
        print(f"   Min frequency: {min_freq:.1f}% {status}\n")

        print("2. PROBABILITY SEPARATION:")
        max_probs = probs.max(axis=1)
        mean_max = max_probs.mean()
        pct_above_70 = (max_probs > 0.7).mean() * 100
        print(f"   Mean max probability: {mean_max:.3f}")
        print(f"   % hours with P>0.7: {pct_above_70:.1f}%")
        status = "✅ PASS" if mean_max > 0.75 else "❌ FAIL (uncertain)"
        print(f"   Status: {status}\n")

        print("3. REGIME PERSISTENCE (Transition Matrix):")
        trans = pd.crosstab(regime_assign.shift(1), regime_assign, normalize='index')
        print(trans.round(3))
        print("\n")

        # --- 4. VISUALIZATION (UPDATED: 3 Separate Plots) ---
        # Create 3 subplots sharing the x-axis
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Plot Regime 0
        axes[0].plot(probs.index, probs['Regime 0'], color='tab:blue', linewidth=1.5)
        axes[0].set_title('Regime 0 Smoothed Probability', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Probability')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05) # Fix y-axis to 0-1

        # Plot Regime 1
        axes[1].plot(probs.index, probs['Regime 1'], color='tab:orange', linewidth=1.5)
        axes[1].set_title('Regime 1 Smoothed Probability', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Probability')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.05, 1.05)

        # Plot Regime 2
        axes[2].plot(probs.index, probs['Regime 2'], color='tab:green', linewidth=1.5)
        axes[2].set_title('Regime 2 Smoothed Probability', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Probability')
        axes[2].set_xlabel('Time Step (Index)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.show()
            
    def fit_msm_full_series(self, max_iter=1000, search_reps=20, dummy =False):
        np.random.seed(123)
        """
        Fit Markov Switching Model on the de-seasonalized data.
        """
        if 'hourly_emissions_detrended' not in self.data.columns:
            raise ValueError("❌ Run dummy_variable_seasonality_extraction() first!")
        
        print("\n🔄 Fitting Markov Switching Model...")
        subset = self.data.copy().reset_index(drop=True)
     
        # Scale variables
        scaler_y = StandardScaler()
        scaler_x = StandardScaler()
        
        y_scaled = scaler_y.fit_transform(subset[['hourly_emissions_detrended']])
        
        X_unscaled = subset[['hourly_generation_renewables_detrended', 'hourly_generation_nonrenewables_detrended']]
        X_scaled = scaler_x.fit_transform(X_unscaled)
        X_comb = X_scaled
        vec = [True, True]
        # add a dummy in case T> MAY 2022
        if dummy:
            Time_dummy = (subset['date_hour'] >= pd.to_datetime('2022-05-01')).astype(int)
            X_interaction = X_scaled * Time_dummy.values.reshape(-1, 1)
            #add dummy to exog
            X_comb = np.hstack([X_scaled,X_interaction])
            vec = [True, True,True, True]
        # Fit model
        ms_model = MarkovAutoregression(
            endog=y_scaled,
            exog=X_comb,
            k_regimes=2,
            order=1,
            trend='c',
            switching_trend=True,
            switching_exog= vec,
            switching_variance=False
        )
        
        ms_results = ms_model.fit(
            maxiter=1000,
            em_iter=50,
            search_reps=20,      
            search_iter=10,
            disp=False
        )   
                
        self.ms_model = ms_model
        self.ms_results = ms_results
        self.scaler_y = scaler_y
        self.scaler_x = scaler_x
        
        # Identify High MEF Regime
        # Get scaled params
        
        p = ms_results.params
        print(p)
        if dummy==True:
            beta_0 =p[6]
            beta_1 = p[7]
        else:
            beta_0 =p[4]
            beta_1 = p[5]

        self.high_regime = 0 if beta_0 > beta_1 else 1
        self.low_regime = 1 - self.high_regime
        
        print(f"✅ MSM Fitted. High MEF Regime is: {self.high_regime}")
        print(ms_results.summary())
        return ms_results
    
    def assign_regimes_to_data(self):
        """Assign probabilities and labels to data"""
        if not hasattr(self, 'ms_results'):
            raise ValueError("❌ Run fit_msm_full_series() first!")
        
        print("\n🏷️  Assigning regimes...")
        raw_probs = self.ms_results.smoothed_marginal_probabilities
        
        if isinstance(raw_probs, pd.DataFrame):
            prob_high = raw_probs.iloc[:, self.high_regime].values
        else:
            prob_high = raw_probs[:, self.high_regime]
        
        # Align with data (AR(p) drops first observation)
        df = self.data.iloc[1:].copy().reset_index(drop=True)
        df['prob_high_regime'] = prob_high
        df['regime'] = (prob_high >= 0.5).astype(int)
        df['regime_label'] = df['regime'].map({1: 'High MEF', 0: 'Low MEF'})
        
        self.data_with_regimes = df
        return df

    # ==================== ANALYSIS & PLOTTING ====================
    
    def analyze_regime_by_time(self):
         
        """Text analysis of regime probabilities"""
        if self.data_with_regimes is None:
             raise ValueError("Run assign_regimes_to_data() first")
             
        df = self.data_with_regimes
        print("\n📊 REGIME ANALYSIS SUMMARY")
        print("-" * 30)
        print("Mean Prob of High MEF by Season:")
        print(df.groupby('season')['prob_high_regime'].mean())
        print("-" * 30)
    
    def plot_regime_time_series(self, save_path="econometrics_fig"):
        """Standard Time Series Plot colored by Regime"""
        self.set_publication_style()
        df = self.data_with_regimes
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
        
        # --- Top Panel: Data ---
        high = df[df['regime'] == 1]
        low = df[df['regime'] == 0]
        
        ax1.scatter(high['date_hour'], high['hourly_emissions_detrended'], color='#d62728', s=1, label='High MEF')
        ax1.scatter(low['date_hour'], low['hourly_emissions_detrended'], color='#1f77b4', s=1, label='Low MEF')
        
        # Styling Top Panel
        ax1.set_ylabel('Emissions (De-seasonalized)')
        ax1.set_title('Time Series by Regime')
        ax1.legend(markerscale=5)
        ax1.tick_params(axis='y') # Set y-tick size for top plot
        
        # --- Bottom Panel: Probability ---
        ax2.plot(df['date_hour'], df['prob_high_regime'], color='black', lw=0.5)
        ax2.fill_between(df['date_hour'], 0, df['prob_high_regime'], color='#d62728', alpha=0.3)
        
        # Styling Bottom Panel
        ax2.set_ylabel('Prob(High MEF)')
        ax2.set_title('Smoothed Probability of High MEF Regime')
        
        # 1. To change the FONT SIZE of the ticks (numbers):
        ax2.tick_params(axis='both', which='major') 
        
        # 2. To add the Label "Date" at the bottom
        ax2.set_xlabel('Date')
        if len(self.ms_results.params)>11: 
            name_path = "Regime_MSM_Dummy.eps"
        else:
            name_path = "Regime_MSM.eps"
        plt.tight_layout()
        if save_path: plt.savefig(save_path + "/" + name_path)
        plt.show()
    
    def plot_regime_polar_charts(self, save_path="econometrics_fig"):
        """
        Create 'Wind Rose' style Polar Bar Charts for High Regime occurrences.
        - Style: Wind Rose (Bars radiating from center).
        - Scale: Area-Preserving (Sqrt scale) so visual area matches % magnitude.
        - Limits: Dynamic (Max % + 10%).
        """
        if self.data_with_regimes is None:
            raise ValueError("Run assign_regimes_to_data() first")
            
        self.set_publication_style()
        df = self.data_with_regimes
        
        # 1. Filter for High Regime
        df_high = df[df['regime'] == 1].copy()
        if len(df_high) == 0:
            print("No High Regime observations found.")
            return

        total_high = len(df_high)
        fig = plt.figure(figsize=(18, 8))

        # --- Helper Function to Plot Each Rose ---
        def plot_rose(ax, categories, counts, color, title):
            # 1. Calculate Shares (Percentages)
            shares = counts / total_high
            
            # 2. Setup Angles and Widths
            N = len(categories)
            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
            width = (2*np.pi / N) * 0.9  # 90% width to leave a small gap between bars
            
            # 3. Setup Axis (North Top, Clockwise, Area-Preserving)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            # Sqrt scale ensures that the AREA of the wedge is proportional to the value
            ax.set_yscale('function', functions=(lambda x: np.sqrt(x), lambda x: x**2))
            
            # 4. Dynamic Limit Calculation (Max + 10%)
            max_share = shares.max()
            limit = max_share + 0.10
            ax.set_ylim(0, limit)
            
            # 5. Draw Bars (Wind Rose)
            # Note: We align 'center' so the bar straddles the tick mark
            ax.bar(theta, shares, width=width, bottom=0.0, color=color, alpha=0.6, edgecolor=color)
            
            # 6. Labels and Grids
            ax.set_xticks(theta)
            if categories ==["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                categories = [cat[:3] for cat in categories]  # Shorten day names
            ax.set_xticklabels(categories)
            
            # Clean up radial ticks (percentages) to avoid clutter
            # We construct nice ticks: e.g., 10%, 20%... up to limit
            ticks = np.arange(0.1, limit, 0.1)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{int(t*100)}%" for t in ticks], color='gray')
            ax.set_xticks(theta)
            ax.set_xticklabels(categories)
            ax.grid(True, alpha=0.3)
            
            # Increase title size too if labels are bigger
            ax.set_title(title, y=1.15)
            

        # --- 1. Seasonal Rose ---
        ax1 = fig.add_subplot(131, projection='polar')
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        s_counts = df_high['season'].value_counts().reindex(seasons, fill_value=0)
        plot_rose(ax1, seasons, s_counts, '#d62728', "Seasonality")

        # --- 2. Monthly Rose ---
        ax2 = fig.add_subplot(132, projection='polar')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Map month numbers 1-12 to names
        m_counts_raw = df_high['month'].value_counts()
        m_counts = [m_counts_raw.get(i, 0) for i in range(1, 13)]
        plot_rose(ax2, months, np.array(m_counts), '#1f77b4', "Monthly")

        # --- 3. Weekday Rose ---
        ax3 = fig.add_subplot(133, projection='polar')
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        w_counts = df_high['day'].value_counts().reindex(days, fill_value=0)
        plot_rose(ax3, days, w_counts, "#15b367", "Weekday")

        plt.tight_layout()
        if save_path: plt.savefig(save_path + "/regime_anal.eps")
        plt.show()

    # ==================== REGIME MARGIN ANALYSIS ====================
    def analyze_marginal_fuel_probability(self, save_path="econometrics_fig"):
        """
        Calculates the 'Marginal Fuel Probability' by regressing Fuel Gen on Total Load.
        Formula: Gen_Fuel = alpha + beta * Load + epsilon
        The beta coefficient represents the share of the marginal MW provided by that fuel.
        """
        if self.data_with_regimes is None:
            raise ValueError(" Run assign_regimes_to_data() first!")
            
        print("\n📈 MARGINAL FUEL PROBABILITY ANALYSIS (Slope of Gen vs Load)")
        print("-" * 60)
        
        df = self.data_with_regimes.copy()
        
        # Calculate Deltas (First Differences)
        cols_to_diff = ['D', 'Gen_Coal', 'Gen_Gas']
        df_diff = df[cols_to_diff].diff().dropna()
        # Re-attach regime (aligned index)
        df_diff['regime_label'] = df.loc[df_diff.index, 'regime_label']
        
        results = []
        
        # Loop through Regimes
        for label in ['Low MEF', 'High MEF']:
            subset = df_diff[df_diff['regime_label'] == label]
            
            row = {'Regime': label}
            
            # Regress Coal on Load
            model_coal = ols("Gen_Coal ~ D", data=subset).fit()
            row['Coal_Beta'] = model_coal.params['D']
            row['Coal_R2'] = model_coal.rsquared
            
            # Regress Gas on Load
            model_gas = ols("Gen_Gas ~ D", data=subset).fit()
            row['Gas_Beta'] = model_gas.params['D']
            row['Gas_R2'] = model_gas.rsquared
            
            results.append(row)
            
        res_df = pd.DataFrame(results).set_index('Regime')
        print(res_df)
        print("-" * 60)
        
        # --- VISUALIZATION (UPDATED FOR VISIBILITY) ---
        self.set_publication_style()
        
        # Increased figure size for better spacing
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plotting the Betas (Marginal Contribution)
        x = np.arange(len(res_df))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, res_df['Coal_Beta'], width, label='Coal', color='#555555')
        rects2 = ax.bar(x + width/2, res_df['Gas_Beta'], width, label='Gas', color='#1f77b4')
        
        # --- BIGGER LABELS & TICKS ---
        ax.set_ylabel('Marginal Probability')
        # X-Axis Ticks
        ax.set_xticks(x)
        ax.set_xticklabels(res_df.index)
        

        # Legend
        ax.legend( loc='upper left', ncol=1)
        ax.set_title('Marginal Fuel Probability by Regime')

        
        ax.axhline(0, color='black', linewidth=0.8)
        
        # Add value labels (Bigger and Bold)
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=18) # <-- Increased size
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path + "/marginal_state.eps")
        plt.show()
        
        
    def analyze_marginal_fuel_probability_sb(self, save_path="econometrics_fig"):
        """
        Calculates and plots Marginal Fuel Probability in a single combined plot.
        Uses Hatching to differentiate Pre and Post-Break periods.
        """
        if self.data_with_regimes is None:
            raise ValueError("Run assign_regimes_to_data() first!")
            
        df = self.data_with_regimes.copy()
        df.index = pd.to_datetime(df.index)
        
        # Define the Structural Break Date [Cite: 1233, 1380]
        break_date = pd.Timestamp('2022-05-31')
        df['period'] = np.where(df['date_hour'] <= break_date, 'Pre-Break', 'Post-Break')
        
        # Calculate First Differences [Cite: 1275, 1282]
        df_diff = df[['D', 'Gen_Coal', 'Gen_Gas']].diff().dropna()
        df_diff['regime_label'] = df.loc[df_diff.index, 'regime_label']
        df_diff['period'] = df.loc[df_diff.index, 'period']
        
        results = []
        for period in ['Pre-Break', 'Post-Break']:
            for label in ['Low MEF', 'High MEF']:
                subset = df_diff[(df_diff['period'] == period) & (df_diff['regime_label'] == label)]
                if len(subset) < 10: continue 
                
                row = {'Period': period, 'Regime': label}
                
                row['Coal_Beta'] = ols("Gen_Coal ~ D", data=subset).fit().params['D']
                row['Gas_Beta'] = ols("Gen_Gas ~ D", data=subset).fit().params['D']
                results.append(row)
                
        res_df = pd.DataFrame(results)

        # --- COMBINED VISUALIZATION ---
        self.set_publication_style()
        fig, ax = plt.subplots(figsize=(14, 8))

        regimes = ['Low MEF', 'High MEF']
        x = np.arange(len(regimes))
        width = 0.2  # Thinner bars to fit 4 per group

        # Style Configuration
        colors = {'Coal': '#555555', 'Gas': '#1f77b4'}
        hatches = {'Pre-Break': '', 'Post-Break': '///'} 
        alphas = {'Pre-Break': 1.0, 'Post-Break': 0.7}

        for i, regime in enumerate(regimes):
            for j, period in enumerate(['Pre-Break', 'Post-Break']):
                p_df = res_df[(res_df['Regime'] == regime) & (res_df['Period'] == period)]
                if p_df.empty: continue

                pos_coal = i - width*1.5 + (j * width)
                pos_gas = i + width*0.5 + (j * width)

                # Draw Coal Bar
                b1 = ax.bar(pos_coal, p_df['Coal_Beta'].values[0], width, 
                            color=colors['Coal'], hatch=hatches[period], 
                            alpha=alphas[period], edgecolor='black',
                            label='Coal' if (i==0 and j==0) else "")
                
                # Draw Gas Bar
                b2 = ax.bar(pos_gas, p_df['Gas_Beta'].values[0], width, 
                            color=colors['Gas'], hatch=hatches[period], 
                            alpha=alphas[period], edgecolor='black',
                            label='Gas' if (i==0 and j==0) else "")

                # Value Labels
                ax.annotate(f'{p_df["Coal_Beta"].values[0]:.2f}', xy=(pos_coal, p_df['Coal_Beta'].values[0]),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=14)
                ax.annotate(f'{p_df["Gas_Beta"].values[0]:.2f}', xy=(pos_gas, p_df['Gas_Beta'].values[0]),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=14)

        # Formatting
        ax.set_ylabel('Marginal Probability')
        ax.set_title('Marginal Fuel Probability by Regime and Period',pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_ylim(0, 0.65) # Adjusted based on typical results [Cite: 1288, 1545]

        # Custom Legend to explain the Hatching (Pre vs Post)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Pre-Break'),
            Patch(facecolor='white', edgecolor='black', hatch='///', label='Post-Break'),
            Patch(facecolor=colors['Coal'], label='Coal'),
            Patch(facecolor=colors['Gas'], label='Gas')
        ]
        ax.legend(handles=legend_elements, loc='upper left',fontsize=14, frameon=True)

        plt.tight_layout()
        if save_path: plt.savefig(save_path + "/pre_post_marginal_gen.eps")
        plt.show()
        
    # ==================== Hourly MSM  Parallel ====================
    def fit_msm_monthly(self, max_iter=2000, search_reps=100, n_jobs=-1):
        """
        Fit Markov Switching Model on the de-seasonalized data.
        n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        if 'hourly_emissions_detrended' not in self.data.columns:
            raise ValueError("❌ Run dummy_variable_seasonality_extraction() first!")
        
        print("\n🔄 Fitting Markov Switching Model...")
        subset = self.data.copy().reset_index(drop=True)
        
        
        # Split into periods
        # 1. Precise Split at May 1, 2022
        break_date = pd.to_datetime('2022-05-01')
        subset_pre = subset[subset['date_hour'] < break_date].copy()
        subset_post = subset[subset['date_hour'] >= break_date].copy()
        #modift trnd into substet post to start from 1
        subset_post['trend'] = np.arange(1, len(subset_post) + 1)
        # 2. Detrending logic (Keeping your OLS structure)
        for df_period in [subset_pre, subset_post]:
            y_formula = "hourly_emissions_mlb ~ C(interval)*C(month) + C(month)*C(year) + trend"
            model_em = ols(y_formula, data=df_period).fit()
            df_period['em_res'] = model_em.resid + model_em.params.iloc[0]
            
            x_formula = "hourly_generation_nonrenewables_mkwh ~ C(interval)*C(month) +C(month)*C(year) + trend"
            model_x = ols(x_formula, data=df_period).fit()
            df_period['nrg_res'] = model_x.resid + model_x.params.iloc[0]
            
            x1_formula = "hourly_generation_renewables_mkwh ~ C(interval)*C(month) +C(month)*C(year) + trend"
            model_x1 = ols(x1_formula, data=df_period).fit()
            df_period['rg_res'] = model_x1.resid + model_x1.params.iloc[0]
        
        
        # Create tasks for parallel processing
        tasks = []
        for subset_df, period in [(subset_pre, '2019-2022_05'), (subset_post, '2022_05-2025')]:
            unique_months = subset_df['month'].unique()
            for month in unique_months:
                target_data = subset_df[subset_df['month'] == month]
                for hour in range(1, 25):  # 1 to 24
                    target_data_hour = target_data[target_data['Local_Hour'] == hour]
                    tasks.append((target_data_hour, period, month, hour, max_iter, search_reps))
        print(f"Total tasks for parallel execution: {len(tasks)}")
        # Parallel execution
        raw = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(self._fit_single_msm)(*task) for task in tasks
        )
        
        results = [r for r in raw if r is not None]
        self.hourly_msm_results = pd.DataFrame(results)
        return pd.DataFrame(results)
    
    def _fit_single_msm(self, subset_df, period, month, hour, max_iter, search_reps ):
        """ Helper function to fit a single MSM model for one hour """
        np.random.seed(123)
        subset_hour = subset_df
        # cols_to_clean = ['em_res', 'nrg_res']
        # #we want to downside the effect of outlires which can cause convergence issues. 
        # for col in cols_to_clean:
        #     lower = subset_hour[col].quantile(0.05)
        #     upper = subset_hour[col].quantile(0.95)
        #     subset_hour[col] = subset_hour[col].clip(lower=lower, upper=upper)
        
        # Calculate average load
        average_load = subset_hour['D'].mean()
        # Scale variables
        scaler_y = StandardScaler()
        scaler_x = StandardScaler()
        
        # ---- scale ----------------------------------------------------------------
        y_scaled = scaler_y.fit_transform(subset_hour[['em_res']])
        X_scaled = scaler_x.fit_transform(subset_hour[['nrg_res']])
        #X_unscaled = subset_hour[['rg_res', 'nrg_res']]
        #X_scaled = scaler_x.fit_transform(X_unscaled)
        # Fit model
        try:
            
            ms_model = MarkovAutoregression(
                endog=y_scaled,
                exog=X_scaled,
                k_regimes=2,
                order=1,
                trend='c',
                switching_trend=True,
                switching_exog=True,
                switching_variance=False
            )
            
            ms_results = ms_model.fit(  tol = 1e-6,
                                        search_reps=search_reps, 
                                        method='powell',
                                        cov_type='opg',
                                        em_iter=50,
                                        disp=False
                                     )   
                
            
        except Exception as e:
            print(f"Model failed for {period} Month: {month} Hour: {hour}. Error: {e}")
            return None
        # Extract and convert parameters
        
        p = ms_results.params
        beta_0 = p[4]
        beta_1 = p[5]
         
    
        idx_H = 4 if beta_0 > beta_1 else 5
        idx_L = 5 if beta_0 > beta_1 else 4
        beta_high_regime_s , SE_beta_high_regime_s = p[idx_H], ms_results.bse[idx_H]
        beta_low_regime_s, SE_beta_low_regime_s = p[idx_L], ms_results.bse[idx_L]
        
        scale_y = float(scaler_y.scale_[0])
        scale_nonren = float(scaler_x.scale_[0])
        convert_factor = scale_y / scale_nonren
        
        beta_high_regime = beta_high_regime_s * convert_factor
        beta_low_regime = beta_low_regime_s * convert_factor
        SE_beta_high_regime = SE_beta_high_regime_s * abs(convert_factor)
        SE_beta_low_regime = SE_beta_low_regime_s * abs(convert_factor)
        
        return dict(
            period=period, month=month, hour=hour,
            beta_low=beta_low_regime,   se_low=SE_beta_low_regime,   
            beta_high=beta_high_regime, se_high=SE_beta_high_regime, 
            average_load=average_load )
        
    # ==================== Hourly MSM  Parallel ====================
    def fit_msm_hourly(self, max_iter=1000, search_reps=50, n_jobs=-1):
        """
        Fit Markov Switching Model on the de-seasonalized data.
        n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        if 'hourly_emissions_detrended' not in self.data.columns:
            raise ValueError("❌ Run dummy_variable_seasonality_extraction() first!")
        np.random.seed(123)
        print("\n🔄 Fitting Markov Switching Model...")
        subset = self.data.copy().reset_index(drop=True)
        
       
        # Split into periods
        # 1. Precise Split at May 1, 2022
        break_date = pd.to_datetime('2022-05-01')
        subset_pre = subset[subset['date_hour'] < break_date].copy()
        subset_post = subset[subset['date_hour'] >= break_date].copy()
        #modift trnd into substet post to start from 1
        subset_post['trend'] = np.arange(1, len(subset_post) + 1)
        # Detrend both subsets
        # 2. Detrending logic (Keeping your OLS structure)
        for df_period in [subset_pre, subset_post]:
            y_formula = "hourly_emissions_mlb ~ C(interval)*C(month) + C(month)*C(year) + trend"
            model_em = ols(y_formula, data=df_period).fit()
            df_period['em_res'] = model_em.resid + model_em.params.iloc[0]
            
            x_formula = "hourly_generation_nonrenewables_mkwh ~ C(interval)*C(month) +C(month)*C(year) + trend"
            model_x = ols(x_formula, data=df_period).fit()
            df_period['nrg_res'] = model_x.resid + model_x.params.iloc[0]
            
            x1_formula = "hourly_generation_renewables_mkwh ~ C(interval)*C(month) +C(month)*C(year) + trend"
            model_x1 = ols(x1_formula, data=df_period).fit()
            df_period['rg_res'] = model_x1.resid + model_x1.params.iloc[0]
            
        # Create tasks for parallel processing
        tasks = []
        for subset_df, period in [(subset_pre, 'Pre-break'), (subset_post, 'After-break')]:
            for hour in range(1, 25):
                tasks.append((subset_df, period, hour, max_iter, search_reps))
        
        # Parallel execution
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._fit_single_msm_hourly)(*task) for task in tasks
        )
        
        # Collect results
        ms_results_year = {
            "hour": [r[0] for r in results],
            "period": [r[1] for r in results],
            "beta_low": [r[2] for r in results],
            "beta_high": [r[3] for r in results],
            "se_low": [r[4] for r in results],
            "se_high": [r[5] for r in results],
            "average_load": [r[6] for r in results]
        }
        
        return pd.DataFrame(ms_results_year)

    def _fit_single_msm_hourly(self, subset_df, period, hour, max_iter, search_reps):
  
        np.random.seed(123)
        subset_hour = subset_df[subset_df['Local_Hour'] == hour]
        # Calculate average load
        average_load = subset_hour['D'].mean()
        # Scale variables
        scaler_y = StandardScaler()
        scaler_x = StandardScaler()
          
        
        y_scaled = scaler_y.fit_transform(subset_hour[['em_res']])
        X_scaled = scaler_x.fit_transform(subset_hour[['nrg_res']])
        #X_unscaled = subset_hour[['rg_res', 'nrg_res']]
        #X_scaled = scaler_x.fit_transform(X_unscaled)
        # Fit model

        ms_model = MarkovAutoregression(
            endog=y_scaled,
            exog=X_scaled,
            k_regimes=2,
            order=1,
            trend='c',
            switching_trend=True,
            switching_exog=True,
            switching_variance=False
        )
        #dolve paramter issues by using search method
        ms_results = ms_model.fit(  
                                    maxiter=max_iter,
                                    search_reps=search_reps, 
                                    method='bfgs',
                                    cov_type='opg',
                                    disp=False
                                    )   
        
        # Extract and convert parameters
        p = ms_results.params
        beta_0 = p[4]
        beta_1 = p[5]
        
        idx_H = 4 if beta_0 > beta_1 else 5
        idx_L = 5 if beta_0 > beta_1 else 4
        beta_high_regime_s , SE_beta_high_regime_s = p[idx_H], ms_results.bse[idx_H]
        beta_low_regime_s, SE_beta_low_regime_s = p[idx_L], ms_results.bse[idx_L]
        
        scale_y = float(scaler_y.scale_[0])
        scale_nonren = float(scaler_x.scale_[0])
        convert_factor = scale_y / scale_nonren
        
        beta_high_regime = beta_high_regime_s * convert_factor
        beta_low_regime = beta_low_regime_s * convert_factor
        SE_beta_high_regime = SE_beta_high_regime_s * abs(convert_factor)
        SE_beta_low_regime = SE_beta_low_regime_s * abs(convert_factor)
        
        return (hour, period, beta_low_regime, beta_high_regime, SE_beta_low_regime, SE_beta_high_regime,average_load)

    def compute_beta_tests(self):
        """
        For each (month, hour) pair test H0: beta_high_before = beta_high_after
        and separately for beta_low, using a Welch two-sample t-test.

        Since we only have one point estimate + SE per cell we use the
        standard z/t approximation:
            t = (b1 - b2) / sqrt(se1^2 + se2^2)
        """
        df = self.hourly_msm_results
        periods = df['period'].unique()
        
        p1, p2 = sorted(periods)          # e.g. '2019-2022', '2022-2025'
        rows = []

        for month in sorted(df['month'].unique()):
            for hour in sorted(df['hour'].unique()):
                d1 = df[(df['period'] == p1) & (df['month'] == month) & (df['hour'] == hour)]
                d2 = df[(df['period'] == p2) & (df['month'] == month) & (df['hour'] == hour)]

                if d1.empty or d2.empty:
                    continue

                for regime in ('high', 'low'):
                    b1  = d1[f'beta_{regime}'].values[0]
                    b2  = d2[f'beta_{regime}'].values[0]
                    se1 = d1[f'se_{regime}'].values[0]
                    se2 = d2[f'se_{regime}'].values[0]

                    se_diff = np.sqrt(se1**2 + se2**2)
                    if se_diff == 0:
                        continue

                    t_stat = (b1 - b2) / se_diff
                    p_val  = 2 * norm.sf(abs(t_stat))   # two-tailed z-test

                    rows.append(dict(
                        month=month, regime=regime,
                        beta_before=b1, beta_after=b2,
                        se_before=se1,  se_after=se2,
                        t_stat=t_stat,  p_value=p_val,
                        significant_5pct=(p_val < 0.05),
                        significant_1pct=(p_val < 0.01),
                    ))

        return pd.DataFrame(rows)
    
    def plot_mef_analysis_hourly(self, data_input,save_path="econometrics_fig"):
        """
        Generates a 2-panel plot with:
        - MEF Y-axis range: 0.85 to 1.65
        - Load histogram visually occupies exactly 1/3 of plot height
        - Load axis ticks only where histogram ends
        - Two separate histograms for each period
        - Single centered legend below both plots with larger size
        """
        self.set_publication_style()
        
        # 1. Load Data

        df = data_input

        # 2. Prepare Data
        periods = df['period'].unique()
        
        # Calculate hourly load for EACH period separately
        hourly_load_by_period = {}
        for period in periods:
            period_data = df[df['period'] == period]
            hourly_load_by_period[period] = period_data.groupby('hour')['average_load'].mean()
        
        # Get overall max for consistent scaling
        all_loads = pd.concat(hourly_load_by_period.values())
        load_max_value = all_loads.max()

        # 3. Setup Plot with space for legend at bottom
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        plot_configs = [
            {'col': 'beta_low',  'se_col': 'se_low',  'ax': axes[0], 'title': 'Low Marginal Emissions (Hourly)',  'ylabel': 'Marginal $CO_2$ Emissions'},
            {'col': 'beta_high', 'se_col': 'se_high', 'ax': axes[1], 'title': 'High Marginal Emissions (Hourly)', 'ylabel': 'Marginal $CO_2$ Emissions'}
        ]

        # MEF axis constraints
        MEF_MIN = 0.7
        MEF_MAX = 1.7
        MEF_RANGE = MEF_MAX - MEF_MIN
        
        # Bar colors for the two periods
        bar_colors = ['#ffcccc', '#ff9999']  # Light pink and darker pink
        bar_alphas = [0.3, 0.5]
        
        # Store handles and labels for the legend
        all_handles = []
        all_labels = []
        
        for idx, config in enumerate(plot_configs):
            ax = config['ax']
            col_name = config['col']
            
            # --- Primary Axis: MEF Lines ---
            linestyles = ['-', '--']
            colors = ['#1f77b4', "#6098e2"]  # Darker and lighter blue
      
            for i, period in enumerate(periods):
                subset = df[df['period'] == period]
                line, = ax.plot(subset['hour'], subset[col_name], 
                        color=colors[i % len(colors)], 
                        linewidth=2.5, 
                        linestyle=linestyles[i % len(linestyles)],
                        zorder=3)
                # ADD Confidence Interval Bands (±1 SE)
                ax.fill_between(subset['hour'],
                                subset[col_name] - subset[config['se_col']],
                                subset[col_name] + subset[config['se_col']],
                                color=colors[i % len(colors)],
                                alpha=0.15,
                                zorder=2,
                                linewidth=0)
            
                # Collect handles only from first plot to avoid duplicates
                if idx == 0:
                    all_handles.append(line)
                    all_labels.append(period)

            # --- MEF Axis limits ---
            ax.set_ylim(MEF_MIN, MEF_MAX)

            # Format Primary Axis
            ax.set_title(config['title'])
            ax.set_xlabel('Hour')
            ax.set_ylabel(config['ylabel'], color='black')
            ax.tick_params(axis='y', labelcolor='black')
            ax.tick_params(axis='x')
            ax.grid(False)
            ax.set_xticks([1, 6, 12, 18, 24])
            ax.spines['top'].set_visible(False)
            
            # --- Secondary Axis: Electricity Demand (Histogram) ---
            ax_load = ax.twinx()
            
            # Calculate load axis range so bars occupy 1/3 of visual height
            load_axis_max = load_max_value * 3
            
            # Plot TWO histograms - one for each period
            width = 0.4  # Width of each bar
            hours = range(1, 25)
            
            for i, period in enumerate(periods):
                offset = (i - 0.5) * width  # Offset bars so they're side-by-side
                hourly_load = hourly_load_by_period[period]
                
                bars = ax_load.bar([h + offset for h in hourly_load.index], 
                        hourly_load.values, 
                        color=bar_colors[i], 
                        alpha=bar_alphas[i], 
                        width=width, 
                        zorder=1)
                
                # Collect bar handles only from first plot
                if idx == 0:
                    all_handles.append(bars)
                    all_labels.append(f'{period} (Load)')
            
            # --- Limit load axis ticks to where histogram ends ---
            ax_load.set_ylim(0, load_axis_max)
            
            # Set tick locations to only show in the bottom 1/3 where data exists
            tick_interval = load_max_value / 3
            load_ticks = [0, tick_interval, tick_interval*2, load_max_value]
            ax_load.set_yticks(load_ticks)
            ax_load.set_yticklabels([f'{int(t)}' for t in load_ticks])
            
            # Format Load Axis
            ax_load.set_ylabel('Electricity Demand', fontsize=16, labelpad=10)
            ax_load.tick_params(axis='y', labelsize=16)
            ax_load.spines['top'].set_visible(False)
            
            # Ensure MEF lines appear on top
            ax.set_zorder(ax_load.get_zorder() + 1)
            ax.patch.set_visible(False)

        # --- Create single centered legend below both plots with LARGER markers ---
            fig.legend(all_handles, all_labels, 
                loc='lower center',
                bbox_to_anchor=(0.5, -0.1),
                ncol=2, 
                frameon=True,
                edgecolor='black',
                fancybox=False,
                markerscale=2.0,  # Makes markers bigger
                handlelength=3.0,  # Makes lines longer
                handleheight=1.5,  # Makes legend entries TALLER
                borderpad=1.0,  # More padding inside legend box
                labelspacing=0.8)  # More vertical space between entries
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # More room for larger legend
        if save_path: plt.savefig(save_path + "/Hourly_mef.eps")
        plt.show()

if __name__ == "__main__":
    # 1. Setup
    file_path = "data.xlsx" # REPLACE THIS WITH YOUR PATH
    model = LightMEF(file_path)
    
    # 2. Load
    # model.load_and_clean_data()
    
    # 3. Create Time Features
    # model.create_time_variables()
    
    # 4. Detrend (Dummy Variable Method)
    # model.dummy_variable_seasonality_extraction()
    
    # 5. Fit Model & Assign Regimes
    # model.fit_msm_full_series()
    # model.assign_regimes_to_data()
    
    # 6. Plots
    # model.plot_time_series_comparison()
    # model.plot_regime_time_series()
    # model.plot_regime_polar_charts()