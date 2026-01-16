"""
US MEF (Marginal Emission Factor) Analysis with Markov Switching
Methodology: Dummy Variable Seasonality Extraction (Kapoor et al., 2023 adapted)
"""

from tracemalloc import start
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import seaborn as sns
from datetime import datetime
import matplotlib as mpl
import warnings

from sympy import beta
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
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
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
            'NRNG': 'hourly_generation_nonrenewables'
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
            'hourly_emissions_mlb', 'hourly_generation_mkwh','hourly_generation_renewables_mkwh','hourly_generation_nonrenewables_mkwh'
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
    
    def fit_msm_full_series(self, max_iter=1000, search_reps=20):
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
        # Fit model
        ms_model = MarkovAutoregression(
            endog=y_scaled,
            exog=X_scaled,
            k_regimes=2,
            order=1,
            trend='c',
            switching_trend=True,
            switching_exog= [True, True],
            switching_variance=False
        )
        
        ms_results = ms_model.fit(maxiter=max_iter, search_reps=search_reps, disp=False)
        
        self.ms_model = ms_model
        self.ms_results = ms_results
        self.scaler_y = scaler_y
        self.scaler_x = scaler_x
        
        # Identify High MEF Regime
        # (Assuming NonRen coefficient is higher in High MEF regime)

        # Get scaled params
        p = ms_results.params
        print(p)
        beta_0 =p[6]
        beta_1 = p[7]
    
            
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
        
        # Align with data (AR(1) drops first observation)
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
    
    def plot_time_series_comparison(self, save_path=None):
        """Compare Original vs De-seasonalized Data"""
        self.set_publication_style()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        df = self.data
        
        # 1. Emissions
        axes[0].plot(df['date_hour'], df['hourly_emissions_mlb'], color='gray', alpha=0.6, label='Original')
        axes[0].plot(df['date_hour'], df['hourly_emissions_detrended'], color='darkred', lw=1, label='De-seasonalized (Dummy Method)')
        axes[0].set_title('Emissions: Original vs. De-seasonalized')
        axes[0].legend()
        
        # 2. Non-Ren Generation
        axes[1].plot(df['date_hour'], df['hourly_generation_nonrenewables_mkwh'], color='gray', alpha=0.6, label='Original')
        axes[1].plot(df['date_hour'], df['hourly_generation_nonrenewables_detrended'], color='darkblue', lw=1, label='De-seasonalized')
        axes[1].set_title('Non-Renewable Gen: Original vs. De-seasonalized')
        axes[1].legend()
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path + "/time_series_comparison.png")
        plt.show()

    def plot_regime_time_series(self, save_path=None):
        """Standard Time Series Plot colored by Regime"""
        self.set_publication_style()
        df = self.data_with_regimes
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot Data
        high = df[df['regime'] == 1]
        low = df[df['regime'] == 0]
        
        ax1.scatter(high['date_hour'], high['hourly_emissions_detrended'], color='#d62728', s=1, label='High MEF')
        ax1.scatter(low['date_hour'], low['hourly_emissions_detrended'], color='#1f77b4', s=1, label='Low MEF')
        ax1.set_ylabel('Emissions (De-seasonalized)')
        ax1.set_title('Time Series colored by Regime',loc='left', weight='bold')
        ax1.legend(markerscale=5)
        
        # Plot Probability
        ax2.plot(df['date_hour'], df['prob_high_regime'], color='black', lw=0.5)
        ax2.fill_between(df['date_hour'], 0, df['prob_high_regime'], color='#d62728', alpha=0.3)
        ax2.set_ylabel('Prob(High MEF)')
        ax2.set_title('Smoothed Probability of High MEF Regime',loc='left', weight='bold')
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path + "/regime_time_series.png")
        plt.show()

    def plot_regime_polar_charts(self, save_path=None):
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
            print("⚠️ No High Regime observations found.")
            return

        total_high = len(df_high)
        fig = plt.figure(figsize=(15, 5))

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
            ax.set_xticklabels(categories, fontweight='bold', fontsize=9)
            
            # Clean up radial ticks (percentages) to avoid clutter
            # We construct nice ticks: e.g., 10%, 20%... up to limit
            ticks = np.arange(0.1, limit, 0.1)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{int(t*100)}%" for t in ticks], fontsize=7, color='gray')
            ax.set_xticks(theta)
            ax.set_xticklabels(categories, fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Increase title size too if labels are bigger
            ax.set_title(title, y=1.15, fontweight='bold', fontsize=13)
            

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
        if save_path: plt.savefig(save_path + "/regime_wind_rose.png")
        plt.show()


    # ==================== Hourly MSM  Parallel ====================
    def fit_msm_hourly(self, max_iter=1000, search_reps=20, n_jobs=-1):
        """
        Fit Markov Switching Model on the de-seasonalized data.
        n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        if 'hourly_emissions_detrended' not in self.data.columns:
            raise ValueError("❌ Run dummy_variable_seasonality_extraction() first!")
        
        print("\n🔄 Fitting Markov Switching Model...")
        subset = self.data.copy().reset_index(drop=True)
        
        # Split into periods
        year_first_period = [2019, 2020, 2021, 2022]
        year_second_period = [2023, 2024, 2025]
        subset_1 = subset[subset['year'].isin(year_first_period)].copy()
        subset_2 = subset[subset['year'].isin(year_second_period)].copy()
        
        # Detrend both subsets
        for subset_df in [subset_1, subset_2]:
            y_formula = "hourly_emissions_mlb ~ C(interval)*C(month)+C(month)*C(year) + trend"
            model_em = ols(y_formula, data=subset_df).fit()
            subset_df['em_res'] = model_em.resid + model_em.params.iloc[0]
            
            x_formula = "hourly_generation_nonrenewables_mkwh ~ C(interval)*C(month) +C(month)*C(year) + trend"
            model_x = ols(x_formula, data=subset_df).fit()
            subset_df['nrg_res'] = model_x.resid + model_x.params.iloc[0]
            
            x1_formula = "hourly_generation_renewables_mkwh ~ C(interval)*C(month) +C(month)*C(year) + trend"
            model_x1 = ols(x1_formula, data=subset_df).fit()
            subset_df['rg_res'] = model_x1.resid + model_x1.params.iloc[0]
        
        # Create tasks for parallel processing
        tasks = []
        for subset_df, period in [(subset_1, '2019-2022'), (subset_2, '2023-2025')]:
            for hour in range(1, 25):
                tasks.append((subset_df, period, hour, max_iter, search_reps))
        
        # Parallel execution
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._fit_single_msm)(*task) for task in tasks
        )
        
        # Collect results
        ms_results_year = {
            "hour": [r[0] for r in results],
            "period": [r[1] for r in results],
            "beta_low": [r[2] for r in results],
            "beta_high": [r[3] for r in results],
            "average_load": [r[4] for r in results]
        }
        
        return pd.DataFrame(ms_results_year)

    def _fit_single_msm(self, subset_df, period, hour, max_iter, search_reps):
        """Helper function to fit a single MSM model for one hour."""
        subset_hour = subset_df[subset_df['Local_Hour'] == hour]
        # Calculate average load
        average_load = subset_hour['D'].mean()
        # Scale variables
        scaler_y = StandardScaler()
        scaler_x = StandardScaler()
          
        
        y_scaled = scaler_y.fit_transform(subset_hour[['em_res']])
        #X_scaled = scaler_x.fit_transform(subset_hour[['nrg_res']])
        X_unscaled = subset_hour[['rg_res', 'nrg_res']]
        X_scaled = scaler_x.fit_transform(X_unscaled)
        # Fit model
        ms_model = MarkovAutoregression(
            endog=y_scaled,
            exog=X_scaled,
            k_regimes=2,
            order=1,
            trend='c',
            switching_trend=True,
            switching_exog=[True, True],
            switching_variance=False
        )
        
        ms_results = ms_model.fit(maxiter=max_iter, search_reps=search_reps, disp=False)
        
        # Extract and convert parameters
        p = ms_results.params
        beta_0 = p[6]
        beta_1 = p[7]
        
        beta_high_regime_s = p[6] if beta_0 > beta_1 else p[7]
        beta_low_regime_s = p[7] if beta_0 > beta_1 else p[6]
        
        scale_y = float(scaler_y.scale_[0])
        scale_nonren = float(scaler_x.scale_[1])
        convert_factor = scale_y / scale_nonren
        
        beta_high_regime = beta_high_regime_s * convert_factor
        beta_low_regime = beta_low_regime_s * convert_factor
        
        return (hour, period, beta_low_regime, beta_high_regime, average_load)

    def plot_mef_analysis_hourly(self, data_input):
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
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        plot_configs = [
            {'col': 'beta_low', 'ax': axes[0], 'title': 'Low Marginal Emissions (Hourly)', 'ylabel': 'Marginal $CO_2$ Emissions'},
            {'col': 'beta_high', 'ax': axes[1], 'title': 'High Marginal Emissions (Hourly)', 'ylabel': 'Marginal $CO_2$ Emissions'}
        ]

        # MEF axis constraints
        MEF_MIN = 0.8
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
            colors = ['#1f77b4', '#aec7e8']  # Darker and lighter blue
            
            for i, period in enumerate(periods):
                subset = df[df['period'] == period]
                line, = ax.plot(subset['hour'], subset[col_name], 
                        color=colors[i % len(colors)], 
                        linewidth=2.5, 
                        linestyle=linestyles[i % len(linestyles)],
                        zorder=3)
                
                # Collect handles only from first plot to avoid duplicates
                if idx == 0:
                    all_handles.append(line)
                    all_labels.append(period)

            # --- MEF Axis limits ---
            ax.set_ylim(MEF_MIN, MEF_MAX)

            # Format Primary Axis
            ax.set_title(config['title'], fontsize=13, pad=12, weight='bold')
            ax.set_xlabel('Hour', fontsize=11)
            ax.set_ylabel(config['ylabel'], fontsize=11, color='black')
            ax.tick_params(axis='y', labelcolor='black', labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
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
            ax_load.set_ylabel('Electricity Demand', fontsize=11, labelpad=10)
            ax_load.tick_params(axis='y', labelsize=10)
            ax_load.spines['top'].set_visible(False)
            
            # Ensure MEF lines appear on top
            ax.set_zorder(ax_load.get_zorder() + 1)
            ax.patch.set_visible(False)

        # --- Create single centered legend below both plots with LARGER markers ---
            fig.legend(all_handles, all_labels, 
                loc='lower center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=2, 
                frameon=True,
                fontsize=11,
                edgecolor='black',
                fancybox=False,
                markerscale=2.0,  # Makes markers bigger
                handlelength=3.0,  # Makes lines longer
                handleheight=1.5,  # Makes legend entries TALLER
                borderpad=1.0,  # More padding inside legend box
                labelspacing=0.8)  # More vertical space between entries
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # More room for larger legend
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