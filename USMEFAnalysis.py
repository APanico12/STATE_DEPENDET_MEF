"""
US MEF (Marginal Emission Factor) Analysis
Database: Region_US48_Hourly_NEW
Spatial Dimension: US (Lower 48)
Covered Years: 2019-2022
Approach: Intra-day and inter-day
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from arch.unitroot import DFGLS
from scipy.stats import f
from statsmodels.tsa.stattools import bds
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.formula.api import ols
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")


class USMEFAnalysis:
    """US Marginal Emission Factor Analysis"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.data = None
    # --- Publication Style ---
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
            'font.size': 7,
            'axes.titlesize': 8,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 6,
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

    @staticmethod
    def get_figsize(width_mm, ratio=0.618):
        """
        Get figure size in inches for a specific width in mm.
        
        Common Journal Widths:
        - Nature/Science Single Column: 89 mm (3.5 inches)
        - Nature/Science Double Column: 183 mm (7.2 inches)
        """
        width_inches = width_mm / 25.4
        height_inches = width_inches * ratio
        return (width_inches, height_inches)

    # --- USAGE EXAMPLE ---
    # set_publication_style()
    # fig, ax = plt.subplots(figsize=get_figsize(89)) # Single column width
    # plt.plot([1, 2, 3], [1, 4, 9], label='Model A')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Intensity (a.u.)')
    # sns.despine() # Final cleanup    
    def load_and_clean_data(self, start_date='2019-01-01', end_date='2025-09-30'):
        """Load and clean full EIA hourly dataset (Region_US48.xlsx)"""

        # --- Load Excel data ---
        df = pd.read_excel(self.data_path)
        df.columns = df.columns.str.strip()  # clean up column names

        # --- Parse datetime columns ---
        df['bid_offer_date'] = pd.to_datetime(df['UTC time']).dt.date
        df['date_hour'] = pd.to_datetime(df['UTC time'])

        # --- Filter by date range ---
        df = df[
            (df['bid_offer_date'] >= pd.to_datetime(start_date).date()) &
            (df['bid_offer_date'] <= pd.to_datetime(end_date).date())
        ].copy()
        # --- Create hourly interval (1–24 repeated) ---
        df['interval'] = df['date_hour'].dt.hour + 1  # 1-24
        # --- Rename columns to match R naming convention ---
        rename_map = {
            'Demand': 'D',
            'Net generation': 'NG',
            'Sum (NG)': 'hourly_generation',
            'CO2 Emissions Generated': 'hourly_emissions',
            'CO2 Emissions Intensity for Generated Electricity': 'CO2EmissionsIntensity'
        }
        df = df.rename(columns=rename_map)

        # --- Convert units ---
        df['hourly_emissions_mlb'] = df['hourly_emissions'] * 2204.6 / 1_000_000  # metric tons → million lbs
        df['hourly_generation_mkwh'] = df['hourly_generation'] * 1000 / 1_000_000  # GWh → million kWh

        # --- Select final columns  ---
        keep_cols = [
            'bid_offer_date', 'interval', 'date_hour', 
            'D', 'NG',
            'hourly_generation', 'hourly_emissions', 
            'CO2EmissionsIntensity', 
            'hourly_emissions_mlb', 'hourly_generation_mkwh'
        ]
        self.data = df[[col for col in keep_cols if col in df.columns]].copy()

        # --- Add index column T ---
        self.data['T'] = np.arange(1, len(self.data) + 1)

        # --- Summary ---
        print("✅ Data loaded and cleaned successfully.")
        print(f"   Date range: {start_date} → {end_date}")
        print(f"   Observations: {len(self.data)}")
        print(f"   Columns: {list(self.data.columns)}")

        return self.data
        
    
    def create_time_variables(self):
        """Create time dummies and factor variables"""
        df = self.data
        df['bid_offer_date'] = pd.to_datetime(df['bid_offer_date'], errors='coerce')
        # Extract time components
        df['month'] = df['bid_offer_date'].dt.month
        df['year'] = df['bid_offer_date'].dt.year
        df['day'] = df['bid_offer_date'].dt.day_name()
        df['dom'] = df['bid_offer_date'].dt.day
        
        # Month dummies
        for i in range(1, 13):
            df[f'm{i}'] = (df['month'] == i).astype(int)
        
        # Season (astronomical) 
        season_map = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
                     5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
        df['season'] = df['month'].map(season_map)
        
        # Hour dummies (1-24)
        for i in range(1, 25):
            df[f'h{i}'] = (df['interval'] == i).astype(int)
        
        # Day of week dummies
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day in enumerate(days, 1):
            df[f'd{i}'] = (df['day'] == day).astype(int)
        
        # Year dummies
        for year in df['year'].unique():
            df[f'y{year}'] = (df['year'] == year).astype(int)
        
        # Trend
        df['trend'] = np.arange(1, len(df) + 1)
        
        # Year-month and year-season
        df['yearmonth'] = df['bid_offer_date'].dt.to_period('M').astype(str)
        df['yearseason'] = df['year'].astype(str) + '-' + df['season']
        
        # Handle winter spanning years
        winter_mask = (df['month'].isin([1, 2])) & (df['year'] > df['year'].min())
        df.loc[winter_mask, 'yearseason'] = 'Winter_' + (df.loc[winter_mask, 'year'] - 1).astype(str) + '_' + df.loc[winter_mask, 'year'].astype(str)
        
        winter_mask_end = (df['month'] == 12)
        df.loc[winter_mask_end, 'yearseason'] = 'Winter_' + df.loc[winter_mask_end, 'year'].astype(str) + '_' + (df.loc[winter_mask_end, 'year'] + 1).astype(str)
        
        # Factor variables
        df['factoryear'] = df['year'].astype('category')
        df['factorinterval'] = df['interval'].astype('category')
        df['factormonth'] = df['month'].astype('category')
        df['factordow'] = df['day'].astype('category')
        df['yeartrend'] = df['year'] - df['year'].min()
        
        # Season trend
        season_order = df.groupby('yearseason')['bid_offer_date'].min().sort_values()
        season_to_num = {season: i for i, season in enumerate(season_order.index)}
        df['seasonstrend'] = df['yearseason'].map(season_to_num)
        
        self.data = df
        print("Time variables created")
        
        return df
    
    def seasonal_adjustment(self):
        """Deseason and detrend hourly emissions and generation using R-style regression with interaction terms"""

        df = self.data.copy()

        # ---- Emissions model ----
        # Equivalent to R:
        # lm(hourly_emissions ~ factorinterval*factormonth + factormonth*factoryear + factordow + trend)
        model_em = ols(
            'hourly_emissions ~ C(factordow, Treatment(reference="Monday")) + factorinterval * factormonth + factormonth * factoryear + trend',
            data=df
        ).fit()

        # Deseasoned & detrended emissions
        df['hourly_emissions_res'] = model_em.resid + model_em.params['Intercept']

        # ---- Generation model ----
        # Equivalent to R:
        # lm(hourly_generation ~ factorinterval*factormonth + factormonth*factoryear + factordow + trend)
        model_gen = ols(
            'hourly_generation ~ C(factordow, Treatment(reference="Monday")) + factorinterval * factormonth + factormonth * factoryear + trend',
            data=df
        ).fit()

        # Deseasoned & detrended generation
        df['hourly_generation_res'] = model_gen.resid + model_gen.params['Intercept']

        # ---- Save results ----
        self.data = df

        print("\n✅ Seasonal & Trend Adjustment Completed")
        print(f"Emissions R²:  {model_em.rsquared:.4f}")
        print(f"Generation R²: {model_gen.rsquared:.4f}")
        print(f"Emissions Coefficients: {len(model_em.params)} terms")
        print(f"Generation Coefficients: {len(model_gen.params)} terms")

        return df
        
    def summary_statistics_by_season(self):
        """Calculate summary statistics by season"""
        df = self.data
        
        # Overall statistics
        overall = pd.DataFrame({
            'Variable': ['Emissions', 'Generation'],
            'N': [len(df), len(df)],
            'Mean': [df['hourly_emissions'].mean(), df['hourly_generation'].mean()],
            'Std Dev': [df['hourly_emissions'].std(), df['hourly_generation'].std()],
            'Min': [df['hourly_emissions'].min(), df['hourly_generation'].min()],
            'Max': [df['hourly_emissions'].max(), df['hourly_generation'].max()]
        })
        
        print("\nOverall Summary Statistics:")
        print(overall.to_string(index=False))
        
        # By season
        seasons = df.groupby('yearseason').agg({
            'hourly_emissions': ['count', 'mean', 'std', 'min', 'max'],
            'hourly_generation': ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        print("\nSummary Statistics by Season:")
        print(seasons)
        
        return overall, seasons
    
    def plot_time_series(self):
        """Plot emissions and generation time series"""
        
        df = self.data
        self.set_publication_style()
        fig, axes = plt.subplots(2, 1, figsize=self.get_figsize(89*2))
       
        # Emissions
        axes[0].plot(df['date_hour'], df['hourly_emissions'], 
                    linewidth=0.3, alpha=0.7, color='darkred')
        axes[0].set_title('Hourly Emissions (2019-2022)', 
                         fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Emissions (metric tons CO₂)', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # Generation
        axes[1].plot(df['date_hour'], df['hourly_generation'],
                    linewidth=0.3, alpha=0.7, color='darkblue')
        axes[1].set_title('Hourly Generation (2019-2022)', 
                         fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Generation (MWh)', fontsize=13)
        axes[1].set_xlabel('Date', fontsize=13)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Deseasonalized
        fig, axes = plt.subplots(2, 1, figsize=self.get_figsize(89*2))
       
      
        axes[0].plot(df['date_hour'], df['hourly_emissions_res'],
                    linewidth=0.3, alpha=0.7, color='darkred')
        axes[0].set_title('Hourly Emissions - Deseasoned & Detrended',
                         fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Residual Emissions', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df['date_hour'], df['hourly_generation_res'],
                    linewidth=0.3, alpha=0.7, color='darkblue')
        axes[1].set_title('Hourly Generation - Deseasoned & Detrended',
                         fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Residual Generation', fontsize=13)
        axes[1].set_xlabel('Date', fontsize=13)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_annual_diagnostics(self, lags=100, save_path=None):
        
        """
        Create elegant, publication-quality annual diagnostic plots
        (deseasoned emissions and generation + ACF/PACF).

        Parameters
        ----------
        lags : int, optional
            Number of lags for ACF/PACF (default=100)
        save_path : str, optional
            Directory to save figures. If None, figures are only shown.
        """

        if self.data is None:
            raise ValueError("No data loaded. Run load_and_clean_data() first.")

        # Apply publication style
        self.set_publication_style()

        df = self.data.copy()
        df['date_hourx'] = pd.to_datetime(df['date_hour'], utc=True, errors='coerce')

        years = sorted(df['year'].unique())
        print(f"🧭 Found {len(years)} years: {years}")

        for year in years:
            print(f"\n📅 Processing {year}...")
            df_year = df[df['year'] == year].copy()

            
            # --- Helper: Elegant time series plot ---
            def elegant_plot(x, y, title, ylabel, color="#1a1a1a", lw=0.7):
                fig, ax = plt.subplots(figsize=self.get_figsize(89*2))
                ax.plot(x, y, color=color, lw=lw)
                ax.set_title(title, loc='left', pad=12, weight='bold')
                ax.set_ylabel(ylabel, labelpad=8)
                ax.set_xlabel("")
                ax.grid(True, alpha=0.3)
                ax.spines[['top', 'right']].set_visible(False)
                fig.tight_layout()
                if save_path:
                    fname = f"{save_path}/{title.replace(' ', '_').replace(':', '')}.png"
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                plt.show()
                return fig

            # --- Plot generation residuals ---
            elegant_plot(
                df_year['date_hourx'], df_year['hourly_generation_res'],
                f"Hourly Load – Deseasoned & Detrended ({year})",
                "Hourly load (MWh)"
            )

            # --- ACF/PACF for generation ---
            fig, ax = plt.subplots(2, 1, figsize=self.get_figsize(89*2))
            plot_acf(df_year['hourly_generation_res'].dropna(),
                    lags=lags, ax=ax[0], color="#1f77b4", zero=False)
            plot_pacf(df_year['hourly_generation_res'].dropna(),
                    lags=lags, ax=ax[1], color="#ff7f0e", zero=False, method="ywm")

            for i, title in enumerate(["ACF – Hourly Load", "PACF – Hourly Load"]):
                ax[i].set_title(f"{title} ({year})", loc='left', pad=10, weight='bold')
                ax[i].set_ylim(-1.05, 1.05)  # ensures full bars visible
                ax[i].spines[['top', 'right']].set_visible(False)
                ax[i].grid(True, alpha=0.3)

            fig.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}/ACF_PACF_Load_{year}.png", dpi=300, bbox_inches='tight')
            plt.show()

            # --- Plot emissions residuals ---
            elegant_plot(
                df_year['date_hourx'], df_year['hourly_emissions_res'],
                f"Hourly Emissions – Deseasoned & Detrended ({year})",
                "Hourly emissions (metric tons CO₂)"
            )

            # --- ACF/PACF for emissions ---
            fig, ax = plt.subplots(2, 1, figsize=self.get_figsize(89*2))
            plot_acf(df_year['hourly_emissions_res'].dropna(),
                    lags=lags, ax=ax[0], color="#1f77b4", zero=False)
            plot_pacf(df_year['hourly_emissions_res'].dropna(),
                    lags=lags, ax=ax[1], color="#ff7f0e", zero=False, method="ywm")

            for i, title in enumerate(["ACF – Hourly Emissions", "PACF – Hourly Emissions"]):
                ax[i].set_title(f"{title} ({year})", loc='left', pad=10, weight='bold')
                ax[i].set_ylim(-1.05, 1.05)  # ensures full bars visible
                ax[i].spines[['top', 'right']].set_visible(False)
                ax[i].grid(True, alpha=0.3)

            fig.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}/ACF_PACF_Emissions_{year}.png", dpi=300, bbox_inches='tight')
            plt.show()
    def plot_seasonal_diagnostics(self, lags=100, save_path=None):
        """
        Create publication-quality seasonal diagnostic plots
        (deseasoned emissions and generation + ACF/PACF) by season.
        """

        if self.data is None:
            raise ValueError("No data loaded. Run load_and_clean_data() first.")

        self.set_publication_style()

        df = self.data.copy()
        df['date_hourx'] = pd.to_datetime(df['date_hour'], utc=True, errors='coerce')
        seasons = sorted(df['yearseason'].dropna().unique())

        print(f"🍂 Found {len(seasons)} seasons: {seasons}")

        # Helper: time series plot
        def elegant_plot(x, y, title, ylabel, color="#1a1a1a", lw=0.7):
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(x, y, color=color, lw=lw)
            ax.set_title(title, loc='left', pad=12, weight='bold')
            ax.set_ylabel(ylabel, labelpad=8)
            ax.set_xlabel("")
            ax.grid(True, alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            if save_path:
                fname = f"{save_path}/{title.replace(' ', '_').replace(':', '')}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.show()
            return fig

        # Loop through each season
        for season in seasons:
            print(f"\n📅 Processing season: {season}")
            df_season = df[df['yearseason'] == season].copy()

            # --- Plot generation residuals ---
            elegant_plot(
                df_season['date_hourx'], df_season['hourly_generation_res'],
                f"Hourly Load – Deseasoned & Detrended ({season})",
                "Hourly load (MWh)"
            )

            # --- ACF/PACF for generation ---
            fig, ax = plt.subplots(2, 1, figsize=self.get_figsize(89*2))
            plot_acf(df_season['hourly_generation_res'].dropna(),
                    lags=lags, ax=ax[0], color="#1f77b4", zero=False)
            plot_pacf(df_season['hourly_generation_res'].dropna(),
                    lags=lags, ax=ax[1], color="#ff7f0e", zero=False, method="ywm")

            # Custom thresholds
            ax[0].set_ylim(-0.4, 1.05)   # ✅ ACF from -0.1
            ax[1].set_ylim(-0.5, 1.05)  # ✅ PACF from -0.35

            for i, title in enumerate(["ACF – Hourly Load", "PACF – Hourly Load"]):
                ax[i].set_title(f"{title} ({season})", loc='left', pad=10, weight='bold')
                ax[i].spines[['top', 'right']].set_visible(False)
                ax[i].grid(True, alpha=0.3)

            fig.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}/ACF_PACF_Load_{season}.png", dpi=300, bbox_inches='tight')
            plt.show()

            # --- Plot emissions residuals ---
            elegant_plot(
                df_season['date_hourx'], df_season['hourly_emissions_res'],
                f"Hourly Emissions – Deseasoned & Detrended ({season})",
                "Hourly emissions (metric tons CO₂)"
            )

            # --- ACF/PACF for emissions ---
            fig, ax = plt.subplots(2, 1, figsize=self.get_figsize(89*2))
            plot_acf(df_season['hourly_emissions_res'].dropna(),
                    lags=lags, ax=ax[0], color="#1f77b4", zero=False)
            plot_pacf(df_season['hourly_emissions_res'].dropna(),
                    lags=lags, ax=ax[1], color="#ff7f0e", zero=False, method="ywm")

            # Custom thresholds again
            ax[0].set_ylim(-0.1, 1.05)
            ax[1].set_ylim(-0.35, 1.05)

            for i, title in enumerate(["ACF – Hourly Emissions", "PACF – Hourly Emissions"]):
                ax[i].set_title(f"{title} ({season})", loc='left', pad=10, weight='bold')
                ax[i].spines[['top', 'right']].set_visible(False)
                ax[i].grid(True, alpha=0.3)

            fig.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}/ACF_PACF_Emissions_{season}.png", dpi=300, bbox_inches='tight')
            plt.show()

        print("\n✅ Publication-ready seasonal diagnostic plots generated successfully.")
            
    def plot_lag_nonlinearity(
            self, variable="hourly_emissions_res", lag=1, year=None,
            frac=0.2, save_path=None
        ):
            """
            Plot x_t vs x_{t-lag} with LOESS smoothing to visualize nonlinearity.
            Optimized for one-column journal figures (Tsay, 2019 style).
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from statsmodels.nonparametric.smoothers_lowess import lowess

            if self.data is None:
                raise ValueError("No data loaded. Run load_and_clean_data() first.")
            if variable not in self.data.columns:
                raise ValueError(f"Column '{variable}' not found. Did you run seasonal_adjustment()?")

            # --- Data selection ---
            df_plot = self.data.copy()
            if year:
                df_plot = df_plot[df_plot["year"] == year]
                time_label = str(year)
            else:
                time_label = "Full Sample"

            y_t = df_plot[variable]
            x_t_lag = df_plot[variable].shift(lag)
            mask = ~np.isnan(x_t_lag) & ~np.isnan(y_t)
            x_vals, y_vals = x_t_lag[mask], y_t[mask]

            print(f"⏳ Calculating LOESS smoothing (n={len(x_vals)}, lag={lag})...")
            z = lowess(y_vals, x_vals, frac=frac, it=0)

            # --- Figure settings for one-column layout ---
            plt.style.use("seaborn-v0_8-white")
            fig, ax = plt.subplots(figsize=self.get_figsize(89*2)) # ~89 mm × 66 mm

            # --- Scatter plot ---
            ax.scatter(
                x_vals, y_vals, alpha=0.15, s=3.5,
                color="#2c3e50", label="Observations", edgecolor="none"
            )

            # --- LOESS line ---
            ax.plot(
                z[:, 0], z[:, 1],
                color="#e74c3c", linewidth=0.9,
                label=f"LOESS (frac={frac})"
            )

            # --- Styling ---
            ax.set_title(
                f"Nonlinearity Check ({time_label})", loc="left", fontsize=8, fontweight="bold"
            )
            ax.set_xlabel(f"$x_{{t-{lag}}}$", fontsize=7)
            ax.set_ylabel(f"$x_t$", fontsize=7)

            # Reference lines
            ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
            ax.axvline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)

            # Axis limits with small margins
            x_margin = 0.05 * (x_vals.max() - x_vals.min())
            y_margin = 0.05 * (y_vals.max() - y_vals.min())
            ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)
            ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)

            # Compact ticks and font sizes
            ax.tick_params(axis="both", labelsize=6, width=0.3, pad=2)
            ax.legend(frameon=False, fontsize=6, loc="best", handlelength=1.5)

            plt.tight_layout(pad=0.5)

            # --- Save or show ---
            if save_path:
                filename = f"{save_path}/Lag{lag}_{variable}_{year}.pdf"
                plt.savefig(filename, bbox_inches="tight", dpi=600)
                print(f"💾 Figure saved to: {filename}")
            else:
                plt.show()

            
    def run_annual_nonlinearity_tests_r(self, y_res_col="hourly_emissions_res", save_path=None):
        """
        Perform annual stationarity and nonlinearity tests using R packages
        (urca, NTS, fNonlinear) via rpy2.
        Reproduces the R workflow shown in Tsay (2019) and your R script.
        """

        import pandas as pd
        import numpy as np
        import warnings
        import re
        from scipy.stats import f as f_dist

        from rpy2.robjects import r, globalenv, FloatVector
        from rpy2.robjects.packages import importr

        warnings.filterwarnings("ignore")

        # --- Import R packages ---
        urca = importr("urca")
        NTS = importr("NTS")
        fNonlinear = importr("fNonlinear")

        # --- Data checks ---
        if self.data is None:
            raise ValueError("No data loaded. Please load your dataset first.")
        if y_res_col not in self.data.columns:
            raise ValueError(f"Column '{y_res_col}' not found. Run deseasoning first.")

        df = self.data.copy()
        years = sorted(df["year"].unique())
        print(f"🧭 Running R-based tests for {len(years)} years: {years}")

        results_all = []

        for year in years:
            print(f"📅 Processing {year}...")
            y = df.loc[df["year"] == year, y_res_col].dropna().to_numpy()

            if len(y) < 30:
                print(f"⚠️  Skipping {year}: too few data points ({len(y)})")
                continue

            if np.allclose(np.std(y), 0):
                print(f"⚠️  Skipping {year}: near-constant series")
                continue

            # Send Python vector to R
            globalenv["y"] = FloatVector(y)

            # === 1️⃣ ADF Test ===
            adf = urca.ur_df(FloatVector(y), type="none", selectlags="BIC")
            ADF_stat = float(adf.do_slot("teststat")[0])
            ADF_cval = float(adf.do_slot("cval")[1])

            # === 2️⃣ DF-GLS Test ===
            lag_max = int(adf.do_slot("lags")[0])
            dfgls = urca.ur_ers(FloatVector(y), lag_max=lag_max, model="constant")
            DFGLS_stat = float(dfgls.do_slot("teststat")[0])
            DFGLS_cval = float(dfgls.do_slot("cval")[1])

            # === 3️⃣ PP Test ===
            pp = urca.ur_pp(FloatVector(y), type="Z-tau", model="constant", lags="long")
            PP_stat = float(pp.do_slot("teststat")[0])
            PP_cval = float(pp.do_slot("cval")[1])

            # === 4️⃣ KPSS Test ===
            kpss = urca.ur_kpss(FloatVector(y), type="mu", lags="short")
            KPSS_stat = float(kpss.do_slot("teststat")[0])
            KPSS_cval = float(kpss.do_slot("cval")[1])

            # === 5️⃣ Tsay F-test (NTS) ===
            try:
                f_test = NTS.F_test(FloatVector(y), int(pp.do_slot("lag")[0]), thres=float(np.mean(y)))
                F_stat = float(f_test.rx2("test.stat")[0])
                F_pvalue = float(f_test.rx2("p.value")[0])
            except Exception as e:
                print(f"⚠️  Tsay F-test failed ({year}): {e}")
                F_stat, F_pvalue = np.nan, np.nan

            # === 6️⃣ Threshold Nonlinearity Test (TNT) ===
            try:
                yN = int(0.1 * len(y))
                tnt = NTS.thr_test(FloatVector(y), p=int(pp.do_slot("lag")[0]), d=1, ini=yN, include_mean=False)
                TNT_stat = float(tnt[0][0])
                DF1, DF2 = float(tnt[1][0]), float(tnt[1][1])
                TNT_pvalue = 1 - f_dist.cdf(TNT_stat, DF1, DF2)
            except Exception as e:
                print(f"⚠️  TNT test failed ({year}): {e}")
                TNT_stat, TNT_pvalue = np.nan, np.nan

            # === 7️⃣ BDS Test (fNonlinear) ===
            try:
                bds_test = fNonlinear.bdsTest(FloatVector(y), m=6, eps=2 * np.std(y))
                bds_tab = bds_test.do_slot("test")
                bds_stat = float(bds_tab.rx2("statistic")[0])
                bds_pvalue = float(bds_tab.rx2("p.value")[0])
            except Exception as e:
                print(f"⚠️  BDS test failed ({year}): {e}")
                bds_stat, bds_pvalue = np.nan, np.nan

            # === 8️⃣ Peña–Rodríguez Test (PRnd) ===
            try:
                r('suppressMessages(library(NTS))')
                # Run the PRnd R function and capture output
                pr_output = r(f'capture.output(print(PRnd(abs(y), m={int(pp.do_slot("lag")[0])})))')
                pr_line = pr_output[1]
                tokens = re.split(r'\\s+', pr_line.strip())
                PR_stat = float(tokens[-2])
                PR_pvalue = float(tokens[-1])
            except Exception as e:
                print(f"⚠️  PRnd test failed ({year}): {e}")
                PR_stat, PR_pvalue = np.nan, np.nan

            # --- Collect results ---
            results_all.append({
                "year": year,
                "ADF_stat": ADF_stat, "ADF_cval": ADF_cval,
                "DFGLS_stat": DFGLS_stat, "DFGLS_cval": DFGLS_cval,
                "PP_stat": PP_stat, "PP_cval": PP_cval,
                "KPSS_stat": KPSS_stat, "KPSS_cval": KPSS_cval,
                "F_stat": F_stat, "F_pvalue": F_pvalue,
                "TNT_stat": TNT_stat, "TNT_pvalue": TNT_pvalue,
                "BDS_stat": bds_stat, "BDS_pvalue": bds_pvalue,
                "PR_stat": PR_stat, "PR_pvalue": PR_pvalue
            })

        # --- Combine results ---
        df_out = pd.DataFrame(results_all)
        print("\n✅ Completed all R-based tests.")

        # --- Save results ---
        if save_path:
            file_name = f"{save_path}/annual_{y_res_col}_nonlinearity_R.xlsx"
            df_out.to_excel(file_name, index=False)
            print(f"💾 Saved results to: {file_name}")

        return df_out

        
    def mef_by_year(self, include_markov=True):
        """Estimate MEF by year using multiple methods"""
        df = self.data
        years = sorted(df['year'].unique())
        
        results = []
        
        for year in years:
            print(f"\nProcessing year {year}...")
            year_data = df[df['year'] == year].copy()
            
            # Seasonal adjustment for this year
            X = pd.get_dummies(year_data[['factorinterval', 'factormonth', 'factordow']], 
                              drop_first=True)
            X['trend'] = np.arange(len(year_data))
            X = add_constant(X)
            
            # Emissions
            em_model = OLS(year_data['hourly_emissions'], X).fit()
            year_data['em_res'] = em_model.resid + em_model.params['const']
            
            # Generation  
            gen_model = OLS(year_data['hourly_generation'], X).fit()
            year_data['gen_res'] = gen_model.resid + gen_model.params['const']
            
            # OLS MEF
            X_mef = add_constant(year_data['gen_res'])
            ols_model = OLS(year_data['em_res'], X_mef).fit()
            cov = cov_hac(ols_model, nlags=48)
            
            ols_mef = ols_model.params['gen_res']
            ols_se = np.sqrt(cov[1, 1])
            
            # Differences MEF
            em_diff = year_data['em_res'].diff().dropna()
            gen_diff = year_data['gen_res'].diff().dropna()
            X_diff = add_constant(gen_diff)
            diff_model = OLS(em_diff, X_diff).fit()
            
            diff_mef = diff_model.params[gen_diff.name]
            diff_se = diff_model.bse[gen_diff.name]
            
            # ARIMA MEF
            try:
                arima_model = ARIMA(year_data['em_res'], 
                                   exog=year_data['gen_res'],
                                   order=(1, 0, 1)).fit()
                arima_mef = arima_model.params[-1]
                arima_se = arima_model.bse[-1]
            except:
                arima_mef = np.nan
                arima_se = np.nan
            
            # Markov Switching MEF
            ms_high_mef = ms_high_se = ms_low_mef = ms_low_se = np.nan
            p11 = p22 = p12 = p21 = dur_high = dur_low = np.nan
            
            if include_markov:
                try:
                    print(f"  Estimating Markov Switching model for {year}...")
                    endog = year_data['em_res'].values
                    exog_ms = add_constant(year_data['gen_res'].values)
                    
                    ms_model = MarkovRegression(
                        endog=endog,
                        k_regimes=2,
                        exog=exog_ms,
                        switching_variance=False
                    )
                    
                    ms_results = ms_model.fit(maxiter=500, disp=False)
                    
                    # Extract regime coefficients
                    regime_mefs = [ms_results.params[f'gen_res[{i}]'] for i in range(2)]
                    regime_ses = [ms_results.bse[f'gen_res[{i}]'] for i in range(2)]
                    
                    high_idx = np.argmax(regime_mefs)
                    low_idx = np.argmin(regime_mefs)
                    
                    ms_high_mef = regime_mefs[high_idx]
                    ms_high_se = regime_ses[high_idx]
                    ms_low_mef = regime_mefs[low_idx]
                    ms_low_se = regime_ses[low_idx]
                    
                    # Transition probabilities
                    trans_mat = ms_results.transition_matrix
                    p11 = trans_mat[high_idx, high_idx]
                    p22 = trans_mat[low_idx, low_idx]
                    p12 = trans_mat[high_idx, low_idx]
                    p21 = trans_mat[low_idx, high_idx]
                    
                    dur_high = 1 / (1 - p11) if p11 < 1 else np.inf
                    dur_low = 1 / (1 - p22) if p22 < 1 else np.inf
                    
                    print(f"  MS Model converged successfully")
                    
                except Exception as e:
                    print(f"  MS estimation failed: {str(e)[:50]}...")
            
            # Average emissions
            avg_em = year_data['hourly_emissions'].sum() / year_data['hourly_generation'].sum()
            
            results.append({
                'Year': year,
                'OLS_MEF': ols_mef,
                'OLS_SE': ols_se,
                'Diff_MEF': diff_mef,
                'Diff_SE': diff_se,
                'ARIMA_MEF': arima_mef,
                'ARIMA_SE': arima_se,
                'MS_High_MEF': ms_high_mef,
                'MS_High_SE': ms_high_se,
                'MS_Low_MEF': ms_low_mef,
                'MS_Low_SE': ms_low_se,
                'P11': p11,
                'P22': p22,
                'P12': p12,
                'P21': p21,
                'Dur_High': dur_high,
                'Dur_Low': dur_low,
                'Avg_Emissions': avg_em
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*100)
        print("MEF ESTIMATES BY YEAR")
        print("="*100)
        print(results_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print("="*100)
        
        # Visualization
        self._plot_mef_by_year(results_df, include_markov)
        
        return results_df
    
    def _plot_mef_by_year(self, results_df, include_markov=True):
        """Plot MEF estimates by year"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        years = results_df['Year']
        
        # OLS with CI
        ax.errorbar(years, results_df['OLS_MEF'], 
                   yerr=1.96*results_df['OLS_SE'],
                   fmt='o-', label='OLS', markersize=10, capsize=5, capthick=2,
                   linewidth=2.5, color='orange')
        
        # Differences with CI
        ax.errorbar(years + 0.05, results_df['Diff_MEF'],
                   yerr=1.96*results_df['Diff_SE'],
                   fmt='s-', label='Differences', markersize=8, capsize=5, capthick=2,
                   linewidth=2, color='gray', alpha=0.8)
        
        # ARIMA with CI
        ax.errorbar(years + 0.1, results_df['ARIMA_MEF'],
                   yerr=1.96*results_df['ARIMA_SE'],
                   fmt='^-', label='ARIMA', markersize=8, capsize=5, capthick=2,
                   linewidth=2, color='red', alpha=0.8)
        
        # Markov Switching
        if include_markov and 'MS_High_MEF' in results_df.columns:
            ax.errorbar(years + 0.15, results_df['MS_High_MEF'],
                       yerr=1.96*results_df['MS_High_SE'],
                       fmt='d-', label='MS-High', markersize=8, capsize=5, capthick=2,
                       linewidth=2, color='blue', alpha=0.8)
            
            ax.errorbar(years + 0.2, results_df['MS_Low_MEF'],
                       yerr=1.96*results_df['MS_Low_SE'],
                       fmt='v-', label='MS-Low', markersize=8, capsize=5, capthick=2,
                       linewidth=2, color='purple', alpha=0.8)
        
        # Average emissions
        ax.plot(years, results_df['Avg_Emissions'], 
               'D-', label='Average Emissions', markersize=9, 
               linewidth=2.5, color='pink')
        
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('MEF (kg CO₂/kWh)', fontsize=14, fontweight='bold')
        ax.set_title('US MEF Estimates by Year and Method', 
                    fontsize=17, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(years)
        
        plt.tight_layout()
        plt.show()
        
        # Plot transition probabilities if available
        if include_markov and 'P11' in results_df.columns:
            self._plot_transition_probs(results_df)
    
    def mef_by_season(self):
        """Estimate MEF by season"""
        df = self.data
        seasons = sorted(df['yearseason'].unique())
        
        results = []
        
        for season in seasons:
            season_data = df[df['yearseason'] == season].copy()
            
            # Skip if too few observations
            if len(season_data) < 100:
                continue
            
            # Seasonal adjustment
            X = pd.get_dummies(season_data[['factorinterval', 'factormonth', 'factordow']], 
                              drop_first=True)
            X['trend'] = np.arange(len(season_data))
            X = add_constant(X)
            
            em_model = OLS(season_data['hourly_emissions'], X).fit()
            season_data['em_res'] = em_model.resid + em_model.params['const']
            
            gen_model = OLS(season_data['hourly_generation'], X).fit()
            season_data['gen_res'] = gen_model.resid + gen_model.params['const']
            
            # OLS MEF
            X_mef = add_constant(season_data['gen_res'])
            ols_model = OLS(season_data['em_res'], X_mef).fit()
            
            ols_mef = ols_model.params['gen_res']
            ols_se = ols_model.bse['gen_res']
            
            # Average emissions
            avg_em = season_data['hourly_emissions'].sum() / season_data['hourly_generation'].sum()
            
            results.append({
                'Season': season,
                'OLS_MEF': ols_mef,
                'OLS_SE': ols_se,
                'Avg_Emissions': avg_em,
                'N': len(season_data)
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("MEF ESTIMATES BY SEASON")
        print("="*80)
        print(results_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print("="*80)
        
        return results_df


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("US MEF Analysis Module with Markov Switching")
    print("="*80)
    print("\nExample workflow:")
    print("""
    # 1. Initialize and load data
    analysis = USMEFAnalysis("Region_US48_Hourly_NEW.csv")
    data = analysis.load_and_clean_data(start_date='2019-01-01', end_date='2022-12-31')
    
    # 2. Create time variables
    analysis.create_time_variables()
    
    # 3. Seasonal adjustment
    analysis.seasonal_adjustment()
    
    # 4. Summary statistics
    overall_stats, seasonal_stats = analysis.summary_statistics_by_season()
    
    # 5. Visualizations
    analysis.plot_time_series()
    
    # 6. Estimate MEF by year (includes Markov Switching)
    mef_yearly = analysis.mef_by_year(include_markov=True)
    
    # 7. Estimate MEF by season
    mef_seasonal = analysis.mef_by_season()
    
    # Access specific results
    print(f"2019 OLS MEF: {mef_yearly.loc[0, 'OLS_MEF']:.6f}")
    print(f"2019 MS High MEF: {mef_yearly.loc[0, 'MS_High_MEF']:.6f}")
    print(f"2019 MS Low MEF: {mef_yearly.loc[0, 'MS_Low_MEF']:.6f}")
    """)
    print("\nKey Features:")
    print("  ✓ Markov Switching with 2 regimes")
    print("  ✓ Transition probabilities and expected durations")
    print("  ✓ High/Low emission state identification")
    print("  ✓ Comprehensive error handling")
    print("  ✓ Visual comparison of all methods")
    print("="*80)