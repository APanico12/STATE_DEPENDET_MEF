"""
US MEF (Marginal Emission Factor) Analysis
Database: Region_US48_Hourly_NEW
Spatial Dimension: US (Lower 48)
Covered Years: 2019-2022
Approach: Intra-day and inter-day
"""

import re
from unittest import result
from Cython import p_const_ptrdiff_t
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
import statsmodels.api as sm
from patsy import dmatrices
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
 
    def load_and_clean_data(self, start_date='2019-01-01', end_date='2025-12-31'):
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
            'CO2 Emissions Intensity for Generated Electricity': 'CO2EmissionsIntensity',
            'RNG':'hourly_generation_renewables',
            'NRNG':'hourly_generation_nonrenewables'
        }
        df = df.rename(columns=rename_map)

        # --- Convert units ---
        df['hourly_emissions_mlb'] = df['hourly_emissions'] * 2204.6 / 1_000_000  # metric tons → million lbs
        df['hourly_generation_mkwh'] = df['hourly_generation'] * 1000 / 1_000_000  # GWh → million kWh
        df['hourly_generation_renewables_mkwh'] = df['hourly_generation_renewables'] * 1000 / 1_000_000  # GWh → million kWh
        df['hourly_generation_nonrenewables_mkwh'] = df['hourly_generation_nonrenewables'] * 1000 / 1_000_000  # GWh → million kWh
        # --- Select final columns  ---
        keep_cols = [
            'bid_offer_date', 'interval', 'date_hour', 
            'D', 'NG','Gen_Gas','Gen_Coal',
            'hourly_generation','hourly_generation_renewables','hourly_generation_nonrenewables', 'hourly_emissions', 
            'CO2EmissionsIntensity', 
            'hourly_emissions_mlb', 'hourly_generation_mkwh','hourly_generation_renewables_mkwh','hourly_generation_nonrenewables_mkwh'
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
        #print(f'month : {df["month"]}, year : {df["year"]}, day : {df["dom"]}')
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
    
    def seasonal_adjustment_iteraction(self):

        df = self.data.copy()

        # ---- Emissions model ----
        # Equivalent to R:
        # lm(hourly_emissions ~ factorinterval*factormonth + factormonth*factoryear + factordow + trend)
        model_em = ols(
            'hourly_emissions_mlb ~ C(factordow, Treatment(reference="Monday")) + C(factorinterval)*C(factormonth) + C(factormonth)*C(factoryear) + trend',
            data=df
        ).fit()
        print(model_em.summary())
        # Deseasoned & detrended emissions
        df['hourly_emissions_res'] = model_em.resid + model_em.params['Intercept']

        # ---- Generation model ----
        # Equivalent to R:
        # lm(hourly_generation ~ factorinterval*factormonth + factormonth*factoryear + factordow + trend)
        model_gen = ols(
            'hourly_generation_mkwh ~ C(factordow, Treatment(reference="Monday")) + C(factorinterval)*C(factormonth) + C(factormonth)*C(factoryear) + trend',
            data=df
        ).fit()
        print(model_gen.summary())
        # Deseasoned & detrended generation
        df['hourly_generation_res'] = model_gen.resid + model_gen.params['Intercept']
        
        #same for hourly_generation_renewables
        model_gen_r = ols(
            'hourly_generation_renewables_mkwh ~  C(factordow, Treatment(reference="Monday")) + C(factorinterval)*C(factormonth) + C(factormonth)*C(factoryear) + trend',
            data=df
        ).fit()
        print(model_gen_r.summary())
        # Deseasoned & detrended renewable generation
        df['hourly_generation_renewables_res'] = model_gen_r.resid + model_gen_r.params['Intercept']
        
        #same for hourly_generation_nonrenewables
        model_gen_nr = ols(
            'hourly_generation_nonrenewables_mkwh ~ C(factordow, Treatment(reference="Monday")) + C(factorinterval)*C(factormonth) + C(factormonth)*C(factoryear) + trend',
            data=df
        ).fit()
        # Deseasoned & detrended non-renewable generation
        df['hourly_generation_nonrenewables_res'] = model_gen_nr.resid + model_gen_nr.params['Intercept']
        print(model_gen_nr.summary())
        
        # ---- Save results ----
        self.data = df

        print("\n✅ Seasonal & Trend Adjustment Completed")
        print(f"Emissions R²:  {model_em.rsquared:.4f}")
        print(f"Generation R²: {model_gen.rsquared:.4f}")
        print(f"Emissions Coefficients: {len(model_em.params)} terms")
        print(f"Generation Coefficients: {len(model_gen.params)} terms")
        print(f"Renewable Generation Coefficients: {len(model_gen_r.params)} terms")
        print(f"Non-Renewable Generation Coefficients: {len(model_gen_nr.params)} terms")

        return df
    
    def seasonal_adjustment(self):
   
        df = self.data.copy()

        # ---- Emissions model ----
        # Equivalent to R:
        # lm(hourly_emissions ~ factorinterval*factormonth + factormonth*factoryear + factordow + trend)
        model_em = ols(
            'hourly_emissions_mlb ~ C(factordow, Treatment(reference="Monday")) + C(factorinterval) + C(factormonth) + C(factoryear)',
            data=df
        ).fit()
        print(model_em.summary())
        # Deseasoned & detrended emissions
        df['hourly_emissions_res'] = model_em.resid + model_em.params['Intercept']

        # ---- Generation model ----
        # Equivalent to R:
        # lm(hourly_generation ~ factorinterval*factormonth + factormonth*factoryear + factordow + trend)
        model_gen = ols(
            'hourly_generation_mkwh ~ C(factordow, Treatment(reference="Monday")) + C(factorinterval) + C(factormonth) + C(factoryear) ',
            data=df
        ).fit()
        print(model_gen.summary())
        # Deseasoned & detrended generation
        df['hourly_generation_res'] = model_gen.resid + model_gen.params['Intercept']
        
        #same for hourly_generation_renewables
        model_gen_r = ols(
            'hourly_generation_renewables_mkwh ~ C(factordow, Treatment(reference="Monday")) + C(factorinterval) + C(factormonth) + C(factoryear)',
            data=df
        ).fit()
        print(model_gen_r.summary())
        # Deseasoned & detrended renewable generation
        df['hourly_generation_renewables_res'] = model_gen_r.resid + model_gen_r.params['Intercept']
        
        #same for hourly_generation_nonrenewables
        model_gen_nr = ols(
            'hourly_generation_nonrenewables_mkwh ~ C(factordow, Treatment(reference="Monday")) + C(factorinterval) + C(factormonth) + C(factoryear)',
            data=df
        ).fit()
        # Deseasoned & detrended non-renewable generation
        df['hourly_generation_nonrenewables_res'] = model_gen_nr.resid + model_gen_nr.params['Intercept']
        print(model_gen_nr.summary())
        # ---- Save results ----
        self.data = df

        print("\n✅ Seasonal & Trend Adjustment Completed")
        print(f"Emissions R²:  {model_em.rsquared:.4f}")
        print(f"Generation R²: {model_gen.rsquared:.4f}")
        print(f"Emissions Coefficients: {len(model_em.params)} terms")
        print(f"Generation Coefficients: {len(model_gen.params)} terms")
        print(f"Renewable Generation Coefficients: {len(model_gen_r.params)} terms")
        print(f"Non-Renewable Generation Coefficients: {len(model_gen_nr.params)} terms")

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
        fig, axes = plt.subplots(2, 1, figsize=(18,12))
       
        # Emissions
        axes[0].plot(df['date_hour'], df['hourly_emissions_mlb'], 
                    linewidth=0.3, alpha=0.7, color='darkred')
        axes[0].set_title('Hourly Emissions', 
                         fontsize=18)
        axes[0].set_ylabel(r'Emissions (Mlbs $CO_2$)')
        
        
        # Generation
        axes[1].plot(df['date_hour'], df['hourly_generation_mkwh'],
                    linewidth=0.3, alpha=0.7, color='darkblue')
        axes[1].set_title('Hourly Generation ')
        axes[1].set_ylabel('Generation (MWh)')
        axes[1].set_xlabel('Date')
        plt.tight_layout()
        plt.show()
        
    def elegant_plot(self, x, y, title, ylabel, color="#1a1a1a", lw=0.7, save_path=None):
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
    
    def plot_annual_diagnostics_dynamic(self, lags=100, save_path='images'):
        """
        Dynamic version: Recalculates residuals each year.
        Generates 3 separate SQUARE plots per variable: Residuals, ACF, and PACF.
        """
        import matplotlib.dates as mdates
        if self.data is None:
            raise ValueError("No data loaded. Run load_and_clean_data() first.")

        self.set_publication_style()

        df = self.data.copy()
        df['date_hourx'] = pd.to_datetime(df['date_hour'], utc=True, errors='coerce')
        years = sorted(df['year'].unique())
        print(f"🧭 Found {len(years)} years: {years}")

        # Define Square Size (e.g., 6x6 inches)
        sq_size = (6, 6)

        for year in years:
            print(f"\n📅 Processing {year}...")
            df_year = df[df['year'] == year].copy()

            # 1. Recalculate residuals
            model_gen = ols("hourly_generation ~ factorinterval * factormonth + factordow + trend", data=df_year).fit()
            df_year['hourly_generation_res'] = model_gen.resid + model_gen.params[0]

            model_em = ols("hourly_emissions ~ factorinterval * factormonth + factordow + trend", data=df_year).fit()
            df_year['hourly_emissions_res'] = model_em.resid + model_em.params[0]

            # ==========================================
            # VARIABLE 1: GENERATION
            # ==========================================
            
            # Plot 1: Residuals Time Series
            fig1, ax1 = plt.subplots(figsize=sq_size)
            ax1.plot(df_year['date_hourx'], df_year['hourly_generation_res']/1000, color='black', linewidth=0.8)
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Ticks every 3 months
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))    # Show month name (Jan, Apr, etc.)
            ax1.set_title(f"Generation Residuals ({year})")
            ax1.set_ylabel("GWh")
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig1.savefig(f"{save_path}/Generation_Res_{year}.eps", bbox_inches='tight')
            plt.show()

            # Plot 2: ACF
            fig2, ax2 = plt.subplots(figsize=sq_size)
            plot_acf(df_year['hourly_generation_res'].dropna(), lags=lags, ax=ax2, color="#1f77b4", zero=False, title=f"ACF Generation ({year})")
            ax2.set_ylim(-1.05, 1.05)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig2.savefig(f"{save_path}/Generation_ACF_{year}.eps", bbox_inches='tight')
            plt.show()

            # Plot 3: PACF
            fig3, ax3 = plt.subplots(figsize=sq_size)
            plot_pacf(df_year['hourly_generation_res'].dropna(), lags=lags, ax=ax3, color="#ff7f0e", zero=False, method="ywm", title=f"PACF Generation ({year})")
            ax3.set_ylim(-1.05, 1.05)
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig3.savefig(f"{save_path}/Generation_PACF_{year}.eps", bbox_inches='tight')
            plt.show()

            # ==========================================
            # VARIABLE 2: EMISSIONS
            # ==========================================

            # Plot 1: Residuals Time Series
            fig4, ax4 = plt.subplots(figsize=sq_size)
            ax4.plot(df_year['date_hourx'], df_year['hourly_emissions_res']/1000, color='black', linewidth=0.8)
            # 2. For Emissions Residuals (fig4/ax4)
            ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax4.set_title(f"Emissions Residuals ({year})")
            ax4.set_ylabel(r"Tons CO2 $10^3$")
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig4.savefig(f"{save_path}/Emissions_Res_{year}.eps", bbox_inches='tight')
            plt.show()

            # Plot 2: ACF
            fig5, ax5 = plt.subplots(figsize=sq_size)
            plot_acf(df_year['hourly_emissions_res'].dropna(), lags=lags, ax=ax5, color="#1f77b4", zero=False, title=f"ACF Emissions ({year})")
            ax5.set_ylim(-1.05, 1.05)
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig5.savefig(f"{save_path}/Emissions_ACF_{year}.eps", bbox_inches='tight')
            plt.show()

            # Plot 3: PACF
            fig6, ax6 = plt.subplots(figsize=sq_size)
            plot_pacf(df_year['hourly_emissions_res'].dropna(), lags=lags, ax=ax6, color="#ff7f0e", zero=False, method="ywm", title=f"PACF Emissions ({year})")
            ax6.set_ylim(-1.05, 1.05)
            ax6.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig6.savefig(f"{save_path}/Emissions_PACF_{year}.eps", bbox_inches='tight')
            plt.show()
     
    def plot_seasonal_diagnostics_dynamic(self, lags=100, save_path=None):
        """
        Create publication-quality seasonal diagnostic plots.
        For each season (yearseasonv2), regress hourly generation/emissions
        against time dummies, get residuals, and plot ACF/PACF.

        Parameters
        ----------
        lags : int
            Number of lags for ACF/PACF (default = 100)
        save_path : str or None
            Directory to save plots. If None, plots will be shown only.
        """

        if self.data is None:
            raise ValueError("No data loaded. Run load_and_clean_data() first.")

        self.set_publication_style()

        df = self.data.copy()
        df['date_hourx'] = pd.to_datetime(df['date_hour'], utc=True, errors='coerce')
        seasons = sorted(df['yearseasonv2'].dropna().unique())
        print(f"Found {len(seasons)} seasonal groups: {seasons}")

        for season in seasons:
            print(f"\n📅 Processing season: {season}")
            df_season = df[df['yearseasonv2'] == season].copy()

            # Refit generation model
            model_gen = ols("hourly_generation ~ factorinterval * factormonth + factordow + trend", data=df_season).fit()
            df_season['hourly_generation_res'] = model_gen.resid + model_gen.params[0]

            # Refit emissions model
            model_em = ols("hourly_emissions ~ factorinterval * factormonth + factordow + trend", data=df_season).fit()
            df_season['hourly_emissions_res'] = model_em.resid + model_em.params[0]

            # --- Plot generation residuals ---
            self.elegant_plot(
                df_season['date_hourx'], df_season['hourly_generation_res'],
                f"Hourly Load – Deseasoned & Detrended ({season})",
                "Hourly load (MWh)", save_path=save_path
            )

            # --- ACF/PACF for generation ---
            fig, ax = plt.subplots(2, 1, figsize=self.get_figsize(10 * 2))
            plot_acf(df_season['hourly_generation_res'].dropna(), lags=lags, ax=ax[0], color="#1f77b4", zero=False)
            plot_pacf(df_season['hourly_generation_res'].dropna(), lags=lags, ax=ax[1], color="#ff7f0e", zero=False, method="ywm")

            for i, title in enumerate(["ACF – Hourly Load", "PACF – Hourly Load"]):
                ax[i].set_title(f"{title} ({season})", loc='left', pad=10, weight='bold')
                ax[i].set_ylim(-1.05, 1.05)
                ax[i].spines[['top', 'right']].set_visible(False)
                ax[i].grid(True, alpha=0.3)

            fig.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}/ACF_PACF_Load_{season}.png", dpi=300, bbox_inches='tight')
            plt.show()

            # --- Plot emissions residuals ---
            self.elegant_plot(
                df_season['date_hourx'], df_season['hourly_emissions_res'],
                f"Hourly Emissions – Deseasoned & Detrended ({season})",
                "Hourly emissions (metric tons CO₂)", save_path=save_path
            )

            # --- ACF/PACF for emissions ---
            fig, ax = plt.subplots(2, 1, figsize=self.get_figsize(10 * 2))
            plot_acf(df_season['hourly_emissions_res'].dropna(), lags=lags, ax=ax[0], color="#1f77b4", zero=False)
            plot_pacf(df_season['hourly_emissions_res'].dropna(), lags=lags, ax=ax[1], color="#ff7f0e", zero=False, method="ywm")

            for i, title in enumerate(["ACF – Hourly Emissions", "PACF – Hourly Emissions"]):
                ax[i].set_title(f"{title} ({season})", loc='left', pad=10, weight='bold')
                ax[i].set_ylim(-1.05, 1.05)
                ax[i].spines[['top', 'right']].set_visible(False)
                ax[i].grid(True, alpha=0.3)

            fig.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}/ACF_PACF_Emissions_{season}.png", dpi=300, bbox_inches='tight')
            plt.show()
            print("\n✅ Seasonal diagnostics (dynamic residuals) completed.")
        
    def run_uniroot_tests(self,
                            y_col="hourly_emissions", 
                            group_col="year"):
        
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.formula.api import ols
        from arch.unitroot import  PhillipsPerron, KPSS
        
        df = self.data.copy()
        groups = sorted(df[group_col].unique())
        
        results_all = []
        
        for group in groups:
            # 1. Prepare Data
            subset = df[df[group_col] == group].copy()
            
            # Refit Deseasonalization (OLS)
           
            model = ols(f"{y_col} ~ factorinterval * factormonth + factordow + trend", data=subset).fit()
            # Restore the intercept to center data around the mean (standard practice)
            y_resid = model.resid + model.params[0]
            y = y_resid.to_numpy()
            
            # 2. Unit Root Tests
            dftest = adfuller(y, autolag="BIC")
            KPSStest = KPSS(y)
            pptest = PhillipsPerron(y)
            results = {
                'group_col': group,   
                "ADF_stat": dftest[0],
                "ADF_pval": dftest[1],
                "ADF_lag": dftest[2],
                "ADF_crit_1": dftest[4]['1%'],
                "ADF_crit_5": dftest[4]['5%'],
                "ADF_crit_10": dftest[4]['10%'],
                "KPSS_stat": KPSStest.stat,
                "KPSS_pval": KPSStest.pvalue,
                "KPSS_lag": KPSStest.lags,
                "KPSS_crit_10": KPSStest.critical_values['10%'],
                "KPSS_crit_5": KPSStest.critical_values['5%'],
                "KPSS_crit_1": KPSStest.critical_values['1%'],
                "PP_stat": pptest.stat,
                "PP_pval": pptest.pvalue,
                "PP_lag": pptest.lags,
                "PP_crit_1": pptest.critical_values['1%'],  
                "PP_crit_5": pptest.critical_values['5%'],
                "PP_crit_10": pptest.critical_values['10%']
            }

            results_all.append(results)

        # Convert to DataFrame
        df_results = pd.DataFrame(results_all)
        
        return df_results

    def run_nonlinearity_tests_r(self, 
                                 y_col="hourly_emissions", 
                                 group_col="year",
                                 save_path=None,
                                 file_prefix="nonlinear_test"):
        """
        Run rigorous Unit Root and Nonlinearity tests.
        Uses AIC to determine the optimal lag for Nonlinearity tests.
        """
        import pandas as pd
        import numpy as np
        import re
        import warnings
        from scipy.stats import f as f_dist
        from statsmodels.formula.api import ols
        from rpy2.robjects import r, globalenv, FloatVector
        from rpy2.robjects.packages import importr

        warnings.filterwarnings("ignore")

        # Load R Packages
        
        NTS = importr("NTS")
        fNonlinear = importr("fNonlinear")
    

        df = self.data.copy()
        results_all = []
        
        # Determine groups (Years)
        groups = sorted(df[group_col].unique())
       

        for group in groups:
            # 1. Prepare Data
            subset = df[df[group_col] == group].copy()
            
          
            model = ols(f"{y_col} ~ factorinterval * factormonth + factordow + trend", data=subset).fit()
            # Restore the intercept to center data around the mean (standard practice)
            y_resid = model.resid + model.params[0]
            y = y_resid.to_numpy()
            
            # Send to R
            globalenv["y"] = FloatVector(y)
            row = {group_col: group}

            # ==========================================
            # NONLINEARITY TESTS
            # ==========================================
            
            # CRITICAL STEP: Determine Optimal Lag (p) using AIC

            # R's ar() function automatically selects order based on AIC
            ar_fit = r.ar(FloatVector(y), method="mle")
            optimal_p = int(ar_fit.rx2("order")[0])
            
            # Safety check: If AIC selects 0 lags (white noise), force 1 to allow tests to run
            optimal_p = max(1, optimal_p)
            row["Nonlinear_Lag_Used"] = optimal_p 
            
            # 5. Tsay's F-Test (Uses optimal_p)
            try:
                ftest = NTS.F_test(FloatVector(y), optimal_p, thres=float(np.mean(y)))
                row["F_stat"]   = float(ftest.rx2("test.stat")[0])
                row["F_pvalue"] = float(ftest.rx2("p.value")[0])
            except:
                row["F_stat"] = np.nan

            # 6. Threshold Nonlinearity Test (TNT) (Uses optimal_p)
            # try:
            #     yN = int(0.1 * len(y)) # Trim 10%
            #     # Note: We keep d=1 (delay) as standard, but use the correct AIC 'p'
            #     tnt = NTS.thr_test(FloatVector(y), p=optimal_p, d=1, ini=yN, include_mean=False)
            #     stat, DF1, DF2 = float(tnt[0][0]), float(tnt[1][0]), float(tnt[1][1])
                
            #     row["TNT_stat"]   = stat
            #     row["TNT_pvalue"] = 1 - f_dist.cdf(stat, DF1, DF2)
            # except:
            #     row["TNT_stat"] = np.nan

            # 7. BDS Test (Independence)
            # BDS does not use AR lags; it uses embedding dimension 'm'. 
            # Standard practice is m=6 for hourly data.
            try:
                std_val = float(np.std(y))
                bds = fNonlinear.bdsTest(FloatVector(y), m=6, eps=2 * std_val)
                test_obj = bds.do_slot("test")
                
                row["BDS_stat"]   = float(test_obj.rx2("statistic")[0])
                row["BDS_pvalue"] = float(test_obj.rx2("p.value")[0])
                row["BDS_dim"]    = 6 
            except:
                row["BDS_stat"] = np.nan

            # 8. Peña-Rodríguez (PR) Test (Uses optimal_p)
            try:
                cmd = f'capture.output(print(PRnd(abs(y), m={optimal_p})))'
                pr_output = r(cmd)
                
                if len(pr_output) >= 2:
                    tokens = re.split(r'\s+', pr_output[1].strip())
                    row["PR_stat"]   = float(tokens[-2])
                    row["PR_pvalue"] = float(tokens[-1])
                else:
                    row["PR_stat"] = np.nan
            except:
                row["PR_stat"] = np.nan

            results_all.append(row)

        df_out = pd.DataFrame(results_all)
        if save_path:
            df_out.to_excel(f"{save_path}/{file_prefix}.xlsx", index=False)
            print(f"💾 Saved results with AIC lags to: {save_path}/{file_prefix}.xlsx")
            
        return df_out
    
    def compare_models_standardized(self):
        """
        Fits OLS, ARIMA, and MSM on STANDARDIZED data to ensure valid AIC/BIC comparison.
        """
        import numpy as np
        import pandas as pd
        import itertools
        # Check for pmdarima
        try:
            import pmdarima as pm
        except ImportError:
            pass # We will use the manual grid search from before if this fails

        from statsmodels.regression.linear_model import OLS
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        from sklearn.preprocessing import StandardScaler
        from patsy import dmatrices
        import warnings

        warnings.filterwarnings("ignore")
        np.random.seed(987)

        # 1. DATA PREP
        df = self.data.copy()
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()
        
        # 2. GLOBAL SCALING (Crucial for Comparison)
        print("🔹 Standardizing data for valid AIC/BIC comparison...")
        
        y_col_raw = 'hourly_emissions_res'
        exog_cols_raw = ['hourly_generation_renewables_res', 'hourly_generation_nonrenewables_res']
        
        # Scale Target (Y)
        scaler_y = StandardScaler()
        y_scaled = pd.Series(
            scaler_y.fit_transform(df[[y_col_raw]]).flatten(),
            index=df.index,
            name='y_standardized'
        )
        
        
        # Scale Exogenous (X)
        scaler_x = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler_x.fit_transform(df[exog_cols_raw]),
            columns=exog_cols_raw,
            index=df.index
        )

        results_list = []

        def get_model_stats(model_name, model_result):
            import numpy as np
            
            # 1. Extract base metrics
            aic = model_result.aic
            bic = model_result.bic
            llf = model_result.llf
            
            # 2. Handle HQIC (Manual calculation for OLS if missing)
            if hasattr(model_result, 'hqic'):
                hqic = model_result.hqic
            else:
                # Manual Calculation: -2*LLF + 2*k*ln(ln(n))
                n = model_result.nobs
                k = len(model_result.params)
                hqic = -2 * llf + 2 * k * np.log(np.log(n))
            
            return {
                "Model": model_name,
                "AIC": round(aic, 2),
                "BIC": round(bic, 2),
                "HQIC": round(hqic, 2),
                "Log_Likelihood": round(llf, 2)
            }

        # =========================================================
        # MODEL 1: OLS (Standardized)
        # =========================================================
        try:
            import statsmodels.api as sm
            # Since data is already de-seasonalized, we do NOT need dummies/interactions.
            # We simply regress Scaled Emissions on Scaled Generation + Constant.
            
            # 1. Add Constant to the standardized exogenous variables
            X_ols = sm.add_constant(X_scaled)
            
            # 2. Safety cast to ensure float type (avoids the object error)
            X_ols = X_ols.astype(float)
            y_scaled_clean = y_scaled.astype(float)

            # 3. Fit
            ols_mod = OLS(y_scaled_clean, X_ols).fit()
            results_list.append(get_model_stats("OLS (Standardized)", ols_mod))
            
        except Exception as e:
            print(f"⚠️ OLS Failed: {e}")
            
        # =========================================================
        # MODEL 3: AUTO-ARIMA (Optimized)
        # =========================================================
        try:
            print("🔹 Running Auto-ARIMA selection (BIC)...")
            
            # Step A: Auto-ARIMA on emissions only (finding optimal p, d, q)
            # We exclude exog here to find the residual structure first (common practice)
            auto_model = pm.auto_arima(
                y_scaled,
                exogenous=None,       
                information_criterion='bic',
                seasonal=False,       # Set True if you want SARIMA
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            # Extract orders
            p_opt, d_opt, q_opt = auto_model.order
            print(f"   ✅ Optimal Order Found: p={p_opt}, d={d_opt}, q={q_opt}")
            
            # Use the found order
            final_order = (p_opt, d_opt, q_opt)
            
            # Step B: Fit the full ARIMAX model using statsmodels (with exog)
            # We use statsmodels for the final fit to ensure consistent AIC/BIC calculation logic
            arima_mod = ARIMA(
                endog=y_scaled, 
                exog=X_scaled, 
                order=final_order
            )
            arima_res = arima_mod.fit()
            
    
            results_list.append(get_model_stats(f"ARIMA {final_order} (Standardized)", arima_res))
            
        except Exception as e:
            print(f"⚠️ ARIMA Failed: {e}")

        # =========================================================
        # MODEL 3: MS-ARX (Standardized)
        # =========================================================
        try:
            print("🔹 Fitting MS-ARX on standardized data...")
            
            ms_mod = MarkovAutoregression(
                endog=y_scaled, 
                exog=X_scaled, 
                k_regimes=2, 
                order=1, 
                trend='c', 
                switching_trend=True, 
                switching_exog=[True, True], 
                switching_variance=False
            )
            ms_res = ms_mod.fit(disp=False)
            results_list.append(get_model_stats("MS-ARX (Standardized)", ms_res))
            
        except Exception as e:
            print(f" MSM Failed: {e}")

        # =========================================================
        # OUTPUT
        # =========================================================
        summary_df = pd.DataFrame(results_list)
        if not summary_df.empty:
            summary_df = summary_df.set_index('Model')
            summary_df.sort_values(by='BIC', ascending=True, inplace=True)
        
        print("\n✅ Comparison Complete (All Standardized).")
        return summary_df
    
    def mef_estimate(self, group_col='year', include_markov=True):
            import numpy as np
            import pandas as pd
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools.tools import add_constant
            from statsmodels.stats.sandwich_covariance import cov_hac
            from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
            from patsy import dmatrices
            np.random.seed(987)

            df = self.data.copy()
            groups = sorted(df[group_col].unique())
            results = []
            #create dataset to store Residuals
            resid_dict = {'year': [], 'Residuals': []}
            for group in groups:
                print(f"\n🔹 Processing {group_col} = {group}...")
                subset = df[df[group_col] == group].copy()
                
                # Check for sufficient sample size
                if subset.shape[0] < 50:
                    print("   Skipped: not enough observations.")
                    continue

                # ------------------------
                # 1. Detrending Strategy (Frisch-Waugh-Lovell)
                # ------------------------
                # We use the same logic as your R code: remove seasonality, keep residuals + mean
                
                # Detrend hourly_emissions
                y_formula = "hourly_emissions_mlb ~ C(factordow, Treatment(reference='Monday')) + C(factorinterval)*C(factormonth)+trend"
                y_y, y_X = dmatrices(y_formula, data=subset, return_type='dataframe')
                y_model = OLS(y_y, y_X).fit()
                subset['em_res'] = y_model.resid + y_model.params.iloc[0] 

                # Detrend generation components
                for col in ['hourly_generation_renewables_mkwh', 'hourly_generation_nonrenewables_mkwh']:
                    x_formula = f"{col} ~ C(factordow, Treatment(reference='Monday')) + C(factorinterval)*C(factormonth)+trend"
                    x_y, x_X = dmatrices(x_formula, data=subset, return_type='dataframe')
                    x_model = OLS(x_y, x_X).fit()
                    subset[col + '_res'] = x_model.resid + x_model.params.iloc[0]
              
                # ------------------------
                # 2. OLS MEF
                # ------------------------
                target_col = 'hourly_generation_nonrenewables_mkwh'

                # 2. Define the formula
                formula = (
                    'hourly_emissions_mlb ~ hourly_generation_renewables_mkwh + '
                    'hourly_generation_nonrenewables_mkwh + '
                    'C(factorinterval)*C(factormonth) + C(factordow)'
                )
                C02, X_mef = dmatrices(formula, data=subset, return_type='dataframe')
                ols_model = OLS(C02, X_mef).fit()
                cov = cov_hac(ols_model, nlags=48) 
               
                ols_mef = ols_model.params[target_col]
                col_index = X_mef.columns.get_loc(target_col)
                ols_se = np.sqrt(cov[col_index, col_index])
               

                # ------------------------
                # 3. First Differences (Hawkes)
                # ------------------------
                em_diff = subset['em_res'].diff()
                gen_r_diff = subset['hourly_generation_renewables_mkwh_res'].diff()
                gen_nr_diff = subset['hourly_generation_nonrenewables_mkwh_res'].diff()
                
                diff_df = pd.concat([em_diff, gen_r_diff, gen_nr_diff], axis=1).dropna()
                diff_df.columns = ['d_em', 'd_ren', 'd_nonren']
                
                diff_model = OLS(diff_df['d_em'], add_constant(diff_df[['d_ren', 'd_nonren']])).fit()
                diff_mef = diff_model.params['d_nonren']
                diff_se = diff_model.bse['d_nonren']
                
                # ------------------------
                # 3. First Differences (Hawkes)
                # ------------------------
                em_diff = subset['hourly_emissions_mlb'].diff()
                gen_r_diff = subset['hourly_generation_renewables_mkwh'].diff()
                gen_nr_diff = subset['hourly_generation_nonrenewables_mkwh'].diff()
                
                diff_df = pd.concat([em_diff, gen_r_diff, gen_nr_diff], axis=1).dropna()
                diff_df.columns = ['d_em', 'd_ren', 'd_nonren']
                
                diff_model_2 = OLS(diff_df['d_em'], add_constant(diff_df[['d_ren', 'd_nonren']])).fit()
                diff_mef_2 = diff_model_2.params['d_nonren']
                diff_se_2 = diff_model_2.bse['d_nonren']

                # ------------------------
                # 4. ARIMA MEF
                # ------------------------
               
                try:
                    import pmdarima as pm
                    # Step A: Auto-ARIMA on emissions only (finding d and q)
                   # We use 'bic' 
                    auto_model = pm.auto_arima(
                        subset['em_res'],
                        exogenous=None,       # R code runs auto.arima WITHOUT xreg first
                        information_criterion='bic',
                        seasonal=False,       # Assuming non-seasonal for residuals
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore'
                    )
                    
                    # Extract orders
                    # auto_model.order returns (p, d, q)
                    p_opt, d_opt, q_opt = auto_model.order
                    print(f"   ARIMA orders selected:p={p_opt} d={d_opt}, q={q_opt}")
                    # Step B: Force p=1 (as per R code: order = c(1,d,q))
                    final_order = (p_opt, d_opt, q_opt)
                    
                    # Step C: Fit the specific model with Exogenous Regressors
                    # Note: Your R code used 'hourly_generation_res'.
                    # Since your Python class splits this into Ren/NonRen, we use both as exog.
                    arima_exog = subset[['hourly_generation_renewables_mkwh_res', 'hourly_generation_nonrenewables_mkwh_res']]
                    
                    arima_model = ARIMA(
                        endog=subset['em_res'],
                        exog=arima_exog,
                        order=final_order
                    ).fit()
                    
                    # Extract Results
                    target_col = 'hourly_generation_nonrenewables_mkwh_res'
                    arima_mef = arima_model.params[target_col]
                    arima_se = arima_model.bse[target_col]
                    
                except:
                    arima_mef = arima_se = np.nan

                # ------------------------
                # 5. Markov Switching MEF (SCALED & ROBUST)
                # ------------------------
                
                
                ms_high_mef = ms_high_se = ms_low_mef = ms_low_se = np.nan
                p11 = p00 = dur_high = dur_low = np.nan

                if include_markov:
                    try:
                        # A. Prepare Data
                        ms_data = subset.reset_index(drop=True)
                        
                        # Check sufficient observations
                        if len(ms_data) < 100:
                            raise ValueError("Insufficient data for Markov Switching")
                        
                        # B. Scale variables
                        from sklearn.preprocessing import StandardScaler
                        scaler_y = StandardScaler()
                        scaler_x = StandardScaler()

                        ms_data['em_scaled'] = scaler_y.fit_transform(ms_data[['em_res']])
                        
                        X_unscaled = ms_data[['hourly_generation_renewables_mkwh_res', 'hourly_generation_nonrenewables_mkwh_res']]
                        X_scaled_array = scaler_x.fit_transform(X_unscaled)
                        
                        y_scaled = ms_data['em_scaled']
                        
                        # C. Create exog WITHOUT manual constant
                        X_scaled = pd.DataFrame(X_scaled_array, columns=['gen_ren', 'gen_nonren'])

                        # D. Model with trend='c' (let the model add the constant) # Replicate R's sw=c(T,T,T,F)
                        ms_model = MarkovAutoregression(
                            endog=y_scaled,
                            exog=X_scaled,
                            k_regimes=2,
                            order =1,
                            trend = 'c',
                            switching_trend=True,
                            switching_exog=[True, True],  # Allow slopes to switch
                            switching_variance= False 
                        )

                        # E. Fit with better initialization
                        ms_results = ms_model.fit()
                        
                        #create check for the residuals to assests if the  model MSM-ARX(1) is a good fit for the data, if not we are sad but we will use it anyway
                        resids = ms_results.model.endog[1:] - ms_results.fittedvalues
                        unscaled_resid = resids * float(scaler_y.scale_[0])
                        #here extend helps to create a long format dataset for residuals with corresponding years for plotting
                        resid_dict['year'].extend([group] * len(unscaled_resid))
                        resid_dict['Residuals'].extend(unscaled_resid)

                        # Get scaled coefficients and SE for the non-renewable generation term (the MEF)
                        beta_nr_0_s = ms_results.params.get('x2[0]', np.nan)
                        beta_nr_1_s = ms_results.params.get('x2[1]', np.nan)
                        se_nr_0_s = ms_results.bse.get('x2[0]', np.nan)
                        se_nr_1_s = ms_results.bse.get('x2[1]', np.nan)

                        # UNSCALE: beta_unscaled = beta_scaled * (std_y / std_x)
                        scale_y = float(scaler_y.scale_[0])
                        # scaler_x.scale_ has scales for ['hourly_generation_renewables_res', 'hourly_generation_nonrenewables_res']
                        scale_nonren = float(scaler_x.scale_[1])
                        convert_factor = scale_y / scale_nonren
                        
                        beta_nr_0 = beta_nr_0_s * convert_factor
                        beta_nr_1 = beta_nr_1_s * convert_factor
                        se_nr_0 = se_nr_0_s * convert_factor
                        se_nr_1 = se_nr_1_s * convert_factor

                        # G. Sort regimes based on the MEF value for non-renewable generation
                        if beta_nr_0 > beta_nr_1:
                            ms_high_mef, ms_low_mef = beta_nr_0, beta_nr_1
                            ms_high_se, ms_low_se = se_nr_0, se_nr_1
                            p00 = ms_results.params.get('p[0->0]', np.nan) # High MEF regime is regime 0
                            p10 = ms_results.params.get('p[1->0]', np.nan) # Low MEF regime is regime 1
                            p01 = 1 - p00
                            p11 = 1 - p10
                            
                        else:
                            ms_high_mef, ms_low_mef = beta_nr_1, beta_nr_0
                            ms_high_se, ms_low_se = se_nr_1, se_nr_0
                            p10 = 1 - ms_results.params.get('p[0->0]', np.nan)  # Low MEF regime is regime 0
                            p00 = 1 - ms_results.params.get('p[1->0]', np.nan)  # High MEF regime is regime 1
                            p01 = 1 - p00
                            p11 = 1 - p10

                        
                        dur_high = 1 / p01 if p01 > 1e-6 else np.inf
                        dur_low = 1 / p10 if p10 > 1e-6 else np.inf

                    except Exception as e:
                        print(f"⚠️ MS estimation failed: {e}")

                # ------------------------
                # Aggregation
                # ------------------------
                avg_em = subset['hourly_emissions_mlb'].sum() / (
                    subset['hourly_generation_renewables_mkwh'].sum() + subset['hourly_generation_nonrenewables_mkwh'].sum()
                )
                
                results.append({
                    group_col: group,
                    'OLS_MEF': ols_mef, 'OLS_SE': ols_se,
                    'Diff_MEF': diff_mef, 'Diff_SE': diff_se,
                    'Diff_MEF_2': diff_mef_2, 'Diff_SE_2': diff_se_2,
                    'ARIMA_MEF': arima_mef, 'ARIMA_SE': arima_se,
                    'MS_High_MEF': ms_high_mef, 'MS_High_SE': ms_high_se,
                    'MS_Low_MEF': ms_low_mef, 'MS_Low_SE': ms_low_se,
                    'P_high': p00, 'P_Low': p11,
                    'Dur_High': dur_high, 'Dur_Low': dur_low,
                    'Avg_Emissions': avg_em
                })
            #create a dataframe for residuals to be used in plotting diagnostics
            self.msm_residuals = pd.DataFrame(resid_dict)
            self.data['MSM_residuals'] = self.msm_residuals['Residuals']
            return pd.DataFrame(results)
        
    def plot_annual_diagnostics_MSM(self, lags=100, save_path='images_msm'):
        
        """
        Dynamic version: Recalculates residuals each year.
        Generates 3 separate SQUARE plots per variable: Residuals, ACF, and PACF.
        """
        import matplotlib.dates as mdates
        if self.msm_residuals is None:
            raise ValueError("No data loaded. Run load_and_clean_data() first.")

        self.set_publication_style()


        df = self.msm_residuals.copy()
        years = sorted(df['year'].unique())
        print(f"🧭 Found {len(years)} years: {years}")
        
        sq_size = (6, 6)
        
        for year in years:
            print(f"\n📅 Processing {year}...")
            df_year = df[df['year'] == year].copy()
            df_plot = df_year.dropna(subset=['Residuals'])
             # Plot 1: Residuals Time Series
            fig4, ax4 = plt.subplots(figsize=sq_size)
            ax4.plot( df_year['Residuals']/1000, color='black', linewidth=0.8)
            # 2. For Emissions Residuals (fig4/ax4)
            ax4.set_title(f"Residuals MSM ({year})")
            ax4.set_ylabel(r"Tons CO2 $10^3$")
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig4.savefig(f"{save_path}/MSM_Res_{year}.eps", bbox_inches='tight')
            plt.show()
        

            # Plot 2: ACF
            fig5, ax5 = plt.subplots(figsize=sq_size)
            plot_acf(df_plot['Residuals'].dropna(), lags=lags, ax=ax5, color="#1f77b4", zero=False, title=f"ACF MSM ({year})")
            ax5.set_ylim(-1.05, 1.05)
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig5.savefig(f"{save_path}/MSM_ACF_{year}.eps", bbox_inches='tight')
            plt.show()

            # Plot 3: PACF
            fig6, ax6 = plt.subplots(figsize=sq_size)
            plot_pacf(df_plot['Residuals'].dropna(), lags=lags, ax=ax6, color="#ff7f0e", zero=False, method="ywm", title=f"PACF MSM ({year})")
            ax6.set_ylim(-1.05, 1.05)
            ax6.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: fig6.savefig(f"{save_path}/MSM_PACF_{year}.eps", bbox_inches='tight')
            plt.show()    
        
    def plot_smoothed_probabilities(self, lags=1):
        """
        Fits a Markov Switching Model for the ENTIRE time series and plots:
        1. Time Series Colored by Regime
        2. Smoothed Probabilities
        3. Viterbi Path (Most Likely Regime Sequence)
        """
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        from sklearn.preprocessing import StandardScaler
        import matplotlib.patches as mpatches


        # 1. Use Full Dataset
        subset = self.data.copy().reset_index(drop=True)
        
        
        # 2. Scale Variables (Crucial for convergence on large datasets)
        print(" Scaling variables...")
        scaler_y = StandardScaler()
        scaler_x = StandardScaler()
        
        y_scaled = scaler_y.fit_transform(subset[['hourly_emissions_res']])
        X_unscaled = subset[['hourly_generation_renewables_res', 'hourly_generation_nonrenewables_res']]
          
        X_scaled_array = scaler_x.fit_transform(X_unscaled)                # C. Create exog WITHOUT manual constant
        X_scaled = pd.DataFrame(X_scaled_array, columns=['gen_ren', 'gen_nonren'])

        # 3. Fit Markov Switching Model
        print(f"⏳ Fitting MSM on full time series ({len(subset)} observations)... this may take a moment.")
        ms_model = MarkovAutoregression(
                            endog=y_scaled,
                            exog=X_scaled,
                            k_regimes=2,
                            order =1,
                            trend = 'c',
                            switching_trend=True,
                            switching_exog=[True, True],  # Allow slopes to switch
                            switching_variance=False 
                        )

            
        ms_results = ms_model.fit()
        print(ms_results.summary())
        
        print("✅ Model fitted successfully.")

        # 4. Extract Results

        smoothed_prob_0, smoothed_prob_1 = zip(*ms_results.smoothed_marginal_probabilities)  # Transpose to DataFrame format
        smoothed_probs = pd.DataFrame([smoothed_prob_0, smoothed_prob_1]).T
        print(ms_results.params)
        
        # Identify High vs Low MEF Regime
        beta_0 = ms_results.params[6]
        beta_1 = ms_results.params[7]
        print(f" Regime 0 MEF: {beta_0:.4f}, Regime 1 MEF: {beta_1:.4f}")
        if beta_0 > beta_1:
            high_regime = 0
            low_regime = 1
        else:
            high_regime = 1
            low_regime = 0
            
        print(f"🔹 Regime {high_regime} identified as HIGH MEF")
        print(f"🔹 Regime {low_regime} identified as LOW MEF")
        subset = subset.iloc[1:, :].reset_index(drop=True) # Adjust for initial missing due to AR(1)
        # 5. Plotting
        self.set_publication_style()
        fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # Plot A: Time Series Colored by Regime
        axes[0].set_title(f"A. Hourly Emissions Colored by Regime (Full Series)", loc='left', weight='bold')
        axes[0].plot(subset['date_hour'], subset['hourly_emissions_res'], color='gray', alpha=0.3, lw=0.3, label='Observed')
        
        
        
        # Overlay Low Regime
        low_mask = (smoothed_probs[low_regime]).values>= 0.5
        axes[0].scatter(subset.loc[low_mask, 'date_hour'], subset.loc[low_mask, 'hourly_emissions_res'], 
                       color='#1f77b4', s=0.5, alpha=0.5, label='Low MEF Regime')
        
        axes[0].set_ylabel("Emissions (Residuals)")
        # Custom legend to avoid clutter from scatter points
        legend_elements = [
            mpatches.Patch(color='gray', alpha=0.3, label='Observed'),
            mpatches.Patch(color='#d62728', label='High MEF Regime'),
            mpatches.Patch(color='#1f77b4', label='Low MEF Regime')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right', frameon=True)

        # Plot B: Smoothed Probabilities
        axes[1].set_title("B. Smoothed Probability of High MEF Regime", loc='left', weight='bold')
        # Plot area chart
        axes[1].fill_between(subset['date_hour'], 0, smoothed_probs[high_regime], color='#d62728', alpha=0.3)
        axes[1].plot(subset['date_hour'], smoothed_probs[high_regime], color='#d62728', lw=0.5)
        axes[1].set_ylabel("Probability")
        axes[1].set_ylim(0, 1.05)
        
        # Plot C: Viterbi Path (Discrete States)
        axes[2].set_title("C. Viterbi Path (Most Likely Regime Sequence)", loc='left', weight='bold')
        
        # Map probabilities to discrete states
        states = smoothed_probs.idxmax(axis=1)
        binary_path = states.apply(lambda x: 1 if x == high_regime else 0)
        
        axes[2].step(subset['date_hour'], binary_path, where='mid', color='black', lw=0.8)
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(['Low MEF', 'High MEF'])
        axes[2].set_ylabel("Regime")
        axes[2].set_xlabel("Date")
        
        # Shade High MEF areas
        axes[2].fill_between(subset['date_hour'], 0, 1, where=(binary_path==1), 
                            color='#d62728', alpha=0.1, step='mid')

        plt.tight_layout()
        plt.show()
    
    def plot_mef_by_year(self, results_df, include_markov=True, hawkes=True):
        """
        Plot MEF estimates by year in two subplots:
        1. MSM Estimates vs Average
        2. Benchmark Estimates (OLS, ARIMA) vs Average
        Fits linear regression trends with Confidence Intervals.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        self.set_publication_style()
        
        # Create 2 subplots side-by-side, sharing Y-axis for easy comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
        
        years = results_df['year'].astype(int)
        
        # Helper function to plot Data Points + Error Bars + Regression Trend
        def add_trend(ax, x, y, yerr, label, color, marker):
            # 1. Plot the Regression Line with Shaded CI (using Seaborn)
            # scatter=False because we plot specific points with error bars manually below
            sns.regplot(x=x, y=y, ax=ax, scatter=False, color=color, 
                        label=f'{label} (Trend)', ci=95, line_kws={'linestyle': '--', 'alpha': 0.8})
            
            # 2. Plot the specific Yearly Points with their Standard Errors
            ax.errorbar(x, y, yerr=1.96*yerr, fmt=marker, label=f'{label} (Obs)', 
                        markersize=8, capsize=5, capthick=2, linewidth=0, color=color, alpha=0.9)

        # ==========================================
        # LEFT PLOT: MSM vs Average
        # ==========================================
        ax1.set_title("Regime-Dependent Marginal Emissions", fontsize=20)
        
        # 1. Average Emissions (Baseline)
        add_trend(ax1, years, results_df['Avg_Emissions'], 0, 'Average', 'pink', 'D')

        # 2. Markov Switching Models
        if include_markov and 'MS_High_MEF' in results_df.columns:
            # High Regime
            add_trend(ax1, years, results_df['MS_High_MEF'], results_df['MS_High_SE'], 
                      'MS-HIGH', 'blue', 'd')
            # Low Regime
            add_trend(ax1, years, results_df['MS_Low_MEF'], results_df['MS_Low_SE'], 
                      'MS-LOW', 'purple', 'v')

        # ==========================================
        # RIGHT PLOT: Benchmarks vs Average
        # ==========================================
        ax2.set_title("Benchmark Models Marginal Emissions", fontsize=20)

        # 1. Average Emissions (for comparison)
        add_trend(ax2, years, results_df['Avg_Emissions'], 0, 'Average', 'pink', 'D')

        # 2. US-FE (OLS)
        add_trend(ax2, years, results_df['OLS_MEF'], results_df['OLS_SE'], 
                  'US-FE', 'orange', 'o')

        # 3. ARIMA
        add_trend(ax2, years, results_df['ARIMA_MEF'], results_df['ARIMA_SE'], 
                  'ARIMA', 'red', '^')

        # 4. Hawkes (Optional)
        if hawkes and 'Diff_MEF_2' in results_df.columns:
            add_trend(ax2, years, results_df['Diff_MEF_2'], results_df['Diff_SE_2'], 
                      'HAWKES', 'gray', 's')

        # ==========================================
        # FORMATTING
        # ==========================================
        for ax in [ax1, ax2]:
            ax.set_xlabel('Year', fontsize=20)
            ax.set_xticks(years)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.grid(False)
            # Simply legends to avoid duplication if trends/obs create too many entries
            # We filter handles to only show the main identifiers if preferred, 
            # or keep all to show what is trend vs observation.
            ax.legend(fontsize=16, loc='best', framealpha=0.9)

        ax1.set_ylabel('Marginal Emissions (lbs/kWh)', fontsize=20)
        ax2.set_ylabel('', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Plot transition probabilities if available (kept from your original code)
        if include_markov and 'P11' in results_df.columns:
            self._plot_transition_probs(results_df)
            
   


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
