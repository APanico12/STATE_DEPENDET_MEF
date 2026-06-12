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
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.formula.api import ols
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")


class OtherZonesAnalysis:
    """Other Zones Marginal Emission Factor Analysis"""

    def __init__(self, data_path, sheet_name=None):
        """Initialize with data path"""
        self.data_path = data_path
        self.sheet_name = sheet_name
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
        plt.style.use("seaborn-v0_8-white")

        # 2️⃣ FONT SETTINGS — Helvetica or Arial preferred by most journals
        font_params = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "mathtext.fontset": "stixsans",  # Sans-serif math look
        }

        # 3️⃣ OPTIONAL: LATEX RENDERING (for math-heavy papers)
        if use_latex:
            font_params.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "text.latex.preamble": (
                        r"\usepackage{amsmath} \usepackage{helvet} "
                        r"\usepackage{sansmath} \sansmath"
                    ),
                }
            )

        # 4️⃣ AXES & GRID STYLE
        axes_params = {
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }

        # 5️⃣ TICKS — fine-tuned for print readability
        tick_params = {
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        }

        # 6️⃣ COLORBLIND-FRIENDLY PALETTE (Okabe–Ito)
        cb_palette = [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ]

        # 7️⃣ APPLY ALL STYLE SETTINGS
        mpl.rcParams.update(font_params)
        mpl.rcParams.update(axes_params)
        mpl.rcParams.update(tick_params)
        mpl.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "lines.linewidth": 1.0,
                "lines.markersize": 4,
                "axes.prop_cycle": plt.cycler("color", cb_palette),
            }
        )

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

    def load_and_clean_data(self, start_date="2019-01-01", end_date="2025-12-31"):
        """Load and clean data applying the pipeline from Different_zones.ipynb"""
        print(f"Loading data from {self.data_path}...")
        if self.sheet_name is not None:
            df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        else:
            df = pd.read_excel(self.data_path)
        df.columns = df.columns.str.strip()

        df["UTC time"] = pd.to_datetime(df["UTC time"], errors="coerce")
        df = df[
            (df["UTC time"] >= pd.to_datetime(start_date))
            & (df["UTC time"] <= pd.to_datetime(end_date))
        ].copy()

        # Rename columns to match pipeline
        rename_map = {
            "UTC time": "Date",
            "Demand": "D",
            "Net generation": "NG",
            "CO2 Emissions Generated": "daily_emissions",
            "NG: COL": "Gen_Coal",
            "NG: NG": "Gen_Gas",
            "NG: NUC": "Gen_Nuclear",
            "NG: WND": "Gen_Wind",
            "NG: SUN": "Gen_Sun",
            "NG: OTH": "Gen_Others",
            "NG: OIL": "Gen_Oil",
        }
        df.rename(columns=rename_map, inplace=True)

        # Select relevant columns
        cols = [
            "Date",
            "D",
            "NG",
            "daily_emissions",
            "Gen_Coal",
            "Gen_Gas",
            "Gen_Nuclear",
            "Gen_Wind",
            "Gen_Sun",
            "Gen_Others",
            "Gen_Oil",
        ]
        if "Hour" in df.columns:
            cols.append("Hour")
        self.data = df[[c for c in cols if c in df.columns]].copy()

        print("🔧 Removing outliers (IQR) and interpolating gaps...")
        cols_to_clean = [col for col in cols if col not in ["Date", "Hour"]]

        # Clean each column: Convert to numeric, remove outliers using IQR, and interpolate
        # assure values are >= 0

        for col in cols_to_clean:
            if col not in self.data.columns:
                continue
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
            self.data[col] = self.data[col].clip(lower=0)

            # IQR Method for Outlier Detection
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            # Mask outliers as NaN
            outlier_mask = (self.data[col] < (Q1 - 1.5 * IQR)) | (
                self.data[col] > (Q3 + 1.5 * IQR)
            )
            self.data.loc[outlier_mask, col] = np.nan

            # Cubic Interpolation to fill the NaNs created by outliers

            self.data[col] = self.data[col].interpolate(
                method="linear", limit_direction="both"
            )

        print("🧮 Calculating aggregate renewable and non-renewable generation...")
        renewable_cols = [c for c in ["Gen_Wind", "Gen_Sun"] if c in self.data.columns]
        non_renewable_cols = [
            c
            for c in ["Gen_Coal", "Gen_Gas", "Gen_Nuclear", "Gen_Oil", "Gen_Others"]
            if c in self.data.columns
        ]

        self.data["renewable_gen"] = self.data[renewable_cols].sum(axis=1)
        self.data["non_renewable_gen"] = self.data[non_renewable_cols].sum(axis=1)

        # --- Convert units ---
        self.data["hourly_emissions_mlb"] = (
            self.data["daily_emissions"] * 2204.6 / 1_000_000
        )  # metric tons → million lbs
        self.data["hourly_generation_mkwh"] = (
            self.data["NG"] * 1000 / 1_000_000
        )  # GWh → million kWh
        self.data["hourly_generation_renewables_mkwh"] = (
            self.data["renewable_gen"] * 1000 / 1_000_000
        )  # GWh → million kWh
        self.data["hourly_generation_nonrenewables_mkwh"] = (
            self.data["non_renewable_gen"] * 1000 / 1_000_000
        )  # GWh → million kWh

        # --- Add index column T ---
        self.data["T"] = np.arange(1, len(self.data) + 1)

        # --- Summary ---
        print("✅ Data loaded and cleaned successfully.")
        print(f"   Date range: {start_date} → {end_date}")
        print(f"   Observations: {len(self.data)}")
        print(f"   Columns: {list(self.data.columns)}")

        return self.data

    def extract_seasonality(self):
        """Extract seasonality from the cleaned data using OLS"""
        print("Extracting seasonality...")
        df = self.data.copy()

        # Time variables
        df["Date"] = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        df["year"] = df["Date"].dt.year
        df["day"] = df["Date"].dt.day_name().astype("category")
        df["Hour"] = df["Date"].dt.hour if "Hour" not in df.columns else df["Hour"]
        df["trend"] = np.arange(1, len(df) + 1)

        # Setup formula based on available columns
        if "Hour" in df.columns:
            formula = " ~ C(Hour)*C(month) + C(month)*C(year) + C(day) + trend"
        else:
            formula = " ~ C(month)*C(year) + C(day) + trend"

        print("  -> Fitting emissions...")
        model_em = ols("hourly_emissions_mlb" + formula, data=df).fit()
        df["emissions_res"] = model_em.resid + model_em.params["Intercept"]

        print("  -> Fitting renewable gen...")
        model_ren = ols("hourly_generation_renewables_mkwh" + formula, data=df).fit()
        df["renewable_gen_res"] = model_ren.resid + model_ren.params["Intercept"]

        print("  -> Fitting non renewable gen...")
        model_nonren = ols(
            "hourly_generation_nonrenewables_mkwh" + formula, data=df
        ).fit()
        df["non_renewable_gen_res"] = (
            model_nonren.resid + model_nonren.params["Intercept"]
        )

        self.data = df
        print("Seasonality extraction complete.")
        return df

    def compute_msm_per_year(self, group_col="year"):
        """Compute Markov Switching Model per year using renewable and non-renewable columns"""
        print(f"Computing MSM per {group_col}...")
        df = self.data.copy()

        if group_col not in df.columns:
            df[group_col] = pd.to_datetime(df["Date"]).dt.year

        groups = sorted(df[group_col].dropna().unique())
        results = []

        for group in groups:
            print(f"Processing {group}...")
            subset = df[df[group_col] == group].copy()

            if len(subset) < 50:
                print(f"  -> Skipping {group}, not enough data.")
                continue

            ms_data = subset.reset_index(drop=True)

            # --- Compute Average Emissions ---
            avg_em = ms_data["hourly_emissions_mlb"].sum() / (
                ms_data["hourly_generation_renewables_mkwh"].sum() + ms_data["hourly_generation_nonrenewables_mkwh"].sum()
            )

            # --- Compute OLS (US-FE) ---
            try:
                X_ols = sm.add_constant(ms_data[["renewable_gen_res", "non_renewable_gen_res"]])
                ols_model = sm.OLS(ms_data["emissions_res"], X_ols).fit()
                cov = cov_hac(ols_model, nlags=48)
                ols_mef = ols_model.params["non_renewable_gen_res"]
                col_idx = list(X_ols.columns).index("non_renewable_gen_res")
                ols_se = np.sqrt(cov[col_idx, col_idx])
            except Exception as e:
                print(f"  -> OLS failed for {group}: {e}")
                ols_mef, ols_se = np.nan, np.nan

            high_mef = high_se = low_mef = low_se = np.nan

            scaler_y = StandardScaler()
            scaler_x = StandardScaler()

            y_scaled_array = scaler_y.fit_transform(ms_data[["emissions_res"]])
            y_scaled = pd.Series(y_scaled_array.flatten(), name="emissions")

            X_unscaled = ms_data[["renewable_gen_res", "non_renewable_gen_res"]]
            X_scaled_array = scaler_x.fit_transform(X_unscaled)
            X_scaled = pd.DataFrame(X_scaled_array, columns=["gen_ren", "gen_nonren"])

            try:
                ms_model = MarkovAutoregression(
                    endog=y_scaled,
                    exog=X_scaled,
                    k_regimes=2,
                    order=1,
                    trend="c",
                    switching_trend=True,
                    switching_exog=[True, True],
                    switching_variance=False,
                )
                ms_results = ms_model.fit(search_reps=50)

                # Unscale
                scale_y = float(scaler_y.scale_[0])
                scale_nonren = float(scaler_x.scale_[1])
                convert_factor = scale_y / scale_nonren

                beta_nr_0 = ms_results.params.get("x2[0]", np.nan) * convert_factor
                beta_nr_1 = ms_results.params.get("x2[1]", np.nan) * convert_factor
                se_nr_0 = ms_results.bse.get("x2[0]", np.nan) * convert_factor
                se_nr_1 = ms_results.bse.get("x2[1]", np.nan) * convert_factor

                # Sort high vs low MEF
                if beta_nr_0 > beta_nr_1:
                    high_mef, low_mef = beta_nr_0, beta_nr_1
                    high_se, low_se = se_nr_0, se_nr_1
                else:
                    high_mef, low_mef = beta_nr_1, beta_nr_0
                    high_se, low_se = se_nr_1, se_nr_0
            except Exception as e:
                print(f"  -> MSM failed for {group}: {e}")
            
            results.append(
                {
                    "Year": group,
                    "year": group,
                    "OLS_MEF": ols_mef,
                    "OLS_SE": ols_se,
                    "MS_High_MEF": high_mef,
                    "MS_High_SE": high_se,
                    "MS_Low_MEF": low_mef,
                    "MS_Low_SE": low_se,
                    "Avg_Emissions": avg_em,
                }
            )

        res_df = pd.DataFrame(results)
        print("\nMSM Results:")
        print(res_df)
        return res_df

    def plot_mef_by_year(self, results_df, include_markov=True, legend=False, save_path="images_msm"):
        """
        Plot MEF estimates by year in a single plot:
        MSM Estimates vs Average.
        Fits linear regression trends.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pathlib import Path
        import os
        
        region_name = Path(self.data_path).stem.split("_")[0]
        
        self.set_publication_style()
        
        # Set strict physical dimensions (in inches) to match plot_generation_mix
        plot_size = 6.0
        plot_width = plot_size
        plot_height = plot_size
        
        left_margin = 1.0
        bottom_margin = 0.8
        top_margin = 0.6
        right_margin = 2.0 if legend else 0.5
        
        fig_width = left_margin + plot_width + right_margin
        fig_height = bottom_margin + plot_height + top_margin
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_axes([
            left_margin / fig_width,
            bottom_margin / fig_height,
            plot_width / fig_width,
            plot_height / fig_height
        ])
        
        year_col = 'year' if 'year' in results_df.columns else 'Year'
        years = results_df[year_col].astype(int)
        
        # Helper function to plot Data Points + Error Bars + Regression Trend
        def add_trend(ax, x, y, yerr, label, color, marker):
            # 1. Plot the Regression Line with CI (using Seaborn)
            sns.regplot(x=x, y=y, ax=ax, scatter=False, color=color, 
                        label=f'{label} (Trend)', ci=95, line_kws={'linestyle': '--', 'alpha': 0.8})
            
            # 2. Plot the specific Yearly Points with their Standard Errors
            ax.errorbar(x, y, yerr=1.96*yerr, fmt=marker, label=f'{label} (Obs)', 
                        markersize=8, capsize=5, capthick=2, linewidth=0, color=color, alpha=0.9)

        # ==========================================
        # PLOT: MSM vs Average
        # ==========================================
        ax.set_title(f"{region_name}", loc="center", pad=15, fontweight='bold', fontsize=24)
        
        if 'Avg_Emissions' in results_df.columns:
            add_trend(ax, years, results_df['Avg_Emissions'], 0, 'Average', 'pink', 'D')

        if include_markov and 'MS_High_MEF' in results_df.columns:
            add_trend(ax, years, results_df['MS_High_MEF'], results_df['MS_High_SE'], 'MS-HIGH', 'blue', 'd')
            add_trend(ax, years, results_df['MS_Low_MEF'], results_df['MS_Low_SE'], 'MS-LOW', 'purple', 'v')

        # ==========================================
        # FORMATTING
        # ==========================================
        ax.set_xlabel('Year', fontsize=22)
        ax.set_xticks(years)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(False)
        
        if legend:
            ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.9)

        ax.set_ylabel('Marginal Emissions (lbs/kWh)', fontsize=22)
        ax.set_ylim(0.4, 1.8)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f"{save_path}/{region_name}_MEF.eps", bbox_inches='tight')
            print(f"Saved MEF plot to {save_path}/{region_name}_MEF.eps")
            fig.savefig(f"{save_path}/{region_name}_MEF.svg", bbox_inches='tight')
            print(f"Saved MEF plot to {save_path}/{region_name}_MEF.svg")
            
        plt.show()
    
    def plot_generation_mix(self, show_legend=False, smooth_days=14, save_path="images_msm"):
        """
        Plots a smoothed stacked area chart of generation mix as a square.
        
        Parameters:
        - show_legend: bool, whether to display the legend to the right of the plot
        - smooth_days: int, the number of days to use for the rolling average to smooth the curves
        """
        from pathlib import Path
        import matplotlib.pyplot as plt
        import os

        self.set_publication_style()

        # 1. Extract region name from the file path
        # E.g., 'data/MISO_2021.csv' -> 'MISO'
        region_name = Path(self.data_path).stem.split("_")[0]
        df = self.data.copy()
        df.set_index("Date", inplace=True)

        # 2. Set strict physical dimensions (in inches) to make it a SQUARE
        plot_size = 6.0
        plot_width = plot_size
        plot_height = plot_size

        # Define static vertical/left margins
        left_margin = 1.0
        bottom_margin = 0.8
        top_margin = 0.6

        # 3. Dynamically adjust the RIGHT margin based on the legend
        right_margin = 2.0 if show_legend else 0.5

        # Calculate total figure size
        fig_width = left_margin + plot_width + right_margin
        fig_height = bottom_margin + plot_height + top_margin

        fig = plt.figure(figsize=(fig_width, fig_height))

        # 4. Add the axes using explicit fractional coordinates [left, bottom, width, height]
        # This guarantees the inner plot is exactly 6x6 inches, no matter what.
        ax = fig.add_axes(
            [
                left_margin / fig_width,
                bottom_margin / fig_height,
                plot_width / fig_width,
                plot_height / fig_height,
            ]
        )

        Gen_Cols = [
            "Gen_Coal",
            "Gen_Gas",
            "Gen_Nuclear",
            "Gen_Wind",
            "Gen_Sun",
            "Gen_Others",
            "Gen_Oil",
        ]
        Gen_labels = ["Coal", "Gas", "Nuclear", "Wind", "Solar", "Other", "Oil"]

        # 5. Generate the smoothed stacked area plot
        x = df.index
        
        # Apply a rolling average to smooth the hourly noise
        window_size = 24 * smooth_days # Convert days to hours
        y_values = [
            (df[col] * 1000 / 1_000_000).rolling(window=window_size, min_periods=1, center=True).mean() 
            for col in Gen_Cols
        ] 

        # Define a clean, EIA-style color palette
        colors = [
            "#5A9BD5",
            "#70AD47",
            "#FFC000",
            "#A5A5A5",
            "#ED7D31",
            "#4472C4",
            "#9E480E",
        ]

        ax.stackplot(
            x, y_values, labels=Gen_labels, colors=colors[: len(Gen_Cols)], alpha=0.9
        )

        # Formatting the plot
        # Title centered with only the dataset name
        ax.set_title(f"{region_name}", loc="center", pad=15, fontweight='bold', fontsize=24)
        ax.set_ylabel("Generation (GWh)", fontsize=22)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        # Turn off the dotted grid lines
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 6. Add the legend outside to the RIGHT of the protected plot area
        if show_legend:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(
                    1.02,
                    0.5,
                ),  # Anchors legend just outside the right edge
                ncol=1,  # Single column looks much better on the right side
                frameon=False,  # Removes the box around the legend
                fontsize=20,
            )
            
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f"{save_path}/{region_name}_Gen.eps", bbox_inches='tight')
            print(f"Saved Generation Mix plot to {save_path}/{region_name}_Gen.eps")
            fig.savefig(f"{save_path}/{region_name}_Gen.svg", bbox_inches='tight')
            print(f"Saved Generation Mix plot to {save_path}/{region_name}_Gen.svg")

        plt.show()