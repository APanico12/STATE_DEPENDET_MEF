import pandas as pd
import numpy as np
import statsmodels.formula.api as ols

class MEFDecomposition:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_and_prep_data(self, start_date='2019-01-01', end_date='2022-12-31'):
        # 1. Load Data
        df = pd.read_excel('Region_US48.xlsx')
        df.columns = df.columns.str.strip()
        
        # 2. Parse Dates
        df['date_hour'] = pd.to_datetime(df['UTC time'])
        df['date'] = df['date_hour'].dt.date
        df['year'] = df['date_hour'].dt.year
        df['month'] = df['date_hour'].dt.month
        df['hour'] = df['date_hour'].dt.hour
        
        # 3. Filter Date Range
        df = df[
            (df['date'] >= pd.to_datetime(start_date).date()) & 
            (df['date'] <= pd.to_datetime(end_date).date())
        ].copy()

        # 4. Create Time Trend (t=0, 1, 2...) for the interaction term
        # The paper uses Year_t starting at 0
        min_year = df['year'].min()
        df['time_trend'] = df['year'] - min_year

        # 5. Rename Columns for easier formulas
        # We need: Load (D), Coal Gen (NG: COL), Gas Gen (NG: NG)
        rename_map = {
            'Demand': 'Load',
            'NG: COL': 'Gen_Coal',
            'NG: NG': 'Gen_Gas',
            'CO2 Factor: COL': 'EF_Coal', # Emission Factor (tons/MWh)
            'CO2 Factor: NG': 'EF_Gas'
        }
        df = df.rename(columns=rename_map)
        
        # Clean data (remove rows where Gen_Coal or Gen_Gas is NaN)
        self.data = df.dropna(subset=['Load', 'Gen_Coal', 'Gen_Gas'])
        print(f"✅ Data loaded: {len(self.data)} observations")

    def estimate_probability_change(self):
        """
        Estimates Δp (change in probability) using Equation 2 logic:
        Gen_fuel = β*Load + γ*(Load * Year) + FixedEffects
        The coefficient γ (Load:time_trend) is the annual change in probability.
        """
        results = {}
        
        # We use C(month)*C(hour) to control for seasonality/daily patterns (Fixed Effects)
        # We interpret 'time_trend' as the Year variable
        formula_base = " ~ Load + Load:time_trend + C(month):C(hour) + time_trend"
        
        for fuel in ['Gen_Coal', 'Gen_Gas']:
            print(f"⏳ Running regression for {fuel} probability...")
            formula = fuel + formula_base
            
            # Using basic OLS (You can use cov_hac for robust errors if needed)
            model = ols.ols(formula, data=self.data).fit()
            
            # The coefficient of interest is 'Load:time_trend'
            # It tells us: "How much does the response to Load change per year?"
            delta_p = model.params['Load:time_trend']
            results[fuel] = delta_p
            print(f"   🔹 Δp (Annual change in probability) for {fuel}: {delta_p:.5f}")
            
        return results

    def calculate_decomposition(self, delta_p_results):
        """
        Calculates the Probability Effect:
        Effect = β_c * Δp_c + β_g * Δp_g
        """
        # 1. Get Average Marginal Emission Factors (β_c, β_g)
        # The paper uses average emissions of unconstrained units.
        # We will approximate this with the average CO2 Factor from the data.
        # Note: Factor is usually in metric tons/MWh.
        
        beta_c = self.data['EF_Coal'].mean()
        beta_g = self.data['EF_Gas'].mean()
        
        dp_c = delta_p_results['Gen_Coal']
        dp_g = delta_p_results['Gen_Gas']
        
        print("\n📊 DECOMPOSITION RESULTS")
        print("-" * 40)
        print(f"Average Emission Factors (Approx. β):")
        print(f"  Coal (β_c): {beta_c:.3f} tons/MWh")
        print(f"  Gas  (β_g): {beta_g:.3f} tons/MWh")
        print("-" * 40)
        print(f"Observed Probability Shift (Δp per year):")
        print(f"  Coal (Δp_c): {dp_c:.5f}")
        print(f"  Gas  (Δp_g): {dp_g:.5f}")
        print("-" * 40)
        
        # 2. Calculate the Net Effect
        # Formula: (β_c * Δp_c) + (β_g * Δp_g)
        effect_coal = beta_c * dp_c
        effect_gas = beta_g * dp_g
        total_probability_effect = effect_coal + effect_gas
        
        print(f"Calculated Probability Effect on Marginal Emissions:")
        print(f"  Due to Coal: {effect_coal:.5f}")
        print(f"  Due to Gas:  {effect_gas:.5f}")
        print(f"  TOTAL ANNUAL CHANGE: {total_probability_effect:.5f} tons/MWh per year")
        
        if total_probability_effect > 0:
            print("\n👉 Conclusion: Marginal emissions are INCREASING due to fuel switching.")
        else:
            print("\n👉 Conclusion: Marginal emissions are DECREASING due to fuel switching.")

# --- usage ---
decomposition = MEFDecomposition('Region_US48.xlsx - Published Hourly Data.csv')
decomposition.load_and_prep_data(start_date='2019-01-01', end_date='2022-12-31')
# delta_p = decomposition.estimate_probability_change()
# decomposition.calculate_decomposition(delta_p)

