# RRMSW_MEF - US Marginal Emission Factor Analysis

Advanced time-series analysis of US Marginal Emission Factors (MEF) using Markov Switching Regimes and Robust Change Point Detection.

## Overview

This repository implements robust methodologies for analyzing and estimating Marginal Emission Factors (MEF) for the US electricity grid (Lower 48 states) using:

- **Markov Switching Models (MSM)** - To identify and analyze different emissions regimes
- **Dummy Variable Seasonality Extraction (DVSE)** 
- **Robust Change Point Detection (RobustCPD)** - Based on Riani et al. (2019) approach
- **Time Series Analysis** - Including stationarity tests, linearity tests, and ARIMA modeling

## Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate rrmsw-mef
```

### From source
```bash
pip install -e .
```

## Project Structure

```
RRMSW_MEF/
├── main.ipynb                    # Primary analysis notebook for yearly MEFs
├── mainlight.ipynb               # Lightweight analysis notebook for hourly and State Dependet MEFs
├── Gas_Coal.ipynb                # Gas vs Coal comparative analysis
├── Gas_CPD.ipynb                 # Change point detection for gas data
├── DDD.ipynb                     # Triple Differences quasi-expariment for gas coal generation (do not mentioned in the paper)
├── USMEFAnalysis.py              # Main analysis class
├── LightMEF.py                   # MEF analysis class for hourly and State Dependet MEFs
├── RobustCPD.py                  # Robust change point detection
├── plot_style.py                 # Publication-quality plotting utilities
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment file
└── setup.py                      # Package setup
```

## Quick Start

### Basic Usage

```python
from USMEFAnalysis import USMEFAnalysis

# Initialize the analysis
us_analysis = USMEFAnalysis("Region_US48.xlsx")

# Load and prepare data
us_analysis.load_and_clean_data()
us_analysis.create_time_variables()

# Perform seasonal adjustment
result = us_analysis.seasonal_adjustment_iteraction()

# Generate diagnostics
us_analysis.plot_time_series()
us_analysis.plot_annual_diagnostics_dynamic()

# Run statistical tests
df_test = us_analysis.run_uniroot_tests()
df_nonlinearity = us_analysis.run_nonlinearity_tests_r()

# Estimate MEF
result_df = us_analysis.mef_estimate()
us_analysis.plot_mef_by_year(result_df)
```

### Using LightMEF

For a simplified analysis workflow:

```python
from LightMEF import LightMEF

# Initialize
mef = LightMEF("Region_US48.xlsx")

# Load data
mef.load_data()

# Apply publication-quality styling
mef.set_publication_style(use_latex=False)

# Perform Markov Switching estimation
mef.estimate_markov_switching(exog_vars=['seasonal_dummies'])

# Generate plots
mef.plot_regimes()
```

### Change Point Detection

```python
from RobustCPD import RobustCPD

# Initialize detector
cpd = RobustCPD(trimming=0.10, alpha=0.01)

# Fit model with polynomial trend
cpd.fit(y_series, poly_order=0)  # Linear trend
# or
cpd.fit(y_series, poly_order=2)  # Quadratic trend

# Access results
print(f"Change point at index: {cpd.results['break_point_index']}")
print(f"Test statistic: {cpd.results['test_statistic']}")
```

## Data Requirements

The analysis expects time-series data with:
- **Hourly or daily frequency** emissions data
- **Time index** (datetime format)
- **Regional identifiers** (for multi-region analysis)
- **Temporal coverage**: 2019-2025 recommended

### Data Format
Input Excel file should contain:
- Date/Time column
- Emissions measurements
- Optional: Load, temperature, or other covariates

## Main Classes

### USMEFAnalysis
Comprehensive US MEF analysis with full diagnostic suite.

**Key Methods:**
- `load_and_clean_data()` - Data loading and quality checks
- `create_time_variables()` - Generate time-based features
- `seasonal_adjustment_iteraction()` - DVSE implementation
- `run_uniroot_tests()` - Unit root testing (ADF, KPSS, PP, DFGLS, BDS)
- `run_nonlinearity_tests_r()` - Nonlinearity diagnostics
- `mef_estimate()` - Markov Switching estimation
- `plot_*()` - Comprehensive visualization methods

### LightMEF
Lightweight implementation for rapid analysis and prototyping.

**Key Features:**
- Markov Switching Regimes detection
- Automated seasonal adjustment
- Publication-ready plots
- Parallel processing support

### RobustCPD
Robust Change Point Detection following Riani et al. (2019).

**Features:**
- Automatic outlier handling
- Polynomial trend support (linear, quadratic, high-order)
- Statistical significance testing
- Non-parametric approach

## Analysis Workflow

1. **Data Preparation**
   - Load data from Excel/CSV
   - Handle missing values
   - Create time features (month, hour, season, etc.)

2. **Exploratory Analysis**
   - Plot time series
   - Summary statistics by season
   - Visual regime identification

3. **Diagnostic Testing**
   - Unit root tests (ADF, KPSS, Phillips-Perron, DFGLS, BDS)
   - Nonlinearity tests
   - Model comparison

4. **Model Estimation**
   - Markov Switching Regime estimation
   - Coefficient extraction
   - Regime probability calculation

5. **Visualization & Reporting**
   - Temporal MEF plots by year
   - Regime diagnostic plots
   - Publication-quality figures



## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scipy**: Scientific functions
- **matplotlib & seaborn**: Visualization
- **statsmodels**: Time series and econometric models
- **scikit-learn**: Machine learning utilities
- **arch**: Unit root testing (DFGLS, Phillips-Perron)
- **joblib**: Parallel processing

See `requirements.txt` for exact versions.

## References
- Riani, M., et al. (2019). *Robust Change Point Detection with Polynomial Trends*
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series*

## Usage Examples

### Example 1: Complete Analysis Pipeline
See `main.ipynb` for a full working example with all features.

### Example 2: Lightweight Analysis
See `mainlight.ipynb` for a streamlined approach.

### Example 3: Comparative Analysis
See `Gas_Coal.ipynb` for multi-commodity analysis.

### Example 4: Change Point Analysis
See `Gas_CPD.ipynb` for detailed change point detection examples.

## Output Files

Analysis typically generates:
- Time series plots
- Regime diagnostic plots
- Statistical test results (DataFrames)
- MEF estimates by year/season
- Publication-ready figures (high DPI PNG)

## Contributing

For bug reports or feature suggestions, please open an issue in the repository.

## License

See LICENSE file for details.

## Author

A. Panico

## Citation

If you use this code in your research, please cite as:

```bibtex
@software{panico2024rrmsw,
  title={RRMSW_MEF: Robust Regime Markov Switching for Marginal Emission Factor Analysis},
  author={Panico, A.},
  year={2024},
  url={https://github.com/APanico12/RRMSW_MEF}
}
```

## Support

For questions or issues:
1. Check existing notebooks for examples
2. Review inline code documentation
3. Refer to method docstrings in source files
