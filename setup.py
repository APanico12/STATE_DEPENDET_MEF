from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RRMSW_MEF",
    version="1.0.0",
    author="A. Panico",
    description="US Marginal Emission Factor Analysis with Markov Switching Regimes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/APanico12/RRMSW_MEF",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
        "arch>=4.18.0",
        "patsy>=0.5.2",
    ],
)
