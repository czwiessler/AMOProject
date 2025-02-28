# AMOProject

## Project Overview
This repository contains Jupyter notebooks and accompanying data which is first used for Monte Carlo Simulations and then for our optimizsation.

## Contents
- Simulations
    > **Note:**  
    Each notebook contains the complete procedure, from creating the CDF lookup to the final simulation of the respective variable.  
    Since we have three correlated variables, the individual simulations (every section marked with `Appendix`) serve only as an appendix and are not used later.  
    Instead, we only utilize the CDF lookups from these notebooks and perform the actual simulation in the *Random Variables* notebook.

  - `01_MonteCarloSimulationSun.ipynb`: Jupyter notebook for CDF lookup and simulation of solar radiation.
  - `02_MonteCarloSimulationTemperature.ipynb`: Jupyter notebook for CDF lookup and simulation of ambient temperature.
  - `03_MonteCarloSimulationConsumption.ipynb`: Jupyter notebook for CDF lookup and simulation of energy consumption.
- `04_RandomVariables.ipynb`: Jupyter notebook for generating samples that are correlated using Gaussian Copula and Monte Carlo Simulation, to ensure that the marginal distributions correspond to the specified CDFs (as quantiles). In addition, this also has code to cluster the simulation using k-means and saves them in the `data/` folder.
- `05_Optimization.ipynb`: Jupyter notebook with bi-level optimization Julia code.

- #### 📂 `data/`
    Directory containing the input data files required for the simulations. **Note:** Since the consumption data exceeded the git file size limit of 100 MB, we provided this in a zip. In order to reproduce our results you must unzip it.

    - ##### 📂 `results/`
        Directory containing the output results generated by the simulations (1000 for each hour of a month - but not for each day seperate) which we then clustered using k-means.

    - ##### 📂`saved_csv/`
        Directory containing the CDF Functions and representative daily profiles for  all months for Temperature, Sun and Consumption.

- #### 📂 `figures/`
    Directory containing plots, created in Julia.

## Prerequisites
Before running the notebooks, ensure you have the following installed:
- Python 3.9
- Jupyter Notebook
- Required libraries from the requirements.txt.

You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
