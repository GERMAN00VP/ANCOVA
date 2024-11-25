# README for ANCOVA Analysis Script

## Overview

This script, provides tools for performing ANCOVA (Analysis of Covariance) and related statistical analyses. It includes a primary function, `do_ancova`, which integrates multiple steps of ANCOVA analysis and allows for flexible customization of inputs and outputs, including graphical representations of results.

---

## Key Functionality: `do_ancova`

The main purpose of the `do_ancova` function is to perform **parametric** or **non-parametric ANCOVA** on a dataset. It accepts a DataFrame containing the dependent variable, categorical variables, and covariates to evaluate the relationship between them while adjusting for covariates.

### Features:

- **Parametric and Non-Parametric ANCOVA**:  
  Automatically switches between parametric or ranked (non-parametric) ANCOVA depending on the assumptions of normality and homoscedasticity.

- **Interaction Effects**:  
  Allows inclusion of interactions between variables.

- **Post-Hoc Analysis**:  
  Automatically performs Tukey or Dunn post-hoc tests when significant differences are found between groups.

- **Data Visualization**:  
  Generates boxplots and scatterplots with regression lines, including statistical significance indicators.

- **Customizable Options**:  
  Users can customize interactions, colors, and plot details.

---

## Usage: `do_ancova`

### Parameters:

- **`data`**:  
  A pandas DataFrame containing:  
  - **Column 1**: Dependent (response) variable.  
  - **Column 2 (to n categories)**: Categorical independent variable(s).  
  - **Remaining columns**: Continuous covariates.

- **`interactions`** *(Optional)*:  
  Specifies interactions between variables:  
  - `"ALL"`: Includes all interactions.  
  - `list`: List of tuples specifying interacting variables.

- **`plot`** *(Default: False)*:  
  If `True`, generates a regression plot and a boxplot.

- **`save_plot`** *(Default: False)*:  
  If provided with a file path, saves the generated plots to the specified location.

- **`covariate_to_plot`** *(Optional)*:  
  Specifies the covariate to display in plots.

- **`palette`** *(Optional)*:  
  A dictionary mapping categorical levels to colors.

- **`categories`** *(Default: 1)*:  
  Number of categorical variables.

- **`ax`** *(Optional)*:  
  A Matplotlib axis for custom plotting.

- **`y_lab`** *(Optional)*:
        Label for the y-axis in the generated plot. Default is False (no label).

- **`x_lab`** *(Optional)*:
        Label for the x-axis in the generated plot. Default is False (no label).

- **`sum_of_squares_type`** *(Optional)*: 
        Specifies the type of sums of squares for ANCOVA. Default is Type 2 (value = 2).


  ### Output:

1. **Results**:  
   - A summary dictionary with the ANCOVA parameters and outcomes.  
   - An ANOVA table with p-values for each effect.  
   - Post-hoc results (if applicable).

2. **Plots**:  
   - Scatterplot with regression lines for covariates + Boxplot for main categorical copmpaisons.  
   - A Matplotlib axis with a Boxplot for categorical comparisons (allows customizing).

3. **Files (Optional)**:  
   Saves plots to the specified file path if `save_plot` is provided.


### Dependencies

The script relies on the following Python packages:

- `numpy`  
- `pandas`  
- `statsmodels`  
- `scipy`  
- `seaborn`  
- `matplotlib`  
- `scikit_posthocs`  

Install these dependencies using:  

```bash
pip install numpy pandas statsmodels scipy seaborn matplotlib scikit-posthocs
```

### Notes

- Ensure that your dataset has the shape: Cases*Variables. 
- The script assumes the columns are sorted like this: [Response variable, Main category to compare, Other categorical co-variables (optional), Other continous co-variables].
- For multiple categorical variables, specify the number using the categories parameter.

## AN EXAMPLE OF USE:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charge the main function from our package
from Ancova_analysis import do_ancova

```
## This invented dataset contains 150 entries with the following columns:

- **Number of T Cells**: The number of T cells, which is affected by the individual's age and HIV status. Individuals with HIV+ (Untreated) have a significant reduction in T cells, while HIV+ (TAR Treatment) individuals have a minimal reduction compared to HIV- individuals.

- **HIV Status**: A categorical variable representing the individual's HIV status. It can take three values:

        -> HIV- (no HIV)

        -> HIV+ (TAR Treatment) (HIV positive, receiving treatment)

        -> HIV+ (Untreated) (HIV positive, not receiving treatment)
    
- **Sex**: The individual's sex, either Male or Female.

- **Age**: The individual's age, ranging from 20 to 70 years.

The Number of T Cells decreases with age, and the reduction is more significant for individuals with HIV+ (Untreated).

**Lets see if the package is able to capture this differences:**


```python
# Set the seed for reproducibility
np.random.seed(4)

# Number of samples
n = 150

# Categorical variables
sex = np.random.choice(['Male', 'Female'], size=n)
hiv_status = np.random.choice(['HIV-', 'HIV+ (TAR Treatment)', 'HIV+ (Untreated)'], size=n, p=[0.4, 0.3, 0.3])

# Covariate: Age
age = np.random.randint(20, 70, size=n)

# Generate T cell count
t_cells = []
for i in range(n):
    base_t_cells = 1000  # General base for T cells
    age_effect = -3 * (age[i] - 30)  # Mild effect of age
    if hiv_status[i] == 'HIV+ (Untreated)':
        hiv_effect = -200  # Significant reduction for untreated
    elif hiv_status[i] == 'HIV+ (TAR Treatment)':
        hiv_effect = -30  # Minimal reduction for treated
    else:
        hiv_effect = 0  # No effect for HIV-
    noise = np.random.normal(0, 50)  # Random noise
    t_cells.append(base_t_cells + age_effect + hiv_effect + noise)

# Define a palette to select the plotting colors for each category, else it would be randomly assigned
palette = {"HIV-":"skyblue",
           "HIV+ (Untreated)":"salmon",
           "HIV+ (TAR Treatment)":"orange"}


# Create the DataFrame
data_hiv = pd.DataFrame({
    'Number of T Cells': np.round(t_cells).astype(int),
    'HIV Status': hiv_status,
    'Sex': sex,
    'Age': age
})

data_hiv.head()

```

```python 

# Run the main function and display the results

df_results, ancova_summary,post_hoc = do_ancova(data=data_hiv,
                                                palette=palette,
                                                categories=2, # HIV Status and Sex
                                                interactions=[('HIV Status',"Age")], # Test the significance of the interaction of these variables
                                                y_lab="CD4 T Cells (count)",# Set the y_label 
                                                plot=True, # Create the plot
                                                save_plot= "./Images/ANCOVA_Regression_boxplot.png" # Sves the plot in that path
                                                ) 

display(df_results)
display(ancova_summary)
display(post_hoc)

```
![Example Plot](./Images/ANCOVA_Regression_boxplot.png)

```python
# Create two subplots in a row
fig, axs = plt.subplots(ncols=2,figsize=(12,6))


df_results, ancova_summary,post_hoc,ax= do_ancova(data=data_hiv,palette=palette,categories=2, y_lab="CD4 T Cells (count)",plot=True,
          ax=axs[0] # When the axis is provided it returns the boxplot and can be integrated with other subplots as you wish
          )

# Modify the df order to plot the sex differences
data_hiv_sex = data_hiv[['Number of T Cells','Sex','HIV Status','Age']]

df_results, ancova_summary,post_hoc,ax= do_ancova(data=data_hiv_sex,categories=2, y_lab="CD4 T Cells (count)",plot=True,
          ax=axs[1], # The other subplot

          )
# Save and show
plt.savefig("./Images/ANCOVA_two_boxplots.png",bbox_inches="tight")
plt.show()

```

![Example Plot 2](./Images/ANCOVA_two_boxplots.png)
