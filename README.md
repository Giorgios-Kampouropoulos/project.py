# Numeric Dataset Visualization Tool
## A Terminal-Based Python Application for Light Dataset Editing and Visualization
This tool is designed to help users who have CSV datasets containing numerical columns (though it can handle some categorical columns too). It walks you through simple data-cleaning steps, offers multiple plotting routines for quick insight, and allows some light transformations that can be handy in preparing data for machine learning (ML) tasks.

By default, the workflow proceeds like so:

1. Loading a CSV
2. Handling NaNs or Missing Values
3. Type Conversions
4. Optionally Dropping Columns
5. Selecting a Target Variable
6. Menu-Driven Visualization
7. Optional Outlier Handling
   
Even if you’re new to data science or ML, you can experiment and see how different visual representations and outlier strategies shape your dataset. More advanced or curious users might go on to adapt or extend this application further.

## Watch the Video Explanation!
https://youtu.be/BQRN-swed3U
You can find a reference video above. You’re encouraged to check it out. Seeing the tool in action often makes concepts clear, especially if you’re less familiar with data wrangling. Although no strict ML background is required, a little knowledge can help you make the most of these features.

## The Packages Used
Because we wanted a tool that others could build on, we chose widely recognized libraries. Below are the key imports and why they matter:

* **pandas (import pandas as pd)**
  A core library for data manipulation. It supplies the convenient DataFrame structure and makes CSV import, slicing, summarizing, and type conversions straightforward.

* **numpy (import numpy as np)**
  Powers numerical computations, including averages, medians, linspace, and array-based transformations. We use it to convert Pandas extension dtypes (Int64, Float64) into standard NumPy dtypes, thereby preventing type errors in certain visualizations.

* **matplotlib.pyplot (import matplotlib.pyplot as plt)**
  Underpins basic plotting actions, letting us label axes, set titles, and manage figure windows. Seaborn’s advanced plots often rely on Matplotlib behind the scenes.

* **seaborn (import seaborn as sns)**
  Adds aesthetically pleasing plots (e.g., heatmap, pairplot, countplot). This library simplifies correlation matrix creation and many standard data visualizations. Although Plotly was an option, we stuck with Seaborn to keep things tidy.

* **warnings (import warnings)**
  Helps suppress extraneous warnings so that users see only the essential terminal messages rather than spammy caution notes.

* **pylab (import pylab)**
  Bundles multiple Matplotlib commands under a single namespace, critical for generating Q-Q plots (via stats.probplot(..., plot=pylab)).

* **sys.exit (from sys import exit)**
  Allows us to abruptly stop the program if a user continuously fails to provide valid input (for instance, never passing a .csv file).

* **scipy (from scipy import stats / from scipy.stats import zscore)**

  * **stats.probplot()** is used for generating Q-Q plots, helpful when testing for normality.
  * **zscore** is employed for outlier detection based on the (|z|>3) threshold.

* **pytest** (when writing test_project.py)
  Enables a straightforward testing framework for verifying the correctness of our code’s primary functions.

## A Closer Look at the Functions
**main()**
Drives everything from CSV input to null-value handling, data type conversion, column-dropping, target-selection, and eventually invokes the visualization menu. It’s the script’s central coordinator.

**csv_getter()**
Tries multiple times to retrieve a valid CSV file name or path from user input. If after several attempts the user still fails, it terminates the script. This prevents an infinite loop situation and clarifies potential file path errors.

**handle_nulls()**
Launches a mini-menu that asks how you’d like to deal with missing values. You can:

* Drop rows that contain null values,
* Fill numeric columns with their mean,
* Fill numeric columns with their median,
*Or do nothing at all.
After five invalid attempts, the script defaults to dropping rows with nulls. This is to protect the user from infinite prompts or confusion.

**dtype_converter()**
A helper routine that uses df.convert_dtypes(), then explicitly reassigns any “Int64” or “Float64” columns to standard numeric types. Seaborn sometimes struggles with extension dtypes, so this extra step ensures smooth plotting. Any remaining string columns are converted to categories.

**drop_label(df)**
Displays all column indices alongside column labels, letting you enter which columns to remove from the dataset. This is beneficial for removing irrelevant or highly problematic columns early in the process. If you don’t wish to drop anything, simply press Enter.

**get_target(rdf)**
Prompts the user to choose which column index will serve as the target. This is especially relevant for ML contexts or if certain plots (like the pair plot) need a categorical feature for coloring.

**visualize(rdf, target)**
Kicks off the main menu for plotting. Users can:
*Generate a correlation matrix,
*Show bar/histogram distributions,
*Draw box plots,
*Produce pair plots,
*Look at Q-Q plots,
*Display distribution plots,
*And handle outliers.
  Everything is guided by a color-coded, text-based menu system.
  
**bar_distribution_submenu()**
Houses two bar/histogram sub-options:
1. Chart the target across a specific feature.
2. Chart a single feature’s distribution across the entire dataset.
   If the user picks the first option and the target is categorical, we’ll see a countplot. If it’s numeric, we show average target values by category.

**get_column_choice(rdf, prompt_text)**
Lets the user pick a column by index rather than by name. This function is used in multiple places like get_target(), bar_distribution_submenu(), and so on. It will keep asking until a valid integer index is provided.

**outlier_menu(rdf)**
A simple menu that asks which outlier-removal strategy you’d like for numeric columns:

* IQR-based (1.5× rule),
* Z-score-based (|z|>3),
* Skip.
  Whichever strategy is chosen, it’s applied to the dataset, and that updated DataFrame is then used for further steps.

**remove_outliers_iqr(rdf, numeric_cols)**
Filters out rows that exceed Q1 - 1.5×IQR or Q3 + 1.5×IQR. This is a common technique to tame outliers without fully discarding too much data if it’s fairly distributed.

**remove_outliers_zscore(rdf, numeric_cols)**
Computes a z-score for each numeric column, removing any rows where the absolute value of z is bigger than 3. Great if you believe your data is roughly normal.

**plot_correlation_matrix(rdf)**
Generates a correlation heatmap (using Seaborn) for all numeric columns. It’s handy for spotting which variables move in tandem.

**plot_bar_distribution(rdf, feat_col, target, fig_title="Bar Chart")**
Draws a bar chart relating a selected feature and the user’s target variable. If the target is categorical, you get a countplot with different hues; if numeric, you get mean values by category.

**plot_single_feat_distribution(rdf, feat_col, fig_title="Single Feature Distribution")**
Displays a countplot for one chosen feature. This is simpler than a target-based distribution but remains useful for quick checks on how data is spread across categories.

**plot_boxplot(rdf, col, fig_title="Box Plot")**
Shows a box plot for a single numeric column, revealing its quartiles and potential outliers at a glance.

**plot_pairplot(rdf, target=None)**
Creates an all-in-one grid of numeric variables. If the target column is recognized, hue coloring is applied. This can illuminate interactions among multiple features and highlight clusters or patterns in your data.

**plot_qq(rdf, col, fig_title="Q-Q Plot")**
Leverages stats.probplot(...) to examine if a numeric column is normally distributed. Typically, points on a Q-Q plot line up more closely with the diagonal if the data is normal.

**plot_dist(rdf, col, fig_title="Distribution Plot")**
Shows a basic histogram with a KDE overlay for a single column. Useful for checking skewness, multi-modality, or any unusual distribution shape.

## How to Install the Numeric Dataset Visualization Tool

1. **Clone or Download**
    Acquire the project folder from wherever it’s hosted (e.g., GitHub). Make sure all core files are in one place: project.py, test_project.py, requirements.txt, and any sample CSV you’d like to test.

2. **Install Dependencies**
    From the same directory, run:
      pip install -r requirements.txt
    This ensures you have pandas, numpy, matplotlib, seaborn, scipy, and pytest.

3. **Run the Script**
    Invoke:
      python project.py
    Then follow the on-screen instructions. You’ll be prompted for a CSV file path or name, how you want to handle nulls, which columns to drop, which one’s your target, etc. After that, a menu of visualization and outlier-removal options is at your disposal.

4. **Optionally Test with Pytest**
    If you wish to confirm the code’s basic correctness, run:
      pytest
    This will check that the project’s core functions behave as expected (like reading CSV paths or dropping columns).

## Known Issues
* **Non-Numeric Datasets:** This tool is primarily for numeric data. While some categorical columns are supported (e.g., bar plots), a dataset full of strings or mixed types might not yield all plots successfully.
* **Plot Variety:** Although you have many visualization options, there is always more that could be done. Feel free to fork or add more specialized plots or routines.

## Notes for Contributors
Go ahead and break this project—chances are you’ll find edge cases that haven’t been addressed. A few areas ripe for improvement include:

* Adding more advanced visualization methods
* Providing educational commentary on each step’s statistical or ML significance
* Offering an ML model training menu
* Integrating new ways of handling missing or categorical data

Wherever you choose to take this project, all help is welcome. Thanks for reading, and enjoy exploring your data!
