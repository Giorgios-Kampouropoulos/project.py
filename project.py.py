import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pylab
from scipy import stats
from scipy.stats import zscore

warnings.filterwarnings("ignore")


def main():
    """
    The main function:
    1. Retrieves a CSV from the user (or a default path).
    2. Asks how to handle nulls and applies that choice.
    3. Converts data types.
    4. Lets user drop columns.
    5. Lets user choose a target column.
    6. Finally, calls a menu for data visualization,
       which also includes outlier-handling options.
    """
    # Read raw CSV first
    df = pd.read_csv(csv_getter())
    
    # Ask about null handling
    df = handle_nulls(df)

    # Prompt user about Dataframe properties
    df = dtype_converter(df)
    reduced_df = drop_label(df)
    target = get_target(reduced_df)

    print(reduced_df.describe())
    print(f"\nThe target value is: '{target}'\n")

    # Launch our menu-based "GUI"
    visualize(reduced_df, target)


def csv_getter():
    """
    Retrieves a CSV file path. For demonstration, we return a hard-coded
    CSV file name. In a more interactive scenario, you'd prompt the user.

    Returns:
        str: Path to the CSV file
    """
    file = "crop_yield_data.csv"
    return file


def handle_nulls(df):
    """
    Asks the user how to handle null values (NaN) in the DataFrame.
    Options:
      1. Drop all rows with NaN
      2. Fill numeric columns with mean
      3. Fill numeric columns with median
      4. Skip (do nothing)

    Args:
        df (pd.DataFrame): DataFrame with potential null values.

    Returns:
        pd.DataFrame: Updated DataFrame after chosen null-handling strategy.
    """
    # Make a loop for the user prompt in case the user doesn't understand the task
    for i in range(5):
        # Launch a menu with options
        print("\n\033[95mHow do you want to handle null values?\033[0m")
        print("  1. Drop all rows that contain null values")
        print("  2. Fill numeric columns with mean (skip categorical columns)")
        print("  3. Fill numeric columns with median (skip categorical columns)")
        print("  4. Skip (do nothing)")
        choice = input("Please select one option (1-4): \n").strip()

        if choice == "1":
            print("Dropping all rows with any null values...")
            df = df.dropna()
            return df
        elif choice == "2":
            print("Filling null values in numeric columns with mean...")
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
            return df
        elif choice == "3":
            print("Filling null values in numeric columns with median...")
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            return df
        elif choice == "4":
            print("No null-handling applied.")
            return df
        else:
            if i!=4:
                print("Invalid choice. Please pick 1, 2, 3, or 4.")
            else:
                print("Invalid choice. Rows with null values are now going to be removed.")
                df = df.dropna()
                return df



def dtype_converter(df):
    """
    Converts DataFrame columns to more manageable datatypes:
    - Numeric columns become either int or float
    - String columns become 'category'
    """
    conv_df = df.convert_dtypes()
    
    # Convert all "Int64" or "Float64" (extension dtypes) to standard float or int, because seaborn struggles with these dtypes
    for col in conv_df.columns:
        # If it's the nullable integer dtype
        if str(conv_df[col].dtype) == "Int64":
            # Convert to a standard int64
            # If you have nulls, int cast will fail, so float is safer
            conv_df[col] = conv_df[col].astype("float64")
        
        # If it's the nullable float dtype
        elif str(conv_df[col].dtype) == "Float64":
            # Convert to standard float64
            conv_df[col] = conv_df[col].astype("float64")

    # Convert remaining string columns to category
    str_col = conv_df.select_dtypes("string").columns.to_list()
    for label in str_col:
        conv_df[label] = conv_df[label].astype("category")

    return conv_df



def drop_label(df):
    """
    Lets the user drop columns by specifying their indices (displayed to them). In order to facilate
    UX, a dict is made, containing indexes corresponding labels, so that the user only has to input the 
    index of the column to be dropped.

    Args:
        df (pd.DataFrame): The initial dataset.

    Returns:
        tuple: (reduced DataFrame, labels_dict of original columns)
    """
    labels = df.columns
    print("\nThese are the contents of the dataset you provided: \n")
    print(df.describe(), end="\n\n")

    # show user the feature names with indexes
    print("The labels (feature names) in the header are the following: ")
    labels_dict = {}
    for i, label in enumerate(labels):
        print(f"{i}. {label}")
        labels_dict[i] = label

    # get the indexes of the columns to drop
    for i in range(5):
        try:
            labels_indexes = input(
                "If you want to remove any columns from the dataset, enter their corresponding indices (comma-separated). "
                "If not, just press Enter:\n"
            ).strip()
            if labels_indexes == "":
                return df
            # Convert input to a list of column labels to remove
            labels_list = [
                labels_dict[int(l.strip())] for l in labels_indexes.split(",")
            ]
            return df.drop(columns=labels_list)
        except ValueError:
            print("Your input wasn't recognized as integers separated by commas.")
        except KeyError:
            print(
                f"One or more numbers you typed are out of range (0 to {len(df.columns) - 1})."
            )

        if i == 4:
            print("No column will be dropped")
            return df


def get_target(rdf):
    """
    Lets the user choose which column index should be used as the 'target', in a machine learning sense.

    Args:
        labels_dict (dict): A dictionary mapping column index -> column name.

    Returns:
        str: Name of the chosen target column.
    """
    # A helper function was used [get_column_choice()], which will be defined and reused later in this project
    return get_column_choice(rdf, "Input the index of the header that you want to be used as a target:\n")


def visualize(rdf, target):
    """
    Presents a menu-driven interface in the terminal allowing the user to
    select different types of data visualizations, ass well as handle outliers.

    Args:
        rdf (pd.DataFrame): The reduced DataFrame after dropping columns.
        labels_dict (dict): Mapping of original column indices -> column names.
        target (str): The user-selected target column name.
    """
    # Keep a working copy that can be updated if user chooses outlier handling.
    df_in_use = (rdf)

    while True:
        #Light "GUI" Menu
        print("\n\033[91m===== Visualization Menu =====\033[0m")
        print("\033[94mPlease select an option:\033[0m")
        print("1. Correlation Matrix")
        print("2. Histogram / Bar Distribution Options (makes most sense for category-type features)")
        print("3. Box Plot")
        print("4. Pair Plot (takes time to load)")
        print("5. Q-Q Plot")
        print("6. Distribution Plot")
        print("7. Outlier Handling")
        print("8. Quit")

        choice = input("Enter your choice (1-8): ").strip()

        if choice == "1":
            plot_correlation_matrix(df_in_use)
        elif choice == "2":
            bar_distribution_submenu(df_in_use, target)
        elif choice == "3":
            col = get_column_choice(
                df_in_use, "Enter the column index for which you want a Box Plot:\n"
            )
            fig_title = input("Enter a title for your box plot: ").strip()
            plot_boxplot(df_in_use, col, fig_title)
        elif choice == "4":
            plot_pairplot(df_in_use, target)
        elif choice == "5":
            col = get_column_choice(
                df_in_use, "Enter the column index for which you want a Q-Q Plot:\n"
            )
            fig_title = input("Enter a title for your Q-Q plot: ").strip()
            plot_qq(df_in_use, col, fig_title)
        elif choice == "6":
            col = get_column_choice(
                df_in_use,
                "Enter the column index for which you want a Distribution Plot:\n",
            )
            fig_title = input("Enter a title for your distribution plot: ").strip()
            plot_dist(df_in_use, col, fig_title)
        elif choice == "7":
            # outlier handling menu
            df_in_use = outlier_menu(df_in_use)
            print("Data updated after outlier handling. You may continue visualizing.")
        elif choice == "8":
            print("Exiting visualization menu. Goodbye!")
            break
        else:
            print("Invalid choice. Please choose a number from 1 to 8.")


def bar_distribution_submenu(df_in_use, target):
    """
    A submenu for plotting bar/histogram distributions. This visualization option makes most sense for 
    features that are categorical in nature. Two sub-options:
      1) Plot distribution of target with respect to a chosen feature
      2) Plot distribution of the value of a single feature across entire dataset
    """
    while True:
        print("\n\033[96mBar Distribution Options:\033[0m")
        print("1. Bar Chart of target distribution by a specific feature")
        print(
            "2. Bar Chart of a single feature distribution over the entire dataset"
        )
        print("3. Return to previous menu")

        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            # The user chooses a feature to see how target is distributed
            feat_col = get_column_choice(
                df_in_use, "Which feature do you want on the X-axis? \n"
            )
            fig_title = input("Enter a title for your bar chart: ").strip()
            plot_bar_distribution(df_in_use, feat_col, target, fig_title)
        elif choice == "2":
            # The user chooses a single feature to see distribution across entire dataset
            feat_col = get_column_choice(
                df_in_use, "Which feature do you want to count? \n"
            )
            fig_title = input("Enter a title for your single-feature bar chart: ").strip()
            plot_single_feat_distribution(df_in_use, feat_col, fig_title)
        elif choice == "3":
            print("Returning to main menu.")
            break
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")


def get_column_choice(rdf, prompt_text):
    """
    Helper function for letting the user pick a column from the dataset by index.

    Args:
        rdf (pd.DataFrame): The DataFrame from which columns are listed.
        prompt_text (str): The prompt message asking for the user's input.

    Returns:
        str: The column name corresponding to the user's choice.
    """
    print("Available columns:")
    for idx, col_name in enumerate(rdf.columns):
        print(f"{idx}: {col_name}")

    while True:
        user_in = input(prompt_text).strip()
        try:
            col_idx = int(user_in)
            if 0 <= col_idx < len(rdf.columns):
                return rdf.columns[col_idx]
            else:
                print(f"Please enter an index between 0 and {len(rdf.columns)-1}.")
        except ValueError:
            print("Please enter a valid integer for the column index.")


def outlier_menu(rdf):
    """
    Presents the user with multiple strategies for outlier handling:
      1. IQR-based removal
      2. Z-score-based removal
      3. Skip (do nothing)

    Args:
        rdf (pd.DataFrame): Current DataFrame in use.

    Returns:
        pd.DataFrame: Updated DataFrame after chosen outlier-handling strategy.
    """

    numeric_cols = rdf.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found for outlier removal.")
        return rdf

    while True:
        print("\n\033[96mHow do you want to handle outliers?\033[0m")
        print("1. IQR-based removal (1.5x)")
        print("2. Z-score-based removal (|z|>3)")
        print("3. Skip (do nothing)")
        choice = input("Please select one option (1-3): ").strip()

        if choice == "1":
            rdf = remove_outliers_iqr(rdf, numeric_cols)
            print("Outliers removed using IQR method (1.5x).")
            return rdf
        elif choice == "2":
            rdf = remove_outliers_zscore(rdf, numeric_cols)
            print("Outliers removed using Z-score method (|z|>3).")
            return rdf
        elif choice == "3":
            print("Skipping outlier removal.")
            return rdf
        else:
            print("Invalid choice. Please pick 1, 2, or 3.")


def remove_outliers_iqr(rdf, numeric_cols):
    """
    Removes outliers in numeric columns by IQR-based rule (1.5x).

    Args:
        rdf (pd.DataFrame): The DataFrame to modify.
        numeric_cols (list-like): List of numeric column names.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """
    df_clean = rdf.copy()
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]
    return df_clean


def remove_outliers_zscore(rdf, numeric_cols):
    """
    Removes outliers in numeric columns by Z-score rule (|z|>3).

    Args:
        rdf (pd.DataFrame): The DataFrame to modify.
        numeric_cols (list-like): List of numeric column names.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """

    df_clean = rdf.copy()
    for col in numeric_cols:
        # compute z-scores for each row in this column
        col_zscore = zscore(df_clean[col].dropna())
        # we need to drop those rows whose |z| > 3
        non_null_index = df_clean[col].dropna().index
        drop_idx = non_null_index[np.abs(col_zscore) > 3]
        df_clean = df_clean.drop(index=drop_idx)
    return df_clean


def plot_correlation_matrix(rdf):
    """
    Creates and displays a correlation heatmap of the DataFrame.

    Args:
        rdf (pd.DataFrame): The DataFrame for which correlation is plotted.
    """
    corr = rdf.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_bar_distribution(rdf, feat_col, target, fig_title="Bar Chart"):
    import numpy as np
    """
    Plots a bar chart of the target distribution with respect to a chosen
    categorical column. If the target is also categorical, we do a countplot
    with hue=target. If the target is numeric, we do a barplot of the mean
    target value for each category of feat_col.
    """
    plt.figure()
    # Check if the target is categorical or numeric
    if pd.api.types.is_categorical_dtype(rdf[target]) or rdf[target].dtype == object:
        # We'll do a countplot with hue=target
        sns.countplot(data=rdf, x=feat_col, hue=target)
        plt.ylabel("Count")
    else:
        # We'll do a barplot (mean of target by feat_col)
        sns.barplot(data=rdf, x=feat_col, y=target, estimator=np.mean, ci=None)
        plt.ylabel(f"Mean of {target}")

    plt.title(fig_title)
    plt.xlabel(feat_col)

    # Limit x-axis to display only 10 subdivisions of the x-value range
    locs_x, label_objs_x = plt.xticks()
    total_xticks = len(label_objs_x)
    if total_xticks > 10:
        import numpy as np
        keep_indices = np.linspace(0, total_xticks - 1, 10, dtype=int)
        for i, lbl_obj in enumerate(label_objs_x):
            if i not in keep_indices:
                lbl_obj.set_visible(False)
        plt.xticks(locs_x, label_objs_x, rotation=45)

    # Limit y-axis to display only 10 subdivisions of the y-value range
    locs_y, label_objs_y = plt.yticks()
    total_yticks = len(label_objs_y)
    if total_yticks > 10:
        import numpy as np
        keep_indices_y = np.linspace(0, total_yticks - 1, 10, dtype=int)
        for i, lbl_obj in enumerate(label_objs_y):
            if i not in keep_indices_y:
                lbl_obj.set_visible(False)
        plt.yticks(locs_y, label_objs_y)

    plt.tight_layout()
    plt.show()


def plot_single_feat_distribution(rdf, feat_col, fig_title="Single Feature Distribution"):
    """
    Plots a bar chart (countplot) of a single feature column
    across the entire dataset (no target/hue).
    """
    plt.figure()
    sns.countplot(data=rdf, x=feat_col)
    plt.title(fig_title)
    plt.xlabel(feat_col)
    plt.ylabel("Count")

    # Limit x-axis to display only 10 subdivisions of the x-value range
    locs_x, label_objs_x = plt.xticks()
    total_xticks = len(label_objs_x)
    if total_xticks > 10:
        import numpy as np
        keep_indices = np.linspace(0, total_xticks - 1, 10, dtype=int)
        for i, lbl_obj in enumerate(label_objs_x):
            if i not in keep_indices:
                lbl_obj.set_visible(False)
        plt.xticks(locs_x, label_objs_x, rotation=45)

    # Limit y-axis to display only 10 subdivisions of the y-value range
    locs_y, label_objs_y = plt.yticks()
    total_yticks = len(label_objs_y)
    if total_yticks > 10:
        import numpy as np
        keep_indices_y = np.linspace(0, total_yticks - 1, 10, dtype=int)
        for i, lbl_obj in enumerate(label_objs_y):
            if i not in keep_indices_y:
                lbl_obj.set_visible(False)
        plt.yticks(locs_y, label_objs_y)

    plt.tight_layout()
    plt.show()


def plot_boxplot(rdf, col, fig_title="Box Plot"):
    """
    Plots a box plot of a chosen column.

    Args:
        rdf (pd.DataFrame): The dataset.
        col (str): Column name.
        fig_title (str): Title of the plot.
    """
    plt.figure()
    sns.boxplot(x=rdf[col])
    plt.title(fig_title)
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


def plot_pairplot(rdf, target=None):
    """
    Creates and displays a simple pair plot of the DataFrame's numeric columns,
    using hue=target if the target column exists in the DataFrame.
    """
    numeric_cols = rdf.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found for pair plot.")
        return
    
    # If the target column is in rdf, always use it as the hue
    if target in rdf.columns:
        g = sns.pairplot(rdf, vars=numeric_cols, corner=True, diag_kind="kde", hue=target)
    else:
        g = sns.pairplot(rdf[numeric_cols], corner=True, diag_kind="kde")
    
    for ax in g.axes.flatten():
        if ax is not None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, prop={'size': 6})
    
    plt.show()

def plot_qq(rdf, col, fig_title="Q-Q Plot"):
    """
    Creates and displays a Q-Q plot to check normality of a chosen column.
    Requires scipy.stats.probplot()

    Args:
        rdf (pd.DataFrame): The dataset.
        col (str): Column name for Q-Q plot.
        fig_title (str): Title of the plot.
    """
    plt.figure()
    stats.probplot(rdf[col].dropna(), dist="norm", plot=pylab)
    plt.title(fig_title)
    plt.show()


def plot_dist(rdf, col, fig_title="Distribution Plot"):
    """
    Creates and displays a distribution plot of a chosen column (hist + KDE).

    Args:
        rdf (pd.DataFrame): The dataset.
        col (str): Column name for distribution.
        fig_title (str): Title of the plot.
    """
    plt.figure()
    sns.histplot(rdf[col].dropna(), kde=True)
    plt.title(fig_title)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
