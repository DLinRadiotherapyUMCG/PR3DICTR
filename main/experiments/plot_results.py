import os
import pickle

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score

from src.visualization.plot_slices import plot_slices
from src.constants import ENSEMBLE_MEMBERS
from src.config_presets.tools.load_config import load_config
from src.dataset.load_dataset import load_dataset
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed


def plot_violin(df, columns, title) -> plt.Figure:
    """
    Plot violin plots for the specified columns using matplotlib.

    :param df: DataFrame containing the data.
    :param columns: List of column names to plot.
    :param title: Title of the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the violin plot
    for i, column in enumerate(columns):
        ax.violinplot(df[column].dropna(), positions=[i])

    # Set the title
    ax.set_title(title)

    # Set the xticks
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns)

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def plot_box(df, columns, title) -> plt.Figure:
    """
    Plot box plots for the specified columns using matplotlib.

    :param df: DataFrame containing the data.
    :param columns: List of column names to plot.
    :param title: Title of the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the box plot
    ax.boxplot([df[column].dropna() for column in columns])

    # Set the title
    ax.set_title(title)

    # Set the xticks
    ax.set_xticks(range(1, len(columns) + 1))
    ax.set_xticklabels(columns)

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def plot_scatter(df, x_column, y_columns, title) -> plt.Figure:
    """
    Plot scatter plots for the specified columns using matplotlib.

    :param df: DataFrame containing the data.
    :param x_column: Column name for the x-axis.
    :param y_columns: List of column names for the y-axis.
    :param title: Title of the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the scatter plots
    for y_column in y_columns:
        ax.scatter(df[x_column], df[y_column], label=y_column)

    ax.set_xlim(0, 1)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel('Uncertainty')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def plot_histogram(df, column, title, bins=50) -> plt.Figure:
    """
    Plot a histogram for the specified column using matplotlib.

    :param df: DataFrame containing the data.
    :param column: Column name to plot.
    :param title: Title of the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the histogram
    ax.hist(df[column].dropna(), bins=np.linspace(0, 1, bins + 1))

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def plot_calibration(df, pred_column, true_column, title, n_bins=10) -> plt.Figure:
    """
    Plot a calibration plot using matplotlib.

    :param df: DataFrame containing the data.
    :param pred_column: Column name for the predicted values.
    :param true_column: Column name for the true labels.
    :param n_bins: Number of bins to use.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Bin the predicted values
    df['bin'] = pd.cut(df[pred_column], bins=n_bins)

    # Compute the mean predicted value and actual fraction of positives for each bin
    bin_means = df.groupby('bin')[pred_column].mean()
    bin_true = df.groupby('bin')[true_column].mean()

    # Plot the calibration curve
    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.plot(bin_means, bin_true, 's-')

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def plot_auc(df, pred_column, true_column, sort_by, title) -> plt.Figure:
    """
    Plot the accuracy of a group of predictions while iteratively removing the top 5% of samples with the highest uncertainty.

    :param df: DataFrame containing the data.
    :param pred_column: Column name for the predicted values.
    :param true_column: Column name for the true labels.
    :param sort_by: List of column names to sort by.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get number of samples
    n_samples = len(df)
    perc = 0.05
    n_samples_to_remove = int(n_samples * perc)

    # Start a loop that will iterate over each column name in sort_by
    for column in sort_by:
        # Create a copy of the DataFrame and sort it by the current column name
        df_copy = df.sort_values(by=column, ascending=True)

        # Initialize a list to store the accuracy values for the current column
        accuracies = []

        # Start a loop that will run until the DataFrame copy is empty
        while len(df_copy) > 0:
            # Calculate the accuracy of the predictions
            try:
                auc = roc_auc_score(df_copy[true_column], df_copy[pred_column])
                # Append the accuracy to the list
                accuracies.append(auc)
                # Remove the top of the DataFrame copy
                df_copy.drop(df_copy.head(n_samples_to_remove).index, inplace=True)
            except:
                break

        # Plot the accuracy values for the current column
        ax.plot(accuracies, label=column)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('AUC')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def plot_lr_coefficients(config) -> plt.Figure:
    """
    Load each of the 5 logistic regression models and plot their coefficients as bar charts.

    :param config: Configuration dictionary.
    """
    # Initialize a list to store the coefficients of each model
    coefficients_list = []

    reference_coefficients = [0.0193, 0.1054, 0.5234, 1.2763]

    # Start a loop that will iterate over the numbers 1 to 5
    for i in range(0, ENSEMBLE_MEMBERS):
        # Construct the file path for the current model
        model_path = os.path.join(config['paths']['output'], f'lr_model_{i}.pkl')

        # Load the model from the file
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Get the coefficients of the model
        coefficients = model.coef_[0]

        # Append the coefficients to the list
        coefficients_list.append(coefficients)

    # Add the reference coefficients to the list
    coefficients_list.append(reference_coefficients)

    # Convert the list of coefficients to a 2D NumPy array and transpose it
    coefficients_array = np.array(coefficients_list).T

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Define the bar width
    bar_width = 0.15

    # Plot the coefficients as bar charts
    for i, coefficients in enumerate(coefficients_array.T):  # Transpose back to original shape for iteration
        positions = np.arange(len(coefficients))
        # Define the positions based on the shape of the coefficients
        ax.bar(positions + i * bar_width, coefficients, width=bar_width,
               label=f'Model {i + 1 if i < ENSEMBLE_MEMBERS else "Reference"}')

    ax.set_xticks(positions + bar_width * (ENSEMBLE_MEMBERS) / 2)
    ax.set_xticklabels(config['columns']['logistic_regression_features'], rotation=10)  # Rotate labels by 45 degrees

    # Set the title and labels
    ax.set_title('Logistic Regression Coefficients')
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Coefficient Value')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def calculate_metrics(results_df, model_type, results_folder, reference_values=None):
    metrics_df = pd.DataFrame(columns=['Accuracy', 'AUC'])
    true_labels = results_df['label']

    for model_index in range(ENSEMBLE_MEMBERS):
        pred_probs = results_df[f'pred_{model_type}_{model_index}']

        accuracy = round(accuracy_score(true_labels, pred_probs.round()), 3)
        auc = round(roc_auc_score(true_labels, pred_probs), 3)

        metrics_df.loc[f'{model_type}_{model_index}'] = [accuracy, auc]

    # Calculate the mean of the metrics
    metrics_df.loc[f'mean_{model_type}'] = metrics_df.mean()

    ensemble_auc = round(roc_auc_score(true_labels, results_df[f'mean_{model_type}']), 3)
    metrics_df.loc[f'ensemble_{model_type}'] = [None, ensemble_auc]

    # Add the reference model
    if reference_values is not None:
        metrics_df.loc[f'reference_{model_type}'] = reference_values

    # Save the metrics dataframe to a CSV file
    metrics_df.to_csv(os.path.join(results_folder, f'{model_type}_metrics.csv'))


def plot_umcg_calibration(df, pred_column, true_column, title, DL_model=True, n_bins=10):
    """
    Plot a UMCG-style calibration plot using matplotlib.

    :param df: DataFrame containing the data.
    :param pred_column: Column name for the predicted values.
    :param true_column: Column name for the true labels.
    :param title: Title of the plot.
    :param DL_model: Boolean indicating whether the model is a DL model.
    :param n_bins: Number of bins to use.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get the predicted values and true labels
    y_pred = df[pred_column]
    y_true = df[true_column]

    # Calculate the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, pos_label=1, n_bins=n_bins,
                                                                    strategy="quantile")

    # Fit a linear model to the calibration curve
    coef = np.polyfit(mean_predicted_value, fraction_of_positives, 1)
    linear_fit = np.poly1d(coef)

    # Plot the ideal line
    ax.plot([0, 1], [0, 1], "black", label="Ideal")

    # Plot the calibration curve
    shape = 'o' if DL_model else 's'
    label = "DL Model" if DL_model else "CITOR"
    ax.plot(mean_predicted_value, fraction_of_positives, shape, label=label)

    # Plot the best fit line
    ax.plot([0, 1], linear_fit([0, 1]), '--')

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def save_plot(config, fig, name):
    """
    Save a plot to a file.

    :param config: Configuration dictionary.
    :param fig: Figure to save.
    :param name: Name of the file.
    """
    # Construct the file path
    file_path = os.path.join(config['paths']['results'], f'{name}.png')

    # Save the figure
    fig.savefig(file_path)

    # Close the figure
    # plt.close(fig)


def calc_bins(y_pred, y_true, num_bins=10):
    # Assign each prediction to a bin
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(y_pred, bins, right=True)
    # unique, counts = np.unique(binned, return_counts=True)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    labels_oneh = y_true
    for bin in range(num_bins):
        bin_sizes[bin] = len(y_pred[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (y_pred[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_ECE_metrics(bins, _, bin_accs, bin_confs, bin_sizes):
    """
    Calculates the ECE and MCE metrics for a set of bins.
    """
    ECE = 0
    MCE = 0

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def plot_reliability(df, pred_column, true_column, title, n_bins=10):
    """
    Plot a reliability plot using matplotlib.

    :param config: Configuration dictionary.
    :param pred_column: Column name for the predicted values.
    :param true_column: Column name for the true labels.
    :param title: Title of the plot.
    :param DL_model: Boolean indicating whether the model is a DL model.
    :param n_bins: Number of bins to use.
    """

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get the predicted values and true labels
    y_pred = df[pred_column]
    y_true = df[true_column]

    mask = [True if x in [1, 0] else False for x in y_true]

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_pred, y_true, num_bins=n_bins)

    ECE, MCE = get_ECE_metrics(bins, _, bin_accs, bin_confs, bin_sizes)

    bin_centers = bins - 0.05

    # Error bars (light red bars)
    ax.bar(bin_centers, bin_centers, width=0.1, alpha=0.2, edgecolor='black', color='r', hatch='/', zorder=1)

    # Draw (blue) bars and identity line
    ax.bar(bin_centers, bin_accs, width=0.1, alpha=1, edgecolor='black', color='tab:blue', zorder=2)
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, zorder=3)

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE * 100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE * 100))
    ax.legend(handles=[ECE_patch, MCE_patch])

    # Create grid
    ax.grid(True, color='gray', linestyle='dashed', zorder=0)

    # plot gaps
    gaps = (bin_accs - bin_centers)
    gaps = np.clip(gaps, 0, 1)

    ax.bar(bin_centers, gaps, bottom=np.minimum(bin_accs, (bins - 0.05)),
           width=0.1, alpha=0.4, edgecolor='black', color='r', hatch='\\/', zorder=3)

    ECE = round(ECE, 3)
    MCE = round(MCE, 3)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')

    # Show the plot
    plt.show()

    # Return the figure
    return fig


def plot_correlation(df, column1, column2, title, label_column='label'):
    """
    Plot a scatter plot of two columns and a line of best fit.

    :param df: DataFrame containing the data.
    :param column1: Column name for the x-axis.
    :param column2: Column name for the y-axis.
    :param title: Title of the plot.
    :param label_column: Column name for the labels.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the scatter plot with colors based on the label
    scatter = ax.scatter(df[column1], df[column2], c=df[label_column], label=f'{column1} vs {column2}', cmap='brg')

    # Create a legend for the scatter plot
    legend1 = ax.legend(*scatter.legend_elements(), title=label_column, bbox_to_anchor=(0., 0.8, 1., .102))
    ax.add_artist(legend1)

    # Calculate the line of best fit
    m, b = np.polyfit(df[column1], df[column2], 1)

    # Plot the line of best fit
    ax.plot(df[column1], m * df[column1] + b, color='red', label='Line of Best Fit')

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(column1)
    ax.set_ylabel(column2)

    # Add a legend box showing the slope of the line of best fit
    slope_patch = mpatches.Patch(color='red', label=f'Slope = {m:.2f}')
    ax.legend(handles=[slope_patch])

    # Show the plot
    plt.show()

    # Return the figure
    return fig


# Plots the first or last n patients based on a column to sort by
def plot_n_patients(config, df, n, sort_by, title, ascending=True):
    """
    Plot the first or last n patients based on a column to sort by.

    :param df: DataFrame containing the data.
    :param n: Number of patients to plot.
    :param sort_by: Column name to sort by.
    :param title: Title of the plot.
    """

    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=sort_by, ascending=ascending)

    slices = [15, 30, 45, 60, 75, 90]
    rows = []

    # For each of the patients, plot get models inputs
    for i in range(n):
        patient = df_sorted.iloc[i]
        patientID = patient['PatientID']
        test_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], 'test.csv'),
                                           patient_ids=[patientID])
        inputs, clinical_features, targets = test_data[0]

        rows.append({
            'Label': f'Patient: {patientID}\nLabel {patient["label"]}',
            'CT': inputs[0],
            'RTDOSE': inputs[1],
            'RTSTRUCT': inputs[2],
        })

    fig, axs = plot_slices(rows, slices, title=title)

    fig.show()

    # Return the figure
    return fig


def plot_categorical_boxplot(df, title, category_column, plot_columns):
    """
    Plot box plots for the specified columns grouped by a category column.

    :param df: DataFrame containing the data.
    :param category_column: Column name for the categories.
    :param plot_columns: List of column names to plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get the unique categories and create a color map
    categories = sorted(df[category_column].unique())
    colors = cm.rainbow(np.linspace(0, 1, len(categories)))

    # Calculate the total number of rows for percentage calculation
    total_rows = len(df)

    # Initialize a list to hold the legend labels with percentages
    legend_labels_with_percentages = []

    # For each unique category in the category column
    for i, category in enumerate(categories):
        # Create a subset of the dataframe for the current category
        subset = df[df[category_column] == category]

        # Calculate the percentage of data belonging to the current category
        percentage = (len(subset) / total_rows) * 100

        # Create the box plots for the specified columns
        box = ax.boxplot([subset[column].dropna() for column in plot_columns],
                         positions=[i * len(plot_columns) + j for j in range(len(plot_columns))],
                         patch_artist=True)

        # Set the color of the box plots
        for patch in box['boxes']:
            patch.set_facecolor(colors[i])

        # Add the category and its percentage to the legend labels list
        legend_labels_with_percentages.append(f'{category} ({percentage:.2f}%)')

    # Set the title
    ax.set_title(title)

    # Set the xticks
    ax.set_xticks(range(len(categories) * len(plot_columns)))
    ax.set_xticklabels([f'{category}_{column}' for category in categories for column in plot_columns],
                       rotation=90)

    # Create a legend with the updated labels
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color, category in
                      zip(colors, legend_labels_with_percentages)]
    ax.legend(legend_patches, legend_labels_with_percentages, loc='best')

    # Show the plot
    fig.show()

    return fig


def perform_anova(df, category_columns, value_column):
    """
    Perform an ANOVA test for the specified value column grouped by multiple category columns.

    :param df: DataFrame containing the data.
    :param category_columns: List of column names for the categories.
    :param value_column: Column name for the values.
    """
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(columns=['Category', 'Mean', 'F-value', 'p-value'])

    # For each category column
    for category_column in category_columns:
        # Get the unique categories
        categories = df[category_column].unique()

        # Create a list to store the values for each category
        category_values = []

        # For each unique category in the category column
        for category in categories:
            # Create a subset of the dataframe for the current category
            subset = df[df[category_column] == category]

            # Append the values for the current category to the list
            category_values.append(subset[value_column].dropna())

        # Perform the ANOVA test
        f_value, p_value = f_oneway(*category_values)

        # Mean
        means = [np.round(np.mean(values), 3) for values in category_values]

        # Append the results to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            'Category': [category_column],
            'Mean': [means],
            'F-value': [np.round(f_value, 2)],
            'p-value': [np.round(p_value, 5)]
        })], ignore_index=True)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Hide the axes
    ax.axis('off')

    # Create a table and add it to the figure
    table = plt.table(cellText=results_df.values, colLabels=results_df.columns, loc='center',
                      colWidths=[0.2, 0.45, 0.1, 0.15])

    # Color any significant p-value green
    cellDict = table.get_celld()
    for i in range(len(results_df) + 1):
        for j in range(4):
            cellDict[(i, j)].set_fontsize(12)
            if i > 0 and j == 3:
                if results_df.iloc[i - 1, j] < 0.05:
                    cellDict[(i, j)].set_facecolor('aqua')
                if results_df.iloc[i - 1, j] < 0.001:
                    cellDict[(i, j)].set_facecolor('lime')

    # Set the title
    plt.title(
        f'ANOVA results for {value_column} grouped by categories\n(p-values < 0.05 are colored blue, < 0.001 are colored green)')

    # Show the plot
    fig.show()

    return fig


def main():
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    config = load_config(toxicity)

    # Disable randomness
    set_random_seed(config['seed'])

    # Get results
    results_df = pd.read_csv(os.path.join(config['paths']['results'], 'results.csv'),
                             dtype={'PatientID': str, 'label': int})
    features_df = pd.read_csv(os.path.join(config['paths']['csv'], 'test.csv'),
                              delimiter=';',
                              dtype={'PatientID': str})
    merged_df = results_df.merge(features_df, on='PatientID', how='left')

    # Calculate the metrics for the dl and lr models
    calculate_metrics(results_df, 'dl', config['paths']['results'], [0.8, 0.7])
    calculate_metrics(results_df, 'lr', config['paths']['results'], [0.8, 0.7])

    # Make plots
    # Entropy and variance violin plots
    # save_plot(config,
    #           plot_violin(results_df, ['predictive_entropy_lr', 'predictive_entropy_dl'], 'Predictive Entropy'),
    #           'predictive_entropy')
    # save_plot(config,
    #           plot_violin(results_df, ['predictive_variance_lr', 'predictive_variance_dl'], 'Predictive Variance'),
    #           'predictive_variance')

    # Entropy and variance box plots
    # save_plot(config,
    #           plot_box(results_df, ['predictive_entropy_lr', 'predictive_entropy_dl'], 'Predictive Entropy'),
    #           'predictive_entropy')
    # save_plot(config,
    #           plot_box(results_df, ['predictive_variance_lr', 'predictive_variance_dl'], 'Predictive Variance'),
    #           'predictive_variance')

    # Disentangled uncertainties plotted against the mean
    save_plot(config, plot_scatter(results_df, 'mean_dl', ['aleatoric_uncertainty_dl', 'epistemic_uncertainty_dl'],
                                   'DL Uncertainty vs Mean'), 'dl_uncertainty_vs_mean')
    save_plot(config, plot_scatter(results_df, 'mean_lr', ['aleatoric_uncertainty_lr', 'epistemic_uncertainty_lr'],
                                   'LR Uncertainty vs Mean'), 'lr_uncertainty_vs_mean')

    # Histograms
    save_plot(config, plot_histogram(results_df, 'mean_dl', 'Histogram of mean_dl', bins=10), 'histogram_mean_dl')
    save_plot(config, plot_histogram(results_df, 'mean_lr', 'Histogram of mean_lr', bins=10), 'histogram_mean_lr')

    # Calibration plots
    save_plot(config, plot_calibration(results_df, 'mean_dl', 'label', 'DL Calibration plot'), 'calibration_dl')
    save_plot(config, plot_calibration(results_df, 'mean_lr', 'label', 'LR Calibration plot'), 'calibration_lr')

    # UMCG-style calibration plots
    save_plot(config, plot_umcg_calibration(results_df, 'mean_dl', 'label', 'DL Calibration plot'),
              'umcg_calibration_dl')
    save_plot(config, plot_umcg_calibration(results_df, 'mean_lr', 'label', 'LR Calibration plot', DL_model=False),
              'umcg_calibration_lr')

    # Reliability plots
    save_plot(config, plot_reliability(results_df, 'mean_dl', 'label', 'DL Reliability plot'), 'reliability_dl')
    save_plot(config, plot_reliability(results_df, 'mean_lr', 'label', 'LR Reliability plot'), 'reliability_lr')

    # Accuracy vs uncertainty plots
    save_plot(config, plot_auc(results_df, 'mean_dl', 'label',
                               ['predictive_entropy_dl', 'aleatoric_uncertainty_dl', 'epistemic_uncertainty_dl'],
                               'DL AUC vs Uncertainty'), 'accuracy_vs_uncertainty_dl')
    save_plot(config, plot_auc(results_df, 'mean_lr', 'label',
                               ['predictive_entropy_lr', 'aleatoric_uncertainty_lr', 'epistemic_uncertainty_lr'],
                               'LR AUC vs Uncertainty'), 'accuracy_vs_uncertainty_lr')

    # Uncertainty correlation
    save_plot(config, plot_correlation(results_df, 'epistemic_uncertainty_dl', 'epistemic_uncertainty_lr',
                                       'Epistemic Uncertainty DL vs LR'), 'correlation_epistemic_uncertainty')
    save_plot(config, plot_correlation(results_df, 'aleatoric_uncertainty_dl', 'aleatoric_uncertainty_lr',
                                       'Aleatoric Uncertainty DL vs LR'), 'correlation_aleatoric_uncertainty')

    # Plot correlation between mean predictions
    save_plot(config, plot_correlation(results_df, 'mean_dl', 'mean_lr', 'Mean Predictions DL vs LR'),
              'correlation_mean_predictions')

    # Logistic regression coefficients
    save_plot(config, plot_lr_coefficients(config), 'lr_coefficients')

    # # Plot the first 10 with highest aleatoric uncertainty
    # save_plot(config, plot_n_patients(config, results_df, 10, 'aleatoric_uncertainty_dl',
    #                                   'Patients with highest aleatoric uncertainty'),
    #           'patients_highest_aleatoric_uncertainty')
    # # Plot the first 10 with lowest aleatoric uncertainty
    # save_plot(config, plot_n_patients(config, results_df, 10, 'aleatoric_uncertainty_dl',
    #                                   'Patients with lowest aleatoric uncertainty', ascending=False),
    #           'patients_lowest_aleatoric_uncertainty')
    # # Plot the first 10 with highest epistemic uncertainty for DL
    # save_plot(config, plot_n_patients(config, results_df, 10, 'epistemic_uncertainty_dl',
    #                                   'Patients with highest DL epistemic uncertainty'),
    #           'patients_highest_epistemic_uncertainty_dl')
    # # Plot the first 10 with lowest epistemic uncertainty for DL
    # save_plot(config, plot_n_patients(config, results_df, 10, 'epistemic_uncertainty_dl',
    #                                   'Patients with lowest DL epistemic uncertainty', ascending=False),
    #           'patients_lowest_epistemic_uncertainty_dl')
    # # Plot the first 10 with highest epistemic uncertainty for LR
    # save_plot(config, plot_n_patients(config, results_df, 10, 'epistemic_uncertainty_lr',
    #                                   'Patients with highest LR epistemic uncertainty'),
    #           'patients_highest_epistemic_uncertainty_lr')
    # # Plot the first 10 with lowest epistemic uncertainty for LR
    # save_plot(config, plot_n_patients(config, results_df, 10, 'epistemic_uncertainty_lr',
    #                                   'Patients with lowest LR epistemic uncertainty', ascending=False),
    #           'patients_lowest_epistemic_uncertainty_lr')

    COLUMNS = ['Loctum2', 'Sex', 'CT_Artefact', 'Photons', 'T_stage', 'Smoking', 'N_stage', 'P16']

    for column in COLUMNS:
        # DL
        save_plot(config,
                  plot_categorical_boxplot(merged_df, f'DL: Aleatoric+Epistemic by Category ({column})', column,
                                           ['aleatoric_uncertainty_dl', 'epistemic_uncertainty_dl']),
                  f'categorical_boxplot_{column}_aleatoric_epistemic_dl')
        save_plot(config, plot_categorical_boxplot(merged_df, f'DL: Aleatoric by Category ({column})', column,
                                                   ['aleatoric_uncertainty_dl']),
                  f'categorical_boxplot_{column}_aleatoric_dl')
        save_plot(config, plot_categorical_boxplot(merged_df, f'DL: Epistemic by Category ({column})', column,
                                                   ['epistemic_uncertainty_dl']),
                  f'categorical_boxplot_{column}_epistemic_dl')

        # LR
        save_plot(config,
                  plot_categorical_boxplot(merged_df, f'LR: Aleatoric+Epistemic by Category ({column})', column,
                                           ['aleatoric_uncertainty_lr', 'epistemic_uncertainty_lr']),
                  f'categorical_boxplot_{column}_aleatoric_epistemic_lr')
        save_plot(config, plot_categorical_boxplot(merged_df, f'LR: Aleatoric by Category ({column})', column,
                                                   ['aleatoric_uncertainty_lr']),
                  f'categorical_boxplot_{column}_aleatoric_lr')
        save_plot(config, plot_categorical_boxplot(merged_df, f'LR: Epistemic by Category ({column})', column,
                                                   ['epistemic_uncertainty_lr']),
                  f'categorical_boxplot_{column}_epistemic_lr')

        # DL and LR
        save_plot(config,
                  plot_categorical_boxplot(merged_df, f'DL+LR: Aleatoric by Category ({column})', column,
                                           ['aleatoric_uncertainty_dl', 'aleatoric_uncertainty_lr']),
                  f'categorical_boxplot_{column}_aleatoric_dl_lr')
        save_plot(config,
                    plot_categorical_boxplot(merged_df, f'DL+LR: Epistemic by Category ({column})', column,
                                             ['epistemic_uncertainty_dl', 'epistemic_uncertainty_lr']),
                    f'categorical_boxplot_{column}_epistemic_dl_lr')

    for uncertainty_type in ['aleatoric_uncertainty_dl', 'epistemic_uncertainty_dl', 'aleatoric_uncertainty_lr',
                             'epistemic_uncertainty_lr']:
        fig = perform_anova(merged_df, COLUMNS, uncertainty_type)
        save_plot(config, fig, f'anova_{uncertainty_type}')


if __name__ == '__main__':
    main()
