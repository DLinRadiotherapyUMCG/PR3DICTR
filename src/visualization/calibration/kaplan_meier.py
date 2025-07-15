import numpy as np

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts




def make_row_kaplan_meier_plots(config, row_ax, preds, labels, column_names):
    #print(preds)
    for i, col_name in enumerate(column_names):
        y_preds = preds[col_name]
        event_labels = labels[col_name][:, 0]  # assuming the first column is the event label
        time_labels = labels[col_name][:, 1]  # assuming the second column is

        kaplan_meier_subplot(config, row_ax[i], y_preds, event_labels, time_labels, col_name)



def kaplan_meier_subplot(config, ax, y_preds, event_labels, time_labels, endpoint_name):
    # stratify the predictions into high and low risk groups based on the median prediction value
    median_pred = np.median(y_preds)
    lowrisk_group_list = np.where(y_preds < median_pred)[0]
    highrisk_group_list = np.where(y_preds >= median_pred)[0]

    kmf_1 = KaplanMeierFitter()
    kmf_1.fit(time_labels[highrisk_group_list], event_labels[highrisk_group_list], label='High risk').plot_survival_function(ax=ax, show_censors=True, ci_show=True)

    kmf_0 = KaplanMeierFitter()
    kmf_0.fit(time_labels[lowrisk_group_list], event_labels[lowrisk_group_list], label='Low risk').plot_survival_function(ax=ax, show_censors=True, ci_show=True)
    
    # set the axes limits (must be done before adding the risk counts, otherwise the columns will not align)
    max_timepoint = config['evaluation']['visualisations']['kaplan_meier_max_timepoint']  # maximum time point to show in the plot (x-axis limit)
    ax.set_xlim(0, max_timepoint) 
    ax.set_ylim(0.0, 1.0)

    # add the table at the bottom
    add_at_risk_counts(kmf_1, kmf_0, ax=ax, rows_to_show=['At risk'])

    ax.legend( loc='lower left')
    ax.tick_params(axis='both') 
