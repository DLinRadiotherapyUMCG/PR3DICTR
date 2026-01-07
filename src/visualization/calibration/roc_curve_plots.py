from sklearn import metrics
from src.evaluation.metrics.utils import remove_missing


### ROC CURVE PLOTTING
def make_row_ROC_plots(config, row_ax, preds, labels, column_names):
    for i, col_name in enumerate(column_names):

        labels2, preds2 = remove_missing(config, labels[col_name], preds[col_name])

        fpr, tpr, _ = metrics.roc_curve(labels2, preds2)
        auc = metrics.roc_auc_score(labels2, preds2)

        row_ax[i].plot([0,1],[0,1], '--', color='gray', linewidth=2, zorder=3)  # ideal line
        row_ax[i].plot(fpr,tpr,label="AUC="+str(auc))