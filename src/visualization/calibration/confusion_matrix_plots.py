from sklearn import metrics

from src.evaluation.metrics.utils import remove_missing, threshold


### CONFUSION MATRIX PLOTTING

def make_row_confusion_matrix_plots(config, row_ax, preds, labels, column_names):
    for i, col_name in enumerate(column_names):
        #confusion_matrix_subplot(config, row_ax[i], preds[col_name], labels[col_name])


        labels2, preds2 = remove_missing(config, labels[col_name], preds[col_name])
        labels_thresholded, preds_thresholded = threshold(config, labels2, preds2)

        confusion_matrix = metrics.confusion_matrix(labels_thresholded, preds_thresholded)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot(ax=row_ax[i], colorbar=False, include_values=True)

        if i!=0:
            cm_display.ax_.set_ylabel('')
        