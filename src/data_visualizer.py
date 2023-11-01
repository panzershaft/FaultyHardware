from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


class DataVisualizer:
    def __init__(self, data_set):
        self.data_set = data_set

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def plot_precision_recall_curve(y_true, y_prob, title='Precision-Recall Curve'):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.show()
