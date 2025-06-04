# E:/DeepLearn/utils/plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Ensure sklearn is installed

def generate_confusion_matrix_figure(true_labels: np.ndarray, pred_labels: np.ndarray, display_labels: list[str], title: str):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(display_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig_s_base, fig_s_factor = 8, 0.6
    fig_w = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig_h = max(fig_s_base, len(display_labels) * fig_s_factor)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    return fig