"""Classification metrics used by training and map evaluation."""

import numpy as np


def average_accuracy(confusion_matrix):
    """Return per-class accuracies and their unweighted mean."""
    confusion_matrix = np.asarray(confusion_matrix)
    class_totals = confusion_matrix.sum(axis=1)
    per_class = np.divide(
        np.diag(confusion_matrix),
        class_totals,
        out=np.zeros_like(class_totals, dtype=np.float64),
        where=class_totals != 0,
    )
    return per_class, float(per_class.mean())
