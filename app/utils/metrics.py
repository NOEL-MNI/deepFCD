import numpy as np

def dc(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    dc = 2. * intersection.sum() / im_sum

    return dc


def deltaVol(im1, im2):
    """
    absolute difference in volume
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.abs(im2.sum() - im1.sum()) / im1.sum()


def perf_measure_vox(y_pred, y_true):
    # TP = np.zeros(1, dtype=float)
    # FP = np.zeros(1, dtype=float)
    # TN = np.zeros(1, dtype=float)
    # FN = np.zeros(1, dtype=float)

    # for k in range(len(y_pred)):
    #     if y_true[k] == y_pred[k] == 1 :
    #        TP += 1
    #     if y_pred[k] == 1 and y_true[k] != y_pred[k]:
    #        FP += 1
    #     if y_true[k] == y_pred[k] == 0:
    #        TN += 1
    #     if y_pred[k] == 0 and y_true[k] != y_pred[k]:
    #        FN += 1

    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    sensitivity = 100*TP/(TP+FN)
    specificity = 100*TN/(TN+FP)

    print('-'*60)
    print("sensitivity: %.2f" %(sensitivity))
    print("specificity: %.2f" %(specificity))
    print('-'*60)

    perf = {
			'sensitivity': sensitivity,
			'specificity': specificity,
			'TP': TP,
			'FP': FP,
			'TN': TN,
			'FN': FN,
	}

    return perf
