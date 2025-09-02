import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    matthews_corrcoef
)


THRESHOLDS = np.arange(0.00, 1.001, 0.01).tolist()


def calculate_metrics(probs, labels):
    """
    Calculate binary classification metrics.

    Parameters:
      probs : array-like, shape (n_samples,)
          Predicted probabilities for class 1.
      labels : array-like, shape (n_samples,)
          True binary labels (0 or 1).

    Returns:
      acc : float
          Accuracy at threshold 0.5.
      precision : float
          Precision at threshold 0.5.
      recall : float
          Recall at threshold 0.5.
      f1 : float
          F1 score at threshold 0.5.
      mAP : float
          Average precision (mean average precision).
      auc : float
          Area under the ROC curve.
    """
    # Compute binary predictions using threshold 0.5
    preds = np.where(np.array(probs) >= 0.5, 1, 0)

    # Compute metrics for threshold=0.5
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    # Compute metrics that use the full range of thresholds
    mAP = average_precision_score(labels, probs)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # it means only 1 class is present in labels.
        y = list(set(list(labels)))
        assert len(y) == 1
        y = y[0]        # 0 or 1
        auc = -10 - y   # -10 means only "normal" label present for group, -11 means only "abnormal"
        print(f"Cannot calculate AUC, set it as {auc}")

    return acc, precision, recall, f1, mAP, auc


def calculate_fn_group(probs, labels):
    """
    Calculate relative false negatives.

    Parameters:
      probs : array-like, shape (n_samples,)
          Predicted probabilities for class 1.
      labels : array-like, shape (n_samples,)
          True binary labels (expected to be all 1).

    Returns:
      fn_rel : float
          Relative false negatives (FN / total ground-truth positives).
          FN is computed as the number of cases where the predicted class
          (using threshold 0.5) is 0.
    """
    # Compute binary predictions using threshold 0.5
    preds = np.where(np.array(probs) >= 0.5, 1, 0)

    # Since all labels are 1, false negatives are where preds==0.
    fn = np.sum(preds == 0)

    # Relative false negatives: FN divided by total positives (length of labels)
    fn_rel = fn / len(labels)

    return fn_rel


def calculate_fn_group_thresholds(probs, labels, thresholds=THRESHOLDS):
    """
    Calculate relative false negatives for each threshold.

    Parameters:
      probs : array-like, shape (n_samples,)
          Predicted probabilities for class 1.
      labels : array-like, shape (n_samples,)
          True binary labels (expected to be all 1).
      thresholds : list or array-like
          List of threshold values to evaluate.

    Returns:
      fn_rel_list : list of floats
          Relative false negatives (FN / total ground-truth positives) for each threshold.
    """
    probs = np.array(probs)
    total = len(labels)
    fn_rel_list = []
    
    for t in thresholds:
        # Compute binary predictions using threshold t
        preds = np.where(probs >= t, 1, 0)
        # Since all labels are 1, false negatives are those predicted as 0.
        fn = np.sum(preds == 0)
        # Relative false negatives: FN divided by the total number of positive samples.
        fn_rel = fn / total
        fn_rel_list.append(fn_rel)
    
    return fn_rel_list



def calculate_MORE_metrics(preds, labels):
    """
    Calculate binary classification metrics.
    
    Parameters:
      preds : array-like, shape (n_samples,)
          Predicted probabilities for class 1.
      labels : array-like, shape (n_samples,)
          True binary labels (0 or 1).
    
    Returns:
      metr_acc : float
          Accuracy at threshold 0.5.
      recall_val : float
          Recall at threshold 0.5.
      precision_val : float
          Precision at threshold 0.5.
      f1_val : float
          F1 score at threshold 0.5.
      confmat : list of lists
          Confusion matrix (computed at threshold 0.5).
      auroc : float
          Area under the ROC curve.
      ap : float
          Average precision score.
      pr_curve_vals : tuple
          (precision, recall, thresholds) for the precision-recall curve.
      roc_curve_vals : tuple
          (fpr, tpr, thresholds) for the ROC curve.
      mcc_thresholded_vals : list
          List of Matthews Correlation Coefficient for each threshold.
      p_thresholded_vals : list
          List of precision values for each threshold.
      r_thresholded_vals : list
          List of recall values for each threshold.
    """
    # Ensure preds is a numpy array
    preds = np.array(preds)
    
    # Compute binary predictions using threshold 0.5
    binary_preds = (preds >= 0.5).astype(int)
    
    # Metrics at threshold 0.5
    metr_acc   = accuracy_score(labels, binary_preds)
    recall_val = recall_score(labels, binary_preds)
    precision_val = precision_score(labels, binary_preds)
    f1_val     = f1_score(labels, binary_preds)
    confmat    = confusion_matrix(labels, binary_preds).tolist()
    
    # Threshold-independent metrics (computed using continuous probability values)
    auroc = roc_auc_score(labels, preds)
    ap    = average_precision_score(labels, preds)
    pr_curve_vals = precision_recall_curve(labels, preds)
    roc_curve_vals = roc_curve(labels, preds)
    
    # Compute MCC, precision, and recall for each threshold in THRESHOLDS
    mcc_thresholded_vals = []
    p_thresholded_vals   = []
    r_thresholded_vals   = []
    acc_thresholded_vals = []
    f1_thresholded_vals  = []
    
    for t in THRESHOLDS:
        binary_preds_t = (preds >= t).astype(int)
        mcc_val = matthews_corrcoef(labels, binary_preds_t)
        # Use zero_division=0 to avoid errors if no positives are predicted
        p_val   = precision_score(labels, binary_preds_t, zero_division=0)
        r_val   = recall_score(labels, binary_preds_t, zero_division=0)
        acc_val = accuracy_score(labels, binary_preds_t)
        f1_val  = f1_score(labels, binary_preds_t, zero_division=0)
        mcc_thresholded_vals.append(mcc_val)
        p_thresholded_vals.append(p_val)
        r_thresholded_vals.append(r_val)
        acc_thresholded_vals.append(acc_val)
        f1_thresholded_vals.append(f1_val)
    
    return (metr_acc, precision_val, recall_val, f1_val, ap, auroc, 
            confmat, pr_curve_vals, roc_curve_vals,
            mcc_thresholded_vals, p_thresholded_vals, r_thresholded_vals, 
            acc_thresholded_vals, f1_thresholded_vals)

