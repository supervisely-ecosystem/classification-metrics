import numpy as np
import pandas as pd


def get_confusion_matrix_multilabel(img2classes_gt: dict, img2classes_pred: dict, classes: list):

    cols = list(classes) + ["None"]
    confusion_matrix = pd.DataFrame(
        np.zeros((len(cols), len(cols)), dtype=int), columns=cols, index=cols
    )

    for img_name, classes_gt in img2classes_gt.items():
        classes_pred = img2classes_pred.get(img_name)
        # TODO: checking for existence is redundant here.
        if classes_pred is None:
            FN = classes_gt
            FP = []
            TP = []
        else:
            gt, pred = set(classes_gt), set(classes_pred)
            FN = list(gt - pred)
            FP = list(pred - gt)
            TP = list(gt & pred)

        for cls in FN:
            confusion_matrix.loc[cls, "None"] += 1
        for cls in FP:
            confusion_matrix.loc["None", cls] += 1
        for cls in TP:
            confusion_matrix.loc[cls, cls] += 1

    return confusion_matrix


def get_confusion_matrix(img2classes_gt: dict, img2classes_pred: dict, classes: list):

    cols = list(classes) + ["None"]
    confusion_matrix = pd.DataFrame(
        np.zeros((len(cols), len(cols)), dtype=int), columns=cols, index=cols
    )

    for img_name, classes_gt in img2classes_gt.items():
        classes_pred = img2classes_pred.get(img_name)
        # We only have one class in a list for a single-label classification
        class_gt = classes_gt[0] if len(classes_gt) else "None"
        class_pred = classes_pred[0] if len(classes_pred) else "None"
        confusion_matrix[class_gt][class_pred] += 1

    return confusion_matrix


def get_dataframes(
    img2classes_gt: dict, img2classes_pred: dict, classes: list, is_single_label=True
):

    assert img2classes_gt.keys() == img2classes_pred.keys()

    gt = pd.DataFrame(np.zeros((len(img2classes_gt), len(classes)), dtype=int), columns=classes)
    pred = pd.DataFrame(np.zeros((len(img2classes_gt), len(classes)), dtype=int), columns=classes)
    img_names = []

    for i, (img_name, classes_gt) in enumerate(img2classes_gt.items()):
        classes_pred = img2classes_pred[img_name]
        if is_single_label:
            # taking the first item as we already has sorted by conf earlier
            classes_gt = classes_gt[0] if len(classes_gt) else "None"
            classes_pred = classes_pred[0] if len(classes_pred) else "None"
        gt.loc[i, classes_gt] = 1
        pred.loc[i, classes_pred] = 1
        img_names.append(img_name)

    return gt, pred, img_names


def get_confusion_matrix_multilabel_2(
    gt: pd.DataFrame, pred: pd.DataFrame, img_names: list, weighting_mode="none"
):
    weighting_mode = weighting_mode.lower()
    assert weighting_mode in ["none", "gt", "pred", "sample"]
    cols = list(gt.columns) + ["None"]
    n = len(cols)
    dtype = int if weighting_mode is None else float
    confusion_matrix = pd.DataFrame(np.zeros((n, n), dtype=dtype), columns=cols, index=cols)
    cells = [[[] for j in range(n)] for i in range(n)]
    confusion_matrix_imgs = pd.DataFrame(cells, columns=cols, index=cols)
    for gt_row, pred_row, img_name in zip(gt.values, pred.values, img_names):
        matched_mask = gt_row & pred_row
        matched_idxs = matched_mask.nonzero()[0]

        unmatched_mask = np.logical_xor(gt_row, pred_row)
        has_unmatched = np.any(unmatched_mask)
        wrong_idxs_gt = (gt_row & unmatched_mask).nonzero()[0]
        wrong_idxs_pred = (pred_row & unmatched_mask).nonzero()[0]
        if has_unmatched and not len(wrong_idxs_gt):
            wrong_idxs_gt = [-1]
        if has_unmatched and not len(wrong_idxs_pred):
            wrong_idxs_pred = [-1]

        if weighting_mode == "none":
            value = 1
        elif weighting_mode == "gt":
            value = 1 / len(wrong_idxs_gt)
        elif weighting_mode == "pred":
            value = 1 / len(wrong_idxs_pred)
        elif weighting_mode == "sample":
            value = 1 / (len(wrong_idxs_gt) + len(wrong_idxs_pred))

        if has_unmatched:
            confusion_matrix.iloc[wrong_idxs_gt, wrong_idxs_pred] += value
            confusion_matrix_imgs.iloc[wrong_idxs_gt, wrong_idxs_pred] = confusion_matrix_imgs.iloc[
                wrong_idxs_gt, wrong_idxs_pred
            ].applymap(lambda x: x + [img_name])
        if len(matched_idxs):
            confusion_matrix.values[matched_idxs, matched_idxs] += value
            for i in matched_idxs:
                confusion_matrix_imgs.iloc[i, i].append(img_name)

    return confusion_matrix, confusion_matrix_imgs


def get_overall_metrics(report, mlcm):
    df = pd.DataFrame(report)[["micro avg"]].T
    mlcm_sum = mlcm.sum(0)
    df["TP"] = mlcm_sum[1, 1]
    df["FN"] = mlcm_sum[1, 0]
    df["FP"] = mlcm_sum[0, 1]
    df.index = ["total"]
    df = df.rename(columns={"support": "count"})
    return df


def get_per_class_metrics(report, mlcm, classes):
    df = pd.DataFrame(report).iloc[:, : len(classes)].T
    df["TP"] = mlcm[:, 1, 1]
    df["FN"] = mlcm[:, 1, 0]
    df["FP"] = mlcm[:, 0, 1]
    df["Class"] = classes
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    df = df.rename(columns={"support": "count"})
    return df


# def filter_by_class(img2classes: dict, cls: str, not_in=False):
#     img_names = []
#     for img_name, img_classes in img2classes.items():
#         if not_in is False and cls in img_classes:
#             img_names.append(img_name)
#         if not_in is True and cls not in img_classes:
#             img_names.append(img_name)
#     return img_names



# def img_metrics(gt_tags, pred_tags, is_multilabel, suffix=None):
#     if is_multilabel:
#         if suffix is not None:
#             pred_tags = [
#                 tag.name[: -len(suffix)] if tag.name.endswith(suffix) else tag.name
#                 for tag in pred_tags
#             ]
#         else:
#             pred_tags = [tag.name for tag in pred_tags]

#         gt_tags = [tag.name for tag in gt_tags]
#         gt_tags = set(gt_tags)
#         pred_tags = set(pred_tags)
#         tp = len(gt_tags & pred_tags)
#         fp = len(pred_tags - gt_tags)
#         fn = len(gt_tags - pred_tags)
#         return [tp, fp, fn]
#     else:  # single-label
#         gt_tags = gt_tags[0].name
#         if suffix is not None and pred_tags[0].name.endswith(suffix):
#             pred_tags = pred_tags[0].name[: -len(suffix)]
#         else:
#             pred_tags = pred_tags[0].name
#         correct = gt_tags == pred_tags
#         return [correct]
