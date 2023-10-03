import pandas as pd
import sklearn.metrics
import supervisely as sly
import numpy as np
from src import metric_utils, utils

try:
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=False)
    from src.compute_overlap import compute_overlap
except:
    print("Couldn't import compute_overlap. Check cython installation.")
    from src.compute_overlap_slow import compute_overlap


def match_bboxes(pairwise_iou: np.ndarray, min_iou_threshold=0.15):
    # pairwise_iou shape: [pred, gt]
    n_pred, n_gt = pairwise_iou.shape

    matched_idxs = []
    ious_matched = []
    unmatched_idxs_gt, unmatched_idxs_pred = set(range(n_gt)), set(range(n_pred))

    m = pairwise_iou.flatten()
    for i in m.argsort()[::-1]:
        if m[i] < min_iou_threshold or m[i] == 0:
            break
        pred_i = i // n_gt  # row
        gt_i = i % n_gt  # col
        if gt_i in unmatched_idxs_gt and pred_i in unmatched_idxs_pred:
            matched_idxs.append([gt_i, pred_i])
            ious_matched.append(m[i])
            unmatched_idxs_gt.remove(gt_i)
            unmatched_idxs_pred.remove(pred_i)

    return matched_idxs, list(unmatched_idxs_gt), list(unmatched_idxs_pred), ious_matched


def collect_labels_with_tags(labels: list, selected_classes: list, unsuffix: dict = None, return_scores=False, topk=None):
    selected_classes = set(selected_classes)
    bboxes, classes = [], []
    label_ids = []
    all_scores = []
    has_scores = True
    for label in labels:
        label : sly.Label
        if not isinstance(label.geometry, sly.Rectangle):
            continue
        
        tags = []
        scores = []
        for tag in label.tags:
            if unsuffix:
                if unsuffix.get(tag.name) is None:
                    continue
                tag_name = unsuffix[tag.name]
            else:
                tag_name = tag.name
            if tag_name in selected_classes:
                tags.append(tag_name)
                # if the tag has a confidence value
                if isinstance(tag.value, float):
                    scores.append(tag.value)

        if len(tags) == 0:
            continue

        # sort tags by confidence score
        if len(tags) == len(scores):
            tags, scores = list(zip(*sorted(zip(tags, scores), key=lambda pair: pair[1], reverse=True)))
            tags, scores = list(tags), list(scores)
            if topk:
                tags, scores = tags[:topk], scores[:topk]
        else:
            # If we don't have scores for every tag and the mode is single-label and the label has multiple tags,
            # it can lead to incorrect metric calculation.
            if return_scores and has_scores:
                has_scores = False
                sly.logger.warn("Prediction tags don't have confidence scores.")

        rect = label.geometry
        bbox = [rect.left, rect.top, rect.right, rect.bottom]
        bboxes.append(bbox)
        classes.append(tags)
        label_ids.append(label.geometry.sly_id)
        all_scores.append(scores)
    if not has_scores:
        all_scores = None
    if not return_scores:
        return bboxes, classes, label_ids
    else:
        return bboxes, classes, label_ids, all_scores


def compute_image_match(bboxes_gt, bboxes_pred, classes_gt, classes_pred, iou_threshold=0.5):

    if len(bboxes_gt) == 0 and len(bboxes_pred) == 0:
        return [], []

    if len(bboxes_gt) != 0 and len(bboxes_pred) != 0:
        # [Pred x GT]
        pairwise_iou = compute_overlap(np.array(bboxes_pred, dtype=np.float64), np.array(bboxes_gt, dtype=np.float64))
        matched_idxs, unmatched_idxs_gt, unmatched_idxs_pred, box_ious_matched = match_bboxes(pairwise_iou, iou_threshold)
    else:
        matched_idxs = []
        unmatched_idxs_gt = list(range(len(bboxes_gt)))
        unmatched_idxs_pred = list(range(len(bboxes_pred)))

    return matched_idxs, box_ious_matched


def flatten_img2classes_with_bboxes(img2clasees):
    res = {}
    for img_name, classes_query in img2clasees.items():
        for i, classes in enumerate(classes_query):
            res[f"{img_name} (bbox {i:02})"] = classes
    return res


class Metrics:
    def __init__(self, annotations, ds_match, selected_tags, iou_threshold):
        self.id2ann = annotations
        self.ds_match = ds_match
        self.selected_tags = selected_tags
        self.is_multilabel = None
        self.iou_threshold = iou_threshold
        self.classes, classes_with_suffix = list(zip(*selected_tags))
        self.unsuffix = dict(zip(classes_with_suffix, self.classes))


    def calculate(self):
        self.is_multilabel = self.detect_task_type()
        self.match_objects()
        img2classes_gt = {name: x["classes_gt"] for name, x in self.match_info.items()}
        img2classes_pred = {name: x["classes_pred"] for name, x in self.match_info.items()}
        gt, pred, img_names = metric_utils.get_dataframes(img2classes_gt, img2classes_pred, self.classes, is_single_label=False)

        ### Confusion Matrix
        self.confusion_matrix, self.cm_img_names = metric_utils.get_confusion_matrix_multilabel_2(gt, pred, img_names)

        ### Classification metrics
        self.report = sklearn.metrics.classification_report(gt.values, pred.values, target_names=self.classes, output_dict=True)

        ### Multi-label Confusion Matrix
        # [[TN, FP]
        #  [FN, TP]]
        self.mlcm = sklearn.metrics.multilabel_confusion_matrix(gt.values, pred.values)
        
        ### Overall metrics
        self.overall_metrics_df = metric_utils.get_overall_metrics(self.report, self.mlcm)

        ### Per-class metrics
        self.per_class_metrics_df = metric_utils.get_per_class_metrics(self.report, self.mlcm, self.classes)


    def match_objects(self):
        self.match_info = {}
        self.match_image_names = []
        self.total_matched = 0
        self.total_objects = 0
        topk = None if self.is_multilabel else 1

        for i, (dataset_name, item) in enumerate(self.ds_match.items()):
            if item["dataset_matched"] != "both":
                continue
            for img_item in item["matched"]:
                img_gt, img_pred = img_item["left"], img_item["right"]
                img_gt: sly.ImageInfo
                img_pred: sly.ImageInfo
                ann_gt: sly.Annotation = self.id2ann[img_gt.id]
                ann_pred: sly.Annotation = self.id2ann[img_pred.id]

                bboxes_gt, classes_gt, label_ids_gt = collect_labels_with_tags(ann_gt.labels, self.classes)
                bboxes_pred, classes_pred, label_ids_pred, scores_pred = collect_labels_with_tags(ann_pred.labels, self.classes, self.unsuffix, return_scores=True, topk=topk)

                if len(bboxes_gt) == 0:
                    continue

                matched_idxs, matched_iou = compute_image_match(bboxes_gt, bboxes_pred, classes_gt, classes_pred, self.iou_threshold)

                img_name = img_gt.name
                n_digits = len(str(len(matched_idxs)))
                for i, (idxs, iou) in enumerate(zip(matched_idxs, matched_iou)):
                    idx_gt, idx_pred = idxs
                    extra_str = "/".join(classes_gt[idx_gt])
                    item_name = f"{img_name} (bbox #{i:0{n_digits}}: {extra_str})"
                    self.match_info[item_name] = {
                        "img_info_gt": img_gt,
                        "img_info_pred": img_pred,
                        "classes_gt": classes_gt[idx_gt],
                        "classes_pred": classes_pred[idx_pred],
                        "scores_pred": scores_pred[idx_pred] if scores_pred else None,
                        "label_id_gt": label_ids_gt[idx_gt],
                        "label_id_pred": label_ids_pred[idx_pred],
                        "bbox_iou": iou,
                        "img_name": img_name
                    }
                    self.match_image_names.append(item_name)

                self.total_matched += len(matched_idxs)
                self.total_objects += len(bboxes_gt)


    def query_metrics(self, class_gt, class_pred):
        items = []
        img_names = self.cm_img_names.loc[class_gt, class_pred]
        for k in img_names:
            info = self.match_info[k]
            item = {
                "img_and_object_name": k,
                "img_name": info["img_name"],
                "img_info_gt": info["img_info_gt"],
                "img_info_pred": info["img_info_pred"],
                "label_id_gt": info["label_id_gt"],
                "label_id_pred": info["label_id_pred"],
                "iou": info["bbox_iou"],
                "classes_gt": info["classes_gt"],
                "classes_pred": info["classes_pred"],
                "scores_pred": info["scores_pred"]
            }

            gt_classes = set(info["classes_gt"])
            pred_classes = set(info["classes_pred"])
            if self.is_multilabel:
                item["tp"] = len(gt_classes & pred_classes)
                item["fp"] = len(pred_classes - gt_classes)
                item["fn"] = len(gt_classes - pred_classes)
            else:
                item["is_correct"] = gt_classes == pred_classes

            items.append(item)
        return items

    def detect_task_type(self):
        for i, (dataset_name, item) in enumerate(self.ds_match.items()):
            if item["dataset_matched"] != "both":
                continue
            for img_item in item["matched"]:
                img_gt, img_pred = img_item["left"], img_item["right"]
                img_gt: sly.ImageInfo
                img_pred: sly.ImageInfo
                ann_gt: sly.Annotation = self.id2ann[img_gt.id]
                ann_pred: sly.Annotation = self.id2ann[img_pred.id]

                bboxes_gt, classes_gt, label_ids_gt = collect_labels_with_tags(ann_gt.labels, self.classes)                

                if any([len(tags) > 1 for tags in classes_gt]):
                    return True
                
        return False
