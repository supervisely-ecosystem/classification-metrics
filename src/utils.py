import pandas as pd
from itertools import chain
from supervisely import TagMetaCollection, ImageInfo


def collect_matching(ds_matching, tags_gt, tags_pred, selected_tags):
    selected_tags = list(filter(lambda x: bool(x[0]) and bool(x[1]), selected_tags))
    cls2clsGT = dict(map(reversed, selected_tags))
    cls2clsGT.update(dict(map(lambda x: (x[0], x[0]), selected_tags)))
    id2tag_gt = tags_gt.get_id_mapping()
    id2tag_pred = tags_pred.get_id_mapping()
    tagId2classGT_gt = lambda tagId: cls2clsGT[id2tag_gt[tagId].name]
    tagId2classGT_pred = lambda tagId: cls2clsGT[id2tag_pred[tagId].name]
    names_keep = set(chain(*selected_tags))
    ids_keep = set(
        [id for id, tag in chain(id2tag_gt.items(), id2tag_pred.items()) if tag.name in names_keep]
    )

    img2classes_gt = {}
    img2classes_pred = {}
    img_name_2_img_info_gt = {}
    img_name_2_img_info_pred = {}
    ds_name_2_img_names = {}
    for ds_name, ds_values in ds_matching.items():
        if ds_values["dataset_matched"] != "both":
            continue
        ds_name_2_img_names[ds_name] = []
        for img_pair in ds_values["matched"]:
            img_gt, img_pred = img_pair["left"], img_pair["right"]
            filtered_classes_gt = [
                tagId2classGT_gt(tag["tagId"]) for tag in img_gt.tags if tag["tagId"] in ids_keep
            ]
            has_confidence = all([isinstance(tag.get("value"), float) for tag in img_pred.tags])
            img_pred_tags = img_pred.tags
            if has_confidence:
                img_pred_tags = sorted(img_pred.tags, key=lambda tag: tag["value"], reverse=True)
            filtered_classes_pred = [
                tagId2classGT_pred(tag["tagId"])
                for tag in img_pred_tags
                if tag["tagId"] in ids_keep
            ]
            img2classes_gt[img_gt.name] = filtered_classes_gt
            img2classes_pred[img_pred.name] = filtered_classes_pred
            img_name_2_img_info_gt[img_gt.name] = img_gt
            img_name_2_img_info_pred[img_pred.name] = img_pred
            ds_name_2_img_names[ds_name].append(img_gt.name)

    classes = list(zip(*selected_tags))[0]  # classes == left selected tag_names
    return (
        img2classes_gt,
        img2classes_pred,
        classes,
        img_name_2_img_info_gt,
        img_name_2_img_info_pred,
        ds_name_2_img_names,
    )


def filter_imgs_without_tags_(img2tags_gt: dict, img2tags_pred: dict):
    for k, tags in list(img2tags_gt.items()):
        if len(tags) == 0:
            img2tags_gt.pop(k)
            img2tags_pred.pop(k, None)


def is_task_multilabel(img2tags_gt: dict):
    for k, tags in img2tags_gt.items():
        if len(tags) != 1:
            return True
    return False


def filter_tags_by_suffix(tags, suffix):
    # filtering "duplicated with suffix" (cat, cat_nn, dog) -> (cat_nn, dog)
    names = set([tag.name for tag in tags])
    filtered_tags = []
    for tag in tags:
        if tag.name + suffix in names:
            continue
        filtered_tags.append(tag)
    return TagMetaCollection(filtered_tags)


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


def stringify_label_tags(predicted_tags, is_multilabel, is_gt):
    final_message = ""

    for index, tag in enumerate(predicted_tags):
        value = ""
        if tag.value is not None:
            value = f":{round(tag.value, 3)}"
        if not is_multilabel and not is_gt and len(predicted_tags) > 1:
            final_message += f"top@{index + 1} — "
        final_message += f"{tag.name}{value}<br>"

    return final_message


def get_preview_image_pair(img_info_gt, img_info_pred, img_tags_gt, img_tags_pred, is_multilabel):
    return [
        {
            "url": img_info_gt.full_storage_url,
            "title": stringify_label_tags(img_tags_gt, is_multilabel, True),
        },
        {
            "url": img_info_pred.full_storage_url,
            "title": stringify_label_tags(img_tags_pred, is_multilabel, False),
        },
    ]


def validate_dataset_match(ds_matching):
    matched_ds = []
    for ds_name, ds_values in ds_matching.items():
        if ds_values["dataset_matched"] == "both" and len(ds_values["matched"]):
            matched_ds.append(ds_name)
    return matched_ds
