import supervisely as sly
from supervisely import TagMetaCollection


def download_annotations(api: sly.Api, project_id):
    id2ann = {}
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    datasets = api.dataset.get_list(project_id)
    for dataset in datasets:
        ann_infos = api.annotation.get_list(dataset.id)
        for ann_info in ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, project_meta)
            id2ann[ann_info.image_id] = ann
    return id2ann


def download_annotations_gt_and_pred(api: sly.Api, project_id_gt, project_id_pred):
    id2ann = download_annotations(api, project_id_gt)
    id2ann.update(download_annotations(api, project_id_pred))
    return id2ann


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


def validate_dataset_match(ds_matching):
    matched_ds = []
    for ds_name, ds_values in ds_matching.items():
        if ds_values["dataset_matched"] == "both" and len(ds_values["matched"]):
            matched_ds.append(ds_name)
    return matched_ds


def filter_annotation_by_label_id(ann: sly.Annotation, label_id) -> sly.Annotation:
    return ann.clone(labels=[label for label in ann.labels if label.geometry.sly_id == label_id])