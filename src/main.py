import pandas as pd

import supervisely as sly
from supervisely.app import DataJson
from supervisely.app.widgets import (
    ConfusionMatrix,
    Container,
    Card,
    SelectDataset,
    Button,
    MatchDatasets,
    Field,
    MatchTagMetas,
    Input,
    Table,
    Tabs,
    NotificationBox,
    GridGallery,
    Text,
    Switch,
    InputNumber
)

from src import utils, metric_utils
from src import globals as g
from src.classification_with_detection_utils import Metrics


api = g.api


### 1. Select Datasets
select_dataset_gt = SelectDataset(project_id=g.project_id_gt, multiselect=True)
select_dataset_gt._all_datasets_checkbox.check()
select_dataset_pred = SelectDataset(project_id=g.project_id_pred, multiselect=True)
select_dataset_pred._all_datasets_checkbox.check()
card_gt = Card(
    "Ground Truth Project", "Select project with ground truth labels", content=select_dataset_gt
)
card_pred = Card(
    "Prediction Project", "Select project with predicted labels", content=select_dataset_pred
)
select_dataset_btn = Button("Select Datasets")
select_datasets_container = Container([card_gt, card_pred], "horizontal", gap=15)

change_datasets_btn = Button("Change Datasets", "info", "small", plain=True)

### 2. Match Datasets
match_datasets = MatchDatasets()
match_datasets_btn = Button("Match")
match_datasets_warn = NotificationBox(
    "Not matched.",
    "Datasets don't match. Please, check your dataset/image names. They must match",
    box_type="warning",
)
match_datasets_container = Container([match_datasets, match_datasets_btn, match_datasets_warn])
match_datasets_card = Card(
    "Match datasets",
    "Datasets and their images are compared by name. Only matched pairs of images are used in metrics.",
    True,
    match_datasets_container,
)


@match_datasets_btn.click
def on_match_datasets():
    match_datasets_warn.hide()
    project_id_gt = select_dataset_gt._project_selector.get_selected_id()
    project_id_pred = select_dataset_pred._project_selector.get_selected_id()
    if project_id_gt is None or project_id_pred is None:
        raise Exception("Please, select a project and datasets")
    ds_gt = api.dataset.get_list(project_id_gt)
    ds_pred = api.dataset.get_list(project_id_pred)

    match_datasets.set(ds_gt, ds_pred, "GT datasets", "Pred datasets")
    ds_matching = match_datasets.get_stat()
    if len(utils.validate_dataset_match(ds_matching)) == 0:
        match_datasets_warn.show()
        return

    g.project_id_gt = project_id_gt
    g.project_id_pred = project_id_pred

    rematch_tags()

    match_tags_card.uncollapse()
    change_datasets_btn.show()
    match_datasets_btn.disable()
    card_gt.lock()
    card_pred.lock()


@change_datasets_btn.click
def on_change_datasets():
    reset_widgets()


### 3. Match Tags
match_tags_input = Input("_nn")
match_tags_rematch_btn = Button("Rematch tags", button_size="small")
match_tags_rematch_c = Container([match_tags_input, match_tags_rematch_btn])
match_tags_input_f = Field(
    match_tags_rematch_c,
    "Input suffix for Pred tags",
    "If there is no matching you want due to suffix in tags (like \"_nn\"), you can input it manually.",
)
match_tags = MatchTagMetas(selectable=True)
match_tags_btn = Button("Select")
match_tags_notif_note = NotificationBox("Note:", box_type="info")
match_tags_notif_warn = NotificationBox("Not selected.", box_type="warning")
match_tags_container = Container(
    [match_tags_input_f, match_tags, match_tags_btn, match_tags_notif_note, match_tags_notif_warn]
)
match_tags_card = Card(
    "Select tags",
    "Choose tags/classes that will be used for metrics.",
    True,
    match_tags_container,
)


@match_tags_rematch_btn.click
def rematch_tags():
    g.tags_gt = sly.ProjectMeta.from_json(api.project.get_meta(g.project_id_gt)).tag_metas
    g.tags_pred = sly.ProjectMeta.from_json(api.project.get_meta(g.project_id_pred)).tag_metas
    g.suffix = match_tags_input.get_value()
    g.tags_pred_filtered = utils.filter_tags_by_suffix(g.tags_pred, g.suffix)
    match_tags.set(g.tags_gt, g.tags_pred_filtered, "GT tags", "Pred tags", suffix=g.suffix)


@match_tags_btn.click
def on_select_tags():
    match_tags_notif_note.hide()
    match_tags_notif_warn.hide()
    selected_tags = match_tags.get_selected()
    selected_tags_matched = list(
        filter(lambda x: x[0] is not None and x[1] is not None, selected_tags)
    )
    if not g.is_tags_selected:
        if selected_tags_matched:
            match_tags_notif_note.description = (
                f"{len(selected_tags_matched)} matched tags will be used for metrics."
            )
            match_tags_notif_note.show()
            match_tags_btn.text = "Reselect tags"
            match_tags_btn._plain = True
            match_tags_btn._button_size = "small"
            match_tags_btn.update_data()
            DataJson().send_changes()
            additional_params_card.unlock()
            metrics_btn.enable()
            match_tags_input.disable()
            match_tags_rematch_btn.disable()
            g.is_tags_selected = True
        else:
            match_tags_notif_warn.description = f"Please, select at least 1 matched tag."
            match_tags_notif_warn.show()
    else:
        additional_params_card.lock()
        metrics_card.collapse()
        metrics_btn.disable()
        match_tags_input.enable()
        match_tags_rematch_btn.enable()
        match_tags_btn.text = "Select"
        match_tags_btn._plain = False
        match_tags_btn._button_size = None
        match_tags_btn.update_data()
        DataJson().send_changes()
        g.is_tags_selected = False
    g.selected_tags_matched = selected_tags_matched


### 4. Confusion Matrix & Metrics


iou_slider = InputNumber(0.5, 0., 1.0, 0.05)
iou_slider_f = Field(iou_slider, "Box IoU threshold", "If IoU of a ground truth bbox and a predicted bbox is greater than this value, the object will be considered for classification metrics.")
metrics_btn = Button("Calculate metrics")
task_notif_box = NotificationBox()

additional_params_card = Card(
    title="Additional parameters",
    collapsable=False,
    content=Container([iou_slider_f, task_notif_box, metrics_btn], gap=15),
)

confusion_matrix_widget = ConfusionMatrix()
metrics_overall_table = Table()
metrics_overall_table_f = Field(metrics_overall_table, "Overall project metrics")
metrics_per_class_table = Table()
metrics_per_class_table_f = Field(metrics_per_class_table, "Per-class metrics")
multilable_mode_switch = Switch(switched=False)
multilable_mode_text = Text(
    "<i style='color:gray;'>(more details in <a href='https://ecosystem.supervise.ly/apps/classification-metrics#Confusion-Matrix-implementation-details-for-multi-label-task' target='_blank'>Readme</a>)</i>"
)
multilable_mode_desc = Container([multilable_mode_text, multilable_mode_switch])
multilable_mode_switch_f = Field(
    multilable_mode_desc,
    "Count all combinations for misclassified tags",
    "Turn on to get more insights about the classes the model most often confuses. "
    "Note, if enabled, the values in the table will not represent the true number of incorrectly classified images. Instead, it will indicate the number of misclassified tags.",
)
metrics_tab_confusion_matrix = Container(
    [confusion_matrix_widget, multilable_mode_switch_f], gap=20
)
match_rate_info = NotificationBox(box_type="success")
metrics_tabs = Tabs(
    ["Confusion matrix", "Per class", "Overall"],
    [metrics_tab_confusion_matrix, metrics_per_class_table_f, metrics_overall_table_f],
)
metrics_card = Card(
    "Confusion Matrix & Metrics",
    "",
    collapsable=True,
    content=Container([match_rate_info, metrics_tabs]),
)
metrics_per_image = Table()
per_image_notification_box = NotificationBox(
    title="Table for clicked datapoint from Confusion Matrix", description="Click on the Confusion Matrix to see results."
)
card_per_image_table = Card(
    title="Per image metrics",
    description="Click on table row to preview image",
    collapsable=True,
    content=Container([per_image_notification_box, metrics_per_image], gap=5),
)
current_image_tag = Text()
images_gallery = GridGallery(
    2, show_opacity_slider=False, enable_zoom=True, resize_on_zoom=True, sync_views=True
)
card_img_preview = Card(
    title="Image preview",
    description="Ground Truth (left) and Prediction (right)",
    collapsable=True,
    content=Container([current_image_tag, images_gallery], gap=5),
)

@metrics_btn.click
def on_metrics_click():
    ds_match = match_datasets.get_stat()
    
    iou_threshold = iou_slider.get_value()
    id2ann = utils.download_annotations_gt_and_pred(api, g.project_id_gt, g.project_id_pred)
    metrics = Metrics(id2ann, ds_match, g.selected_tags_matched, iou_threshold)
    metrics.calculate()

    update_metric_widgets(metrics)
    g.metrics = metrics


def update_metric_widgets(metrics: Metrics):
    if metrics.is_multilabel:
        task_notif_box.title = "Multi-label classification"
        task_notif_box.description = "Calculated as a multi-label classification (one or more objects have more than one tag)."
    else:
        task_notif_box.title = "Single-label classification"
        task_notif_box.description = "Calculated as a single-label classification."
    task_notif_box.show()

    confusion_matrix_widget._update_matrix_data(metrics.confusion_matrix)
    confusion_matrix_widget.update_data()
    DataJson().send_changes()

    metrics_overall_table._update_table_data(input_data=metrics.overall_metrics_df)
    metrics_overall_table.update_data()
    DataJson().send_changes()

    metrics_per_class_table._update_table_data(input_data=metrics.per_class_metrics_df)
    metrics_per_class_table.update_data()
    DataJson().send_changes()

    confusion_matrix_widget.show()
    metrics_overall_table.show()
    metrics_per_class_table.show()
    if metrics.is_multilabel:
        multilable_mode_switch_f.show()
    metrics_card.uncollapse()

    match_rate_info.title = f"Detection match rate = {round(metrics.total_matched/metrics.total_objects*100, 2)}% ({metrics.total_matched} of {metrics.total_objects} objects are matched by IoU)."


@confusion_matrix_widget.click
def on_confusion_matrix_click(cell: ConfusionMatrix.ClickedDataPoint):
    class_gt = cell.row_name
    class_pred = cell.column_name
    metrics = g.metrics
    items = metrics.query_metrics(class_gt, class_pred)
    g.current_row_items = items

    if metrics.is_multilabel:
        columns = ["IMAGE NAME", "BOX IOU", "TP", "FP", "FN", "PREVIEW"]
    else:
        columns = ["IMAGE NAME", "BOX IOU", "IS CORRECT", "PREVIEW"]

    rows = []
    for item in items:
        row = []
        row.extend([
            item["img_and_object_name"],
            round(item["iou"], 4),
        ])
        if metrics.is_multilabel:
            row.extend([
                item["tp"],
                item["fp"],
                item["fn"],
            ])
        else:
            row.append(item["is_correct"])
        row.append(Table.create_button("Preview"))
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    metrics_per_image.read_pandas(df)
    metrics_per_image.show()

    cell_value = int(cell.cell_value)
    if cell.row_name != "None" and cell.column_name != "None":
        per_image_notification_box.description = f"{int(cell_value)} images with Ground Truth: \"{cell.row_name}\", and Predicted: \"{cell.column_name}\"."
    elif cell.row_name != "None":
        per_image_notification_box.description = f"{int(cell_value)} images with tag \"{cell.row_name}\" in Ground Truth, that is missing in prediction."
    elif cell.column_name != "None":
        per_image_notification_box.description = f"{int(cell_value)} images without tag \"{cell.column_name}\" in Ground Truth, but it was predicted."
    card_per_image_table.uncollapse()

    set_img_to_gallery(items[0], g.metrics.id2ann)
    images_gallery.show()
    card_img_preview.uncollapse()


@multilable_mode_switch.value_changed
def on_mode_changed(is_checked):
    on_metrics_click()


@metrics_per_image.click
def select_image_row(cell: Table.ClickedDataPoint):
    set_img_to_gallery(g.current_row_items[cell.row_index], g.metrics.id2ann)


def set_img_to_gallery(item, id2ann):
    img_info: sly.ImageInfo = item["img_info_gt"]
    ann_gt: sly.Annotation = id2ann[item["img_info_gt"].id]
    ann_pred: sly.Annotation = id2ann[item["img_info_pred"].id]
    classes_gt, classes_pred, scores_pred = item["classes_gt"], item["classes_pred"], item["scores_pred"]
    pred_label_id, gt_label_id = item["label_id_pred"], item["label_id_gt"]

    ann_gt = utils.filter_annotation_by_label_id(ann_gt, gt_label_id)
    ann_pred = utils.filter_annotation_by_label_id(ann_pred, pred_label_id)
    assert len(ann_gt.labels) and len(ann_pred.labels)

    images_gallery.loading = True
    img_url = img_info.full_storage_url

    gt_title = "<br>".join(["Ground Truth:"]+classes_gt)
    if scores_pred:
        pred_title = [f"{cls}: {round(score, 4)}" for cls, score in zip(classes_pred, scores_pred)]
    else:
        pred_title = [f"{cls}" for cls in classes_pred]
    pred_title = "<br>".join(["Predicted:"]+pred_title)

    images_gallery.clean_up()
    images_gallery.append(img_url, ann_gt, title=gt_title, zoom_to=gt_label_id, zoom_factor=1.5)
    images_gallery.append(img_url, ann_pred, title=pred_title, zoom_to=pred_label_id, zoom_factor=1.5)

    images_gallery.loading = False
    current_image_tag.text = f"<b>Image:</b> {img_info.name}"


def reset_widgets():
    change_datasets_btn.hide()
    card_gt.unlock()
    card_pred.unlock()
    match_datasets_btn.enable()
    match_tags_btn.enable()
    match_datasets.set()
    match_tags.set()
    match_tags_card.collapse()
    match_tags_notif_note.hide()
    match_tags_notif_warn.hide()
    metrics_btn.disable()
    task_notif_box.hide()
    confusion_matrix_widget.hide()
    metrics_overall_table.hide()
    metrics_per_class_table.hide()
    metrics_per_image.hide()
    images_gallery.hide()
    metrics_card.collapse()
    card_per_image_table.collapse()
    card_img_preview.collapse()
    match_tags_input.enable()
    g.is_tags_selected = False
    match_tags_btn.text = "Select"
    match_tags_btn._plain = False
    match_tags_btn._button_size = None
    match_tags_btn.update_data()
    DataJson().send_changes()
    match_tags_rematch_btn.enable()
    match_datasets_warn.hide()
    multilable_mode_switch.off()
    multilable_mode_switch_f.hide()
    additional_params_card.lock()


### FINAL APP
final_container = Container(
    [
        select_datasets_container,
        change_datasets_btn,
        match_datasets_card,
        match_tags_card,
        additional_params_card,
        metrics_card,
        Container(
            [card_per_image_table, card_img_preview], direction="horizontal", fractions=[5, 5]
        ),
    ],
    gap=15,
)
app = sly.Application(final_container)
reset_widgets()
