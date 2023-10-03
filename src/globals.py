import supervisely as sly
from dotenv import load_dotenv
import os
from src.classification_with_detection_utils import Metrics

api = sly.Api()

project_id_gt = None
project_id_pred = None
is_tags_selected = False
current_row_items = []
selected_tags_matched = None

metrics: Metrics = None

if sly.is_development():

    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    project_id_gt = os.environ.get("project_id_gt")
    project_id_pred = os.environ.get("project_id_pred")
