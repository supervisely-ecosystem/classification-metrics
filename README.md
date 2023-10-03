<div align="center" markdown>

<img src="" />


# Object Classification Metrics

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Use">How to use</a> •
  <a href="#Related-Apps">Related Apps</a> •
  <a href="#Screenshot">Screenshot</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/.png)](https://supervise.ly)

</div>

# Overview
This app is useful when you have two models in pipeline: Detector + Classifier, that can be achieved with [Apply Detection and Classification Models](https://ecosystem.supervisely.com/apps/apply-det-and-cls-models-to-project), allowing you to evaluate your classifier by comparing ground truth annotations with predictions. In this app predicted tags (classes) must be assigned to the objects (Rectangles), instead of whole images, like in [Image Classification Metrics](https://ecosystem.supervisely.com/apps/classification-metrics) app. The app also supports single-label and multi-label classification tasks.

**Application key points:**

- Supports single-label and multi-label classification tasks (auto-detected).
- Supports auto-generated suffixes, like "_nn".
- Explore multi-class confusion matrix, and see the common metrics like precision, recall, f1 for each class and in overall.

# How to use
0. Be sure that you have **two projects** with the same dataset names and image names to compare. Unmatched datasets and image names will not be included to metrics calculation. Also be sure that objects contains tags.
1. Launch the app. Select two projects: with ground truth labels and with model predictions.
2. Match projects. The app matches images in both projects and produces detailed report about matched pairs, allowing to validate the data.
3. You can match classes and select which will be used in metrics calculation. If your tags from prediction project have a suffix (e.g. "_nn") that might be auto-generated in some apps, you should make sure that they are matched too. If some tags are not matched, input the correct suffix.
4. Click "Calculate" and explore the confusion matrix. Click on the cells to see per-image stats for any interesting cases, where the model gets right or wrong. You can display images with target and predicted tags by clicking on rows in the Per-image table.
5. Also, check the tabs "Overall" and "Per class" to see the metrics like precision, recall, f1-score and others.


# Related apps

1. [Apply Detection and Classification Models](https://ecosystem.supervisely.com/apps/apply-det-and-cls-models-to-project)

2. [Train MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/train) app to train classification model on your data 
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/train" src="https://i.imgur.com/mXG6njU.png" width="350px" style='padding-bottom: 10px'/>

3. [Serve MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/serve) app to load classification model to be applied to your project
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/serve" src="https://i.imgur.com/CU8XHdQ.png" width="350px" style='padding-bottom: 10px'/>

# Screenshot

<img src="" />
