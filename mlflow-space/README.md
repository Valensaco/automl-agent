---
title: AutoML Agent MLflow Log
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# AutoML Agent: MLflow experiment log

A snapshot of the real MLflow tracking data from the AutoML Agent: two AutoGen
agents (Claude) that choose, train and evaluate models on the UCI credit card
default dataset within a fixed budget.

Every run the agents made is here with its settings, validation metrics and
the planner's reasoning (in the `notes` tag). The `FINAL-test` run is the one
time the test set was used.
