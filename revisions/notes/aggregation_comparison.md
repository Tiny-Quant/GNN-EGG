# Aggregated explainer logic change

## Previous behavior
- Aggregation enforced that either a `class_getter` or the explainee model was provided; when no custom getter was passed, the helper always called `_predict_class_from_model` to group motifs. 【F:revisions/new_src/agg_instance.py†L335-L355】【F:revisions/new_src/agg_instance.py†L388-L427】
- Because grouping depended solely on predictions, any graph the explainee misclassified contributed motifs to the wrong class bucket, and classes with only misclassified samples produced empty motif batches. 【F:revisions/new_src/agg_instance.py†L388-L432】

## Current behavior
- The helper now groups motifs by the explainee’s prediction when one is available, keeping the focus on explaining model behavior. If no explainee is supplied it falls back to scalar `data.y` labels, and it raises a clearer error when neither path is available. 【F:revisions/new_src/agg_instance.py†L373-L404】【F:revisions/new_src/agg_instance.py†L474-L493】
- When labels exist, they are tracked to detect missing predicted classes; if a label class receives no motifs from the model, the helper warns and fills that bucket using the label grouping so downstream evaluation does not crash. 【F:revisions/new_src/agg_instance.py†L495-L506】
- A second warning is emitted when every labeled sample is misclassified, highlighting that the aggregated motifs reflect only the model’s (incorrect) view. 【F:revisions/new_src/agg_instance.py†L508-L517】

## New quality-of-life helper
- Aggregation can now surface progress while mining motifs. Pass `progress=True` to `aggregate_instance_explanations` to get a tqdm bar when available, or provide a custom callback `(count, total_or_none)` to integrate with notebook logging. 【F:revisions/new_src/agg_instance.py†L251-L332】【F:revisions/new_src/agg_instance.py†L556-L605】

## Why MUTAG worked but other datasets failed
- On MUTAG, the explainee’s predictions aligned with ground truth, so both classes accumulated motifs and downstream evaluation received non-empty generated batches. 【F:revisions/new_src/agg_instance.py†L439-L493】
- On datasets like BA-2MOTIFS, the explainee misclassified many graphs; motifs were logged under the wrong predicted class, leaving one class empty and causing `run_eval_summary` to raise a “graphs must contain at least one generated graph” error for the missing batch. 【F:revisions/new_src/agg_instance.py†L439-L493】
- The updated logic keeps prediction-based grouping but backfills empty label classes with a warning, preserving the model-centric explanation while avoiding empty batches on misclassified datasets. 【F:revisions/new_src/agg_instance.py†L495-L517】
