# Evaluation records

## Files

- `final_model_metrics.csv`: final independent-test metrics.
- `independent_test_predictions.csv`: all 419 labels, branch probabilities,
  fused probabilities, and predictions.
- `baseline_metrics.csv`: locally evaluated comparison methods.
- `base_probabilities/training_oof.csv`: strict OOF branch probabilities used
  to fit the fusion model.
- `base_probabilities/independent_test.csv`: branch probabilities used for the
  final independent-test evaluation.

## Metric order

All summary tables use `AUC, ACC, MCC, Sn, Sp, Pre, F1`. Sensitivity (`Sn`) is
recall for class 1, and specificity (`Sp`) is recall for class 0.

## Independent-test policy

The independent test contains 419 sequences: 180 positives and 239 negatives.
Its labels are retained here for transparent verification of reported metrics,
not for fitting or model selection.
