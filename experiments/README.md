# Experimental workflow

The experiment directories follow the order in which information flows through
the study. There is deliberately no table/figure-generation stage in this
workflow.

1. `plm_finetuning/`: fine-tune five PLM folds from the external base model.
2. `feature_extraction/`: generate DDE and fold-specific PLM features.
3. `sampling/`: apply branch-specific improved QC-SMOTE inside a training fold.
4. `branch_models/`: fit GRANDE and TabICLv2 and generate strict OOF branch
   probabilities.
5. `fusion/`: fit the fixed seven-input logistic-regression model from branch
   probabilities.
6. `ablation/`: evaluate structural and sampling components under the locked
   split and threshold.
7. `baselines/`: evaluate comparison methods on the same fixed data split.
8. `explainability/`: analyze the PLM, DDE, and fusion levels.

Each subdirectory states its purpose, inputs, outputs, command, runtime, and
known limitations. Generated arrays and model files are written under
`results/` and remain outside Git by default.
