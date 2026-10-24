# Experimental workflow

The experiment directories follow the order in which information flows through
the study. There is deliberately no table/figure-generation stage in this
workflow.

1. `feature_extraction/`: generate DDE or archived-checkpoint PLM features.
2. `sampling/`: apply branch-specific improved QC-SMOTE inside a training fold.
3. `branch_models/`: fit GRANDE and TabICLv2 and generate strict OOF branch
   probabilities.
4. `fusion/`: fit the fixed seven-input logistic-regression model from branch
   probabilities.
5. `ablation/`: evaluate structural and sampling components under the locked
   split and threshold.
6. `baselines/`: evaluate comparison methods on the same fixed data split.
7. `explainability/`: analyze the PLM, DDE, and fusion levels.

Each subdirectory states its purpose, inputs, outputs, command, runtime, and
known limitations. Generated arrays and model files are written under
`results/` and remain outside Git by default.
