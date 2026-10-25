# Model architecture

## PLM branch

The base model is the local PLM-interact human STRING V12 checkpoint. It uses
an ESM architecture with 33 transformer layers, hidden size 1280, 20 attention
heads, and approximately 650 million parameters. PLM-interact was pretrained
for paired-protein interaction representation; it is not one of the five AIP
fine-tuned checkpoints.

Each peptide is paired with the TNF-alpha sequence and tokenized with the local pretrained protein language model. Five models were fine-tuned on five training folds. During inference, the 1280-dimensional CLS representation from the penultimate hidden layer is extracted from each fold checkpoint and averaged across the five folds.

The averaged representation is transformed by the fitted PLM scaler and
evaluated by the finalized GRANDE classifier. In the training workflow, the
classifier is trained fold-by-fold with strict OOF predictions; the final
independent-test classifier is fit from the fixed training split and uses an
internal 15% early-stopping subset. The output is the PLM positive-class
probability, `p_plm`.

## DDE branch

Each sequence is converted into 400 DDE features, one for every ordered pair of the 20 standard amino acids. DDE standardizes the observed dipeptide frequency against its codon-usage-derived expected mean and variance.

The DDE vector is transformed by the fitted DDE scaler and evaluated by the
finalized TabICLv2 classifier. In the training workflow, the classifier is
trained fold-by-fold with strict OOF predictions; the final independent-test
classifier is fit on the full training split. The output is the DDE
positive-class probability, `p_dde`.

## Probability fusion

The fixed logistic-regression model receives seven inputs:

1. `p_plm`;
2. `p_dde`;
3. `abs(p_plm - p_dde)`;
4. `(p_plm + p_dde) / 2`;
5. `p_plm * p_dde`;
6. `abs(p_plm - 0.5)`;
7. `abs(p_dde - 0.5)`.

The final probability is classified at a fixed threshold of 0.5.

## Final independent-test performance

The manifest records the following deployment performance on the fixed independent test set of 419 sequences:

| AUC | ACC | MCC | Sn | Sp | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8559 | 0.8091 | 0.6095 | 0.7667 | 0.8410 | 0.7841 | 0.7753 |

These values describe the fixed archived model and should not be interpreted as performance on arbitrary peptide distributions.
