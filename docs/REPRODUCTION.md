# Reproduction scope

## Available now: finalized-model inference

The repository contains the complete inference implementation for:

- sequence validation and FASTA parsing;
- exact 400-dimensional DDE extraction;
- five-checkpoint PLM representation extraction;
- loading the fitted PLM and DDE branch models;
- construction of the seven fusion inputs;
- fixed-threshold prediction through the fitted fusion model;
- web and command-line prediction interfaces.

With the external PLM base model, five fold checkpoints, and fitted artifacts present, another user can reproduce predictions from the finalized model.

## Not yet packaged: complete training reconstruction

The current repository does not yet include a reviewed, one-command pipeline that starts from raw labeled sequences and reconstructs every fitted object. A complete training release would additionally require:

1. the exact fixed training and independent-test sequence files;
2. the deterministic five-fold split definitions;
3. PLM fine-tuning with the raw 15% early-stopping split inside each training fold;
4. fold-local improved QC-SMOTE processing;
5. strict out-of-fold fitting and prediction for both branches;
6. finalized GRANDE and TabICLv2 training settings;
7. seven-input logistic-regression fusion fitting from out-of-fold branch probabilities;
8. checksum and metric regression tests for all regenerated artifacts.

Until these items are consolidated and independently checked, this repository should be described as a finalized-model inference release, not a complete raw-data training reproduction package.

## Evaluation rule

Independent-test labels must remain unused during model fitting, hyperparameter selection, early stopping, probability fusion, and threshold selection. They are used only once for final reporting.

