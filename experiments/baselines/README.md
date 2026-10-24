# Baseline comparisons

## Purpose

Evaluate comparison methods on the same 3,775/419 sequence split. The complete
metric table is stored in `evaluation/baseline_metrics.csv`.

## Methods evaluated locally

- AIP-TranLAC;
- AIPStack;
- IF-AIP-adapted;
- PepNet;
- DeepAIP-adapted;
- PPTPP-adapted;
- TriStack-strict-adapted.

## Meaning of `adapted`

An `adapted` label means that public architecture or source code was used but a
repository-specific data/evaluation adapter was required. Examples include
missing author feature tables, Windows-only Weka paths, or fold-local feature
preparation needed to prevent leakage. Such rows must not be described as an
exact execution of the author's original training pipeline.

## Required reporting rule

State the public source URL, commit or release, local adaptation, fixed split,
five-fold protocol, threshold, and software environment for every rerun.
Numbers copied directly from another paper must be placed in a separate
literature-reported block rather than mixed with local reruns.

The adapters depend on third-party repositories. They are not vendored here
until each upstream license and redistribution condition has been checked.
