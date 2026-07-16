# SceneRepair: necessity-sufficiency causal-root method

## Problem

A frozen VLM receives multi-view images, a counterfactual spatial question, and answer options. It produces a typed state trajectory and an answer distribution. SceneRepair diagnoses and repairs an upstream causal error without using the benchmark label during inference.

## Typed causal dependency graph

Each state contains a step type, objects, reference frame, operation, relations, visibility, evidence views, and direct prerequisite indices (`depends_on`). Missing dependencies are inferred from the typed order and object overlap. The result is a directed acyclic graph whose edges encode which earlier commitments are required by each state.

## Transition anomaly

A hybrid critic combines symbolic checks, image-conditioned VLM verification, and an optional learned critic. The anomaly of node `t` is `1 - consistency_t`.

## Paired necessity and sufficiency interventions

For candidate node `t`, SceneRepair applies structured interventions to reference frame, operation, relations, object identity, visibility, and visual evidence.

- **Necessity:** perturbing `t` should change the answer distribution or reduce consistency over the descendant closure.
- **Sufficiency:** a constraint-improving intervention at `t` should restore consistency over the descendant closure without changing ancestors.
- **Interventional consistency:** semantically equivalent lexical realizations of the same structured intervention should yield stable effects and model predictions.

The causal-root score is a weighted geometric mean of anomaly, necessity, sufficiency, and interventional consistency. A node is selected only if it passes all component thresholds. Among competitive valid nodes, SceneRepair chooses an ancestor-minimal root rather than the largest downstream symptom.

## Descendant-closed minimal repair

The verified prefix is preserved. The repair prompt receives the localized root, dependency graph, and descendant closure. Candidate suffixes are ranked by consistency, answer confidence, and minimality.

## Calibrated label-free acceptance

Each original/candidate pair is scored repeatedly. SceneRepair computes a one-sided lower confidence bound for the candidate's objective gain. A repair is accepted only when the lower bound exceeds the abstention margin and the consistency gain is non-negative. The benchmark label is used only by `evaluation.py`.

## Main ablations

1. anomaly only;
2. necessity without sufficiency;
3. sequence-only localization without the dependency graph;
4. no interventional-consistency test;
5. full regeneration instead of descendant-closed repair;
6. fixed-margin acceptance instead of calibrated lower confidence bound.
