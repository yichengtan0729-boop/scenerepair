# SceneRepair method specification

## Problem

Given multiple views `I`, a counterfactual spatial question `q`, and options `Y`, a frozen VLM produces a typed trajectory `R=(S_0,...,S_T)` and an answer distribution. SceneRepair must detect and repair a faulty transition without the ground-truth answer.

## Typed states

Each state records step type, involved objects, reference frame, operation, relations before and after, visibility, evidence views, and confidence. The state is the causal unit; arbitrary natural-language sentences are not treated as interchangeable units.

## Transition consistency

The default hybrid critic combines:

- symbolic schema and state-transition checks;
- image-conditioned VLM verification;
- an optional learned lightweight critic.

Weights are normalized over enabled components.

## Interventions

For low-consistency candidate transitions, the method applies:

- reference-frame swap;
- inverse operation;
- relation flip;
- object-identity swap;
- visibility flip;
- view dropout.

For every intervention, it recomputes the answer distribution and transition consistency.

## Localization

The per-intervention causal effect is the weighted sum of Jensen-Shannon divergence in answer distributions and positive consistency improvement. The earliest transition passing anomaly and causal thresholds is selected. This prioritizes an upstream cause over a downstream symptom.

## Repair

The verified prefix is copied exactly. Multiple suffix candidates are generated from the localized transition. Each candidate is ranked by global consistency, answer confidence, and minimality. The original trace is an implicit no-op candidate. The repair is applied only when the best candidate exceeds it by the configured margin.

## Label-free guarantee

`SpatialExample.answer` is not included in any generation, critic, scoring, intervention, or repair prompt. It is consumed only in `evaluation.py`. The unit test `test_label_free.py` verifies that changing the stored label leaves diagnosis unchanged.
