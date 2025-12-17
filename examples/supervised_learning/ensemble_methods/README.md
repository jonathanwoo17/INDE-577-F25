# Ensemble Methods: Bagging Classifier (Bootstrap Aggregating)

This README is for the bagging classifier implemented in
`src/rice_ml/supervised_learning/ensemble_methods.py` (importable as `bagging_classifier`).


## Intuition

Bagging reduces variance by training many base `decision trees` on different
**bootstrap samples** of the same dataset and averaging their predictions.
Decision trees are a common base learner because they can capture complex
patterns but are prone to overfitting—bagging stabilizes them.


## Step-by-Step Algorithm

1. **Validate hyperparameters**
   - Ensure `n_estimators ≥ 1` and `0 < max_samples ≤ 1`.
     - `n_estimators` is the number of `decision trees` in the ensemble method
     - `max_samples` is the fraction of the total training data used to bootstrap across each `decision tree`


2. **Bootstrap sampling for each estimator**
   - Compute the bootstrap size: `n_bootstrap = floor(max_samples * n_samples)`,
     with a minimum of 1 sample.
   - For each of the individual `decision tree` models:
     - Draw data randomly **with replacement** from the training set.
     - Clone a new `decision_tree`
     - Fit the tree on the sampled training data and classes
   - Store each fitted tree for later use 

3. **Predict probabilities** (`predict_proba`)
   - For each trained tree, obtain class probabilities for training data point.
   - **Average** the per-class probabilities across all trees to produce the
     ensemble probability estimates.

4. **Predict class labels** (`predict`)
   - Get averaged probabilities.
   - Take the **argmax** over classes for each sample and map indices back to the set of classes for the final labels.

5. **Aggregate feature importances** (`feature_importances`)
   - After fitting, collect feature importances from each tree.
   - Compute the mean importance per feature to summarize how strongly each
     feature influenced the ensemble decisions.

---

## Why Bagging Helps

- **Variance reduction:** Averaging many high-variance trees yields a more stable
  predictor.
- **Decreased sensitivity to noise:** Bootstrap sampling exposes each tree to different
  subsets, reducing sensitivity to any single noisy example.



