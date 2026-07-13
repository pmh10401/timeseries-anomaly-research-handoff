# Exp137 Operational Flow

## Plain-Language Flow

```text
SQLite DB
  |
  +--> TRAIN normal series only
          |
          v
   Build the normal-state reference
   (normalization and train-only score calibration)
          |
          v
   1st detection
   (ROCKET + KNN local-gap, Spectrogram + PCA + KNN)
          |
          v
   Is there an anomaly candidate?
      | no ----------------------------> No alert
      |
      | yes
      v
   Cross confirmation
   (independent ROCKET + KNN local-gap)
      |
      +--> independent analysis points to the same index
      |       --> Hard alert: automatic user notification
      |
      +--> not enough for automatic alert
              |
              v
          Supplementary confirmation
          (third ROCKET + KNN local-gap)
              |
              +--> additional support --> Priority review
              |
              +--> otherwise ----------> Standard review
```

## Code Mapping

```text
Exp93 candidate generation
  -> Exp133 high / standard confidence tiers
  -> Exp135 supplementary review confirmation
  -> Exp137 route_tiers()
```

## Exp137 Route Definition

```python
hard = set(high)
standard_review = set(standard) - hard
priority_review = set(priority) - hard - standard_review
no_alert = all_test_indices - hard - standard_review - priority_review
```

The code enforces valid test-index bounds. The tiers are mutually exclusive.

## Presentation Vocabulary

| Avoid in company presentation | Use instead |
|---|---|
| Block A | 1st detection / primary candidate evidence |
| Block B | cross confirmation / independent evidence |
| Block C | supplementary confirmation / additional evidence |
| local-gap score only | KNN local-gap anomaly score |
| selected indices | detected time-series samples |
