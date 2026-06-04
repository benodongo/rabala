# PC-HEWS — Lake Victoria Fishing Conditions Predictor

A Django application that fuses **indigenous knowledge (IK)** of Lake Victoria
fishers with **real-time scientific weather data** to forecast fishing
conditions across four classes: `Good`, `Normal`, `Risky`, `Bad`.

The system exposes three prediction modes:

| Mode | URL | What it does |
| --- | --- | --- |
| Indigenous | `/indigenous/` | Mamdani fuzzy inference over 87 elicited rules (rule-only). |
| Integrated | `/integrated/` | Indigenous verdict + Open-Meteo weather fused by a hand-tuned risk function. |
| Ensemble   | `/ensemble/`   | Calibrated ML ensemble (RandomForest + Logistic Regression) trained over indigenous + scientific features. |

This README focuses on the **Integrated** and **Ensemble** pipelines: how
they work, how they are validated, and how to inspect their performance.

---

## 1. Pipeline overview

```
        ┌──────────────────────┐         ┌────────────────────────┐
        │ Indigenous inputs    │         │ Open-Meteo current +   │
        │  · wind name         │         │ daily forecast (lat,   │
        │  · moon phase        │         │ lon)                   │
        │  · cloud condition   │         └──────────┬─────────────┘
        │  · body temperature  │                    │
        └──────────┬───────────┘                    │
                   ▼                                │
        ┌──────────────────────┐                    │
        │ Fuzzy engine          │                   │
        │ (87 Mamdani rules)    │                   │
        │  → lake_pos verdict   │                   │
        │  → posterior vector   │                   │
        │     (per-class μ)     │                   │
        └──────┬────────────┬───┘                   │
               │            │                       │
               │            ▼                       ▼
               │   ┌─────────────────────────────────────────┐
               │   │ Integrated risk function                 │
               │   │   r = max(0.5·μ_ik + 0.5·μ_sci,          │
               │   │           0.85·max(μ_ik, μ_sci))         │
               │   │   → label + risk in [0, 1]               │
               │   └─────────────────────────────────────────┘
               │
               ▼
   ┌──────────────────────────────────────────────────┐
   │ Ensemble model                                    │
   │   sci side : RandomForest.predict_proba(X_sci)    │
   │   ik  side : fuzzy posterior vector (per class)   │
   │   fusion   : LogisticRegression(class_weight=     │
   │              'balanced') with CalibratedClassif-  │
   │              ierCV (sigmoid, k-fold cross-fit)    │
   │   output   : calibrated P(class) + argmax / per-  │
   │              class threshold decision             │
   └──────────────────────────────────────────────────┘
```

---

## 2. Integrated approach

`predictor/scientific.py :: integrate_predictions(indigenous_pred, scientific_data)`

### Inputs

* `indigenous_pred` — verdict from the fuzzy engine (`Good | Normal | Risky | Bad`).
* `scientific_data` — Open-Meteo dict (`precipitation`, `wind_speed`, …) or
  `None` when the fetch fails.

### How risk is computed

Each side produces a normalised risk score in `[0, 1]`:

| Source | Formula |
| --- | --- |
| IK risk | Lookup in `RISK_MAPPING` — `Good 0.2`, `Normal 0.4`, `Risky 0.7`, `Bad 1.0`. |
| Sci risk | `0.55·min(1, precip/15) + 0.45·min(1, wind/10)` clamped to `[0, 1]`. |

Denominators (`precip/15`, `wind/10`) are tuned to Lake Victoria conditions
where **≈15 mm/h precipitation** and **≈10 m/s wind** already saturate to
"dangerous".

The two risks are fused with a **weighted blend + max-override**:

```
weighted   = 0.5 · ik_risk + 0.5 · sci_risk
combined   = max(weighted, 0.85 · max(ik_risk, sci_risk))
```

The max-override prevents a calm IK verdict from diluting a strongly
dangerous sci signal (a previous version using `0.6·ik + 0.4·sci` did
exactly that).

### Labelling

| Combined risk | Label |
| --- | --- |
| `< 0.30` | Excellent fishing conditions |
| `< 0.55` | Normal fishing conditions |
| `< 0.75` | Caution advised — risky conditions |
| `≥ 0.75` | Dangerous fishing conditions |

### Failure modes

* **Open-Meteo unavailable** → falls back to `combined_risk = ik_risk`. The
  UI shows a warning banner.
* **Unknown IK label** → defaults to `0.5` risk.

---

## 3. Ensemble approach

`predictor/scientific.py :: EnsembleModel`

### Feature engineering

For each example:

1. **Scientific features (4-dim)** — `[temp_min, temp_max, rainfall, wind_max]`
   are passed through a `StandardScaler` and then a `RandomForestClassifier`
   (100 trees), whose `predict_proba` becomes a 4-dim feature
   `[P_sci(Good), P_sci(Normal), P_sci(Risky), P_sci(Bad)]`.
2. **Indigenous features (4-dim, `MODEL_SCHEMA_VERSION = 3`)** — the
   **fuzzy posterior** `[μ(Good), μ(Normal), μ(Risky), μ(Bad)]` exposed by
   `fuzzy_logic.fuzzy_posterior()`, which normalises the per-class max
   firing strength across the rule base.

   Earlier (v2) models used a one-hot encoding of `argmax(μ)`. That made
   IK a single high-signal discrete feature and let the linear ensemble
   collapse onto IK whenever the rule base fired confidently. The
   posterior gives the model **graded evidence** — it can now learn to
   discount IK when membership is split.

These are concatenated into an 8-dim feature `[sci_probs | ik_post]`.

### Classifier

* `LogisticRegression(max_iter=1000, class_weight='balanced')`
* Trained with **balanced sample weights** so minority classes (`Good`,
  `Bad`) are not drowned by the dominant `Normal`.
* Wrapped in `CalibratedClassifierCV(method='sigmoid', cv=k)` where
  `k = min(3, smallest_class_count)`. Uses **cross-fit calibration** so
  probabilities aren't overconfident from being calibrated on training
  data. Falls back to `cv='prefit'` only when k-fold is impossible.

### Decision

`EnsembleModel.predict(ik_inputs, sci_features, adjust_thresholds=…)`:

* **Default** → `argmax` of calibrated probabilities.
* **Threshold-adjusted** → any class whose probability exceeds its
  configured threshold qualifies; the highest-prob qualifier wins. Falls
  back to argmax if none qualify.

Default thresholds live in
`predictor/management/commands/train_model.py`
(`Normal 0.6, Good 0.25, Risky 0.25, Bad 0.25`) and can be re-tuned via
`tune_thresholds`.

### Backward compatibility

Loading a v2 model still works — `EnsembleModel.load()` detects the
absence of the `uses_posterior` key and falls back to one-hot IK
encoding. Re-train via `python manage.py train_model` to get a v3 model.

---

## 4. Training pipeline

```bash
# regenerate the ensemble (writes predictor/models/ensemble_model.joblib)
python manage.py train_model

# variants
python manage.py train_model --resampling oversample   # SMOTE
python manage.py train_model --resampling undersample  # RandomUnderSampler
```

Source data: `predictor/data/fishing_data.csv` (24 records of expert-labelled
observations). IK columns are mapped to model inputs via fixed dictionaries
in `train_model.handle`. Temporal split — first 80 % train, last 20 % test.

### Output you should see

```
Loaded N records
Class Distribution: …
Class weights: …
Training Ensemble Model...
IK / label agreement rate: 63.2%
Evaluating model...
Standard Evaluation: …
Threshold-Adjusted Evaluation: …
Sample prediction: Normal (True: Risky)
Prediction probabilities: {…}
```

### IK/label agreement diagnostic

`train_model` prints how often the fuzzy verdict matches the ground-truth
label across the training set. **If agreement exceeds 95 %** a warning is
emitted because the ensemble has no disagreement examples to learn from
and will collapse onto IK. The fix is to collect or relabel cases where
the fuzzy verdict was demonstrably wrong.

---

## 5. Validation and performance

### Reported metrics

`train_model` prints, on the temporal hold-out:

* `sklearn.metrics.classification_report` — precision / recall / F1 per class.
* `balanced_accuracy_score` — mean of per-class recall (robust to class imbalance).
* `f1_score(average='macro')` — macro-F1 (treats every class equally).

Both the **default** (argmax) and **threshold-adjusted** decisions are
evaluated.

### Inspecting model behaviour

```bash
python manage.py feature_importance
```

Prints:

* model metadata (`schema_version`, `trained_at`, `uses_posterior`, classes),
* aggregate **|coefficient| share** between the IK and Sci sides (a fast read
  on which side dominates the linear fusion),
* per-feature coefficients sorted by absolute magnitude.

A healthy v3 model should show both sides contributing — not 100 % IK and not
100 % Sci.

### Threshold tuning

```bash
python manage.py tune_thresholds
```

Sweeps per-class thresholds and persists the best macro-F1 configuration onto
the saved model (`set_class_thresholds`). Re-run after `train_model`.

### Online logging

Every forecast served by the views is logged (`predictor.models`) with
inputs, calibrated probabilities, and the model version. Use this to
collect real-world disagreement data and re-train.

### Known limitations

* **Small dataset** (24 rows at the time of writing). Metrics on the 5-row
  holdout are noisy — treat them as directional, not absolute.
* **Cross-fit calibration** falls back to `prefit` whenever a class has
  fewer than 2 samples in training. This is logged at WARNING level.
* The integrated risk function is **hand-tuned**, not learned. Re-tune the
  denominators (`precip/15`, `wind/10`) and the mixing weights if lake
  conditions or instrumentation change.

---

## 6. Running the application

```bash
# install
python -m venv .venv
.venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# database
python manage.py migrate

# (re)train the ensemble
python manage.py train_model

# run
python manage.py runserver 0.0.0.0:8000
```

Then visit:

* `http://127.0.0.1:8000/`            — landing
* `http://127.0.0.1:8000/indigenous/` — fuzzy-only
* `http://127.0.0.1:8000/integrated/` — fuzzy + live weather
* `http://127.0.0.1:8000/ensemble/`   — calibrated ML

---

## 7. Repository layout

```
predictor/
├── fuzzy_logic.py                       # 87-rule Mamdani engine, fuzzy_posterior
├── scientific.py                        # Open-Meteo client, ScientificModel,
│                                        # EnsembleModel, integrate_predictions
├── forms.py                             # Bootstrap-styled Django forms
├── views.py                             # indigenous / integrated / ensemble views
├── templates/                           # Bootstrap + Material UI
├── data/fishing_data.csv                # training data
├── models/ensemble_model.joblib         # trained ensemble (v3)
└── management/commands/
    ├── train_model.py                   # train + evaluate
    ├── tune_thresholds.py               # per-class threshold sweep
    └── feature_importance.py            # inspect LR coefficients
```
