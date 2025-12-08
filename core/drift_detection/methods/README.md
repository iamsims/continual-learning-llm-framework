# Drift Detection Methods

This directory contains implementations of different drift detection methods for text data. Each method uses a different approach to detect distributional shifts between reference and current data.

## Available Methods

### 1. Domain Classifier (`domain_classifier.py`)

**What it does:**
Trains a binary classifier (TF-IDF + SGDClassifier) to distinguish between reference and current text data.

**How it detects drift:**
- Converts texts to TF-IDF features (bag-of-words representation)
- Trains a classifier to predict whether a sample is from reference or current data
- If the classifier can easily distinguish between the two datasets, drift is detected
- Uses ROC-AUC score as the drift metric:
  - ROC-AUC = 0.5: Classifier performs at random level (no drift)
  - ROC-AUC > threshold: Classifier can distinguish datasets (drift detected)
  - Higher ROC-AUC = easier to distinguish = more drift

**Threshold Calculation (from Evidently source code):**
The threshold is **dynamically calculated** using a random classifier baseline:

1. **Random Classifier Simulation**:
   - Runs 1,000 iterations with random predictions
   - Each iteration calculates ROC-AUC for random labels
   - This simulates what a "no drift" scenario looks like

2. **95th Percentile Threshold**:
   - Takes the 95th percentile of random ROC-AUC scores
   - Typically results in threshold ≈ **0.52-0.55**
   - Hardcoded parameters: `p_value=0.05`, `iter_num=1000`, `seed=42`

3. **Drift Detection**:
   - Drift detected when: `domain_classifier_roc_auc > random_classifier_95_percentile`
   - If classifier performs better than 95% of random classifiers, drift is detected

**Classifier Configuration (hardcoded in Evidently):**
```python
TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")
SGDClassifier(alpha=0.0001, max_iter=50, penalty="l1", loss="modified_huber")
```

**Best for:**
- Detecting vocabulary changes and word usage shifts
- Identifying topic changes in text content
- Understanding which words are characteristic of each dataset
- Provides interpretable results (characteristic words and examples)

**Note:** Unlike embedding drift, the domain classifier threshold **cannot be customized** - it always uses the adaptive 95th percentile approach.

---

### 2. Embedding Drift (`embedding.py`)

**What it does:**
Compares semantic meaning of texts using pre-trained sentence embedding models.

**How it detects drift:**
- Converts texts to dense vector representations using Sentence Transformers
- Each text becomes a point in high-dimensional semantic space
- Compares distributions of embeddings using statistical tests
- Available methods:
  - **MMD (Maximum Mean Discrepancy)**: Measures distance between embedding distributions
  - **Distance-based**: Compares average embeddings using cosine/euclidean distance
  - **Ratio-based**: Checks proportion of drifted dimensions

**Best for:**
- Detecting semantic shifts in text meaning
- Catching changes that preserve vocabulary but alter meaning (e.g., "great" vs "excellent")
- More robust to paraphrasing and synonyms compared to TF-IDF methods

**Threshold modes:**
- **With Bootstrap** (default, `use_bootstrap=True`):
  - Calculates **adaptive threshold** from reference data using bootstrap sampling
  - Splits reference data randomly multiple times, computes drift metric between splits
  - Uses 95th percentile of these values as threshold (configurable via `quantile_probability=0.05`)
  - Adapts to natural variability in your reference data
  - More statistically sound - reduces false positives
  - **Recommended approach** - accounts for your data's inherent variability

- **Without Bootstrap** (`use_bootstrap=False`):
  - Uses **hardcoded static threshold** from Evidently:
    - MMD: threshold = 0.015
    - Distance-based: threshold = 0.2
  - Static thresholds chosen by Evidently developers
  - May be too sensitive or too lenient depending on your data
  - Not recommended unless you have specific requirements

**Configuration:**
- **Default model:** `all-MiniLM-L6-v2` (fast, 384-dimensional embeddings)
- **Default drift method:** `mmd` (Maximum Mean Discrepancy)
- **Default quantile:** `quantile_probability=0.05` (95th percentile)
- All configurable via `DriftDetectionConfig`

---

## How Methods Work Together

The drift detection system uses both methods in combination:
- **Domain Classifier** catches lexical/vocabulary drift
- **Embedding Drift** catches semantic/meaning drift

Drift severity is aggregated based on how many methods detect drift:
- **No drift**: Neither method detects drift
- **Low drift**: Only one method detects drift
- **High drift**: Both methods detect drift



<!-- ADD todo to add text overview also in the enum of methods -->

TODO 
- [] Add text overview to the main drift detection method pipeline