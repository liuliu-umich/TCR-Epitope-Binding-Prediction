# TCR Binding Prediction

This repository presents a computational biology and machine learning project focused on predicting whether a T-cell receptor (TCR) is likely to bind a target antigen epitope from sequence-derived features. The project was built as an end-to-end workflow spanning bioinformatics data preparation, sequence representation engineering, unsupervised feature discovery, and supervised model benchmarking. The main goal was to turn raw TCR-epitope pairing data into a practical modeling pipeline that can help prioritize plausible immune recognition events from sequence information alone.

## Why This Problem Matters

T-cell receptors recognize antigen peptides presented by major histocompatibility complexes, and predicting these interactions is an important challenge in immunology, therapeutic design, and biomarker discovery. Reliable sequence-based prioritization can support downstream experimental work by narrowing large candidate spaces to the most promising TCR-epitope matches.

## What I Built

This project covers the full modeling workflow:

- Cleaned and filtered TCR-epitope datasets derived from VDJdb and related external data sources.
- Constructed positive and negative binding examples for supervised learning.
- Engineered sequence features using amino acid composition descriptors, one-hot sequence representations, and ESM-based protein language model embeddings.
- Built unsupervised features with PCA, UMAP, and KMeans to compress and structure high-dimensional sequence representations.
- Benchmarked supervised models including Logistic Regression, Random Forest, and XGBoost for TCR-epitope binding prediction.

## Data Sources

- `VDJdb` is the primary curated data source used for known TCR sequences with antigen specificity annotations.
- `TRAIT`-style data is used in the repository as an external validation-style dataset and as part of the broader data preparation workflow.
- Local processed artifacts are stored in [`Datasets/`](/Users/newuser/TCR_Binding_Prediction/Datasets).

## Method Overview

The repository reflects a staged pipeline rather than a single script:

1. Data cleaning and pair construction
   Raw sequence files are cleaned, filtered to relevant records, deduplicated, and transformed into positive/negative TCR-epitope pairing datasets.
2. Sequence feature engineering
   The project generates multiple representations of TCR and epitope sequences, including amino acid composition features, one-hot encodings, and ESM embeddings.
3. Unsupervised representation analysis
   High-dimensional sequence features are explored and compressed with PCA and UMAP, then clustered with KMeans to derive additional structure-aware features.
4. Supervised learning and evaluation
   The engineered and derived features are used to train and compare classical ML models for binary binding prediction.

## Results Highlights

- The strongest visible result in [`SupervisedLearningV4.ipynb`](/Users/newuser/TCR_Binding_Prediction/SupervisedLearningV4.ipynb) reports `ROC-AUC 0.9198`, `F1 0.8541`, and `Accuracy 0.8399`.
- The project compares multiple baselines rather than relying on a single model family, including Logistic Regression, Random Forest, and XGBoost.
- The supervised notebooks indicate that enriched sequence representations and derived unsupervised features improved the quality of downstream classification experiments.

## Repository Guide

- [`data_prep.py`](/Users/newuser/TCR_Binding_Prediction/data_prep.py) and [`tl_datacleaning.py`](/Users/newuser/TCR_Binding_Prediction/tl_datacleaning.py)
  Data cleaning, filtering, and positive/negative dataset construction.
- [`pybiomed_features.py`](/Users/newuser/TCR_Binding_Prediction/pybiomed_features.py) and [`protein_seq_representation_extraction.ipynb`](/Users/newuser/TCR_Binding_Prediction/protein_seq_representation_extraction.ipynb)
  Sequence featurization using composition-based features, one-hot representations, and ESM embeddings.
- [`unsupervised learning.ipynb`](/Users/newuser/TCR_Binding_Prediction/unsupervised%20learning.ipynb) and [`utils/unsupervised.py`](/Users/newuser/TCR_Binding_Prediction/utils/unsupervised.py)
  Dimensionality reduction, clustering, and derived unsupervised feature generation.
- [`SupervisedLearningV3.ipynb`](/Users/newuser/TCR_Binding_Prediction/SupervisedLearningV3.ipynb), [`SupervisedLearningV4.ipynb`](/Users/newuser/TCR_Binding_Prediction/SupervisedLearningV4.ipynb), and [`supervised_learning.py`](/Users/newuser/TCR_Binding_Prediction/supervised_learning.py)
  Model training, benchmark comparison, and evaluation workflows.
- [`Datasets/`](/Users/newuser/TCR_Binding_Prediction/Datasets)
  Source and processed dataset artifacts used throughout the notebooks and scripts.

## Technical Skills Demonstrated

- Bioinformatics data cleaning and sequence-focused dataset preparation
- Positive/negative sample construction for biological classification tasks
- Protein and TCR sequence feature engineering
- Dimensionality reduction and clustering for high-dimensional biological data
- Classical machine learning benchmarking and metric-based evaluation
- Python-based notebook workflow for exploratory modeling and analysis

## How To Explore This Repo

1. Start with this README for the high-level project framing.
2. Open [`protein_seq_representation_extraction.ipynb`](/Users/newuser/TCR_Binding_Prediction/protein_seq_representation_extraction.ipynb) to review how sequence embeddings and representations were generated.
3. Review [`unsupervised learning.ipynb`](/Users/newuser/TCR_Binding_Prediction/unsupervised%20learning.ipynb) to see PCA, UMAP, and clustering experiments.
4. Inspect [`SupervisedLearningV4.ipynb`](/Users/newuser/TCR_Binding_Prediction/SupervisedLearningV4.ipynb) for the clearest final model comparison and headline metrics.

## Limitations and Future Improvements

- The project is currently notebook-heavy, which makes the workflow more exploratory than productionized.
- Reproducibility can be improved with a dedicated dependency file and a more standardized environment setup.
- The training and evaluation path could be consolidated into a cleaner package or pipeline structure.
- Future work could include deeper biological validation, stronger external testing, and a more formalized experiment tracking setup.

## Reference

VDJdb: Shugay M, Bagaev DV, Zvyagin IV, Vroomans RM, Crawford JC, Dolton G, Komech EA, Sycheva AL, Koneva AE, Egorov ES, Eliseev AV, Van Dyk E, Dash P, Attaf M, Rius C, Ladell K, McLaren JE, Matthews KK, Clemens EB, Douek DC, Luciani F, van Baarle D, Kedzierska K, Kesmir C, Thomas PG, Price DA, Sewell AK, Chudakov DM. "VDJdb: a curated database of T-cell receptor sequences with known antigen specificity." *Nucleic Acids Research* 46(D1), 2018.
