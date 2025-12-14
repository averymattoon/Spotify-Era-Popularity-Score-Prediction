# Spotify-Era-Popularity-Score-Prediction

# Spotify Bias Analysis Project

**Authors:** Avery Mattoon, Leila Buchan  
**Project:** Biases in Spotify Suggestion

## Overview

This project analyzes temporal biases in Spotify's music popularity prediction using machine learning. It examines whether models trained on audio features from certain eras systematically favor or disadvantage music from different eras (1921-2020).

## Project Structure

```
Spotify-Era-Popularity-Score-Prediction/
│
├── preprocesaing.py             # Data loading, cleaning, and feature selection
├── model_selection.py           # Training and evaluating models
├── model_assessment.py          # Using best model to explore bias
└── README.md                    # This file
```

## Dataset

**Source:** [Spotify Dataset 1921-2020, 600k+ Tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks/data)

**Required File:** `tracks.csv` (download from Kaggle and place in project directory)

## Usage

### Run Complete Pipeline In Order

```bash
python preprocessing.py
python model_selection.py
python model_assessment.py
```

## Expected Outputs

### preprocessing.py
- `tracks_cleaned.csv` - Cleaned dataset
- 
### model_selection.py
- `model_comparison_detailed.csv` - Model performance comparison
- `model_comparison.png` - Model performance visualization

### model_assessment.py
- 'feature_importance_by_era.csv' - Feature importance comparison
- 'feature_importance_evolution.png' - Feature importance heatmap

## Key Features

### Machine Learning Models
- **Baseline:** Linear Regression, Ridge, Lasso
- **Tree-based:** Decision Trees, Random Forests, Gradient Boosting
- **Other:** kNN

### Bias Analysis
- Temporal bias by era (Pre-1950, 1950s-1960s, 1970s-1980s, 1990s-2000s, 2010s-2020s)
- Cross-era model performance

## Key Research Questions

1. Can audio features effectively predict song popularity across different time periods?
2. Do machine learning models exhibit systematic biases favoring certain musical eras?
3. How has the relationship between audio features and popularity evolved over time?
4. Which audio features are most predictive of popularity?

## Citations

Key references used in this project:

1. Ferraro, A., Serra, X., & Bauer, C. (2021). Break the Loop: Gender Imbalance in Music Recommenders. *CHIIR '21*.

2. Ferraro, A., Serra, X., & Bauer, C. (2021). What is fair? Exploring the artists' perspective on the fairness of music streaming platforms. *INTERACT 2021*.

3. Ungruh, J., et al. (2024). Putting popularity bias mitigation to the test: A user-centric evaluation in music recommenders. *RecSys '24*.

## License

This project is for educational purposes as part of a machine learning course._
