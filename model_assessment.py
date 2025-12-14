"""
Part 2: Uses Best Predictor Model to Examine Era-Specific Performance
- Trains Random Forest models within each defined era
- Evaluates cross-era transfer performance using R² heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('tracks_cleaned.csv')

# Define audio features ONLY (no temporal features!)
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]



### Within-Era Model Training and Evaluation ###
eras = ['Pre-1950', '1950s-1960s', '1970s-1980s', '1990s-2000s', '2010s-2020s']
era_models = {}
within_era_performance = {}

for era in eras:
    print(f"\nTraining model for {era}:") # Keep track of era training
    
    # Filter to this era
    era_data = df[df['era'] == era].dropna(subset=audio_features + ['popularity'])
    
    x = era_data[audio_features] # features
    y = era_data['popularity'] # target variable
    
    # Split within this era (80/20)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    # Train model (RF)
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    
    # Evaluate
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Store results
    era_models[era] = model
    within_era_performance[era] = {
        'R²': r2,
        'RMSE': rmse,
        'n_samples': len(era_data),
        'n_train': len(x_train),
        'n_test': len(x_test)
    }
    
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}")
    print(f"  Samples: {len(era_data):,} ({len(x_train):,} train, {len(x_test):,} test)")

# Summary of within-era performance
performance_df = pd.DataFrame(within_era_performance).T
print(performance_df) # quick look

# Save results
performance_df.to_csv('within_era_performance.csv')



### Cross-Era Performance Eval + Heatmap ###
transfer_matrix = pd.DataFrame(index=eras, columns=eras, dtype=float) # empty heatmap

for era_train in eras:
    if era_train not in era_models:
        continue
    
    model = era_models[era_train]
    
    for era_test in eras:
        # Get test data from this era
        test_data = df[df['era'] == era_test].dropna(subset=audio_features + ['popularity'])
        
        x_test = test_data[audio_features]
        y_test = test_data['popularity']
        
        # Predict using the train era's model
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)

        # Quantify using R² and store in matrix
        transfer_matrix.loc[era_train, era_test] = r2
        
print(transfer_matrix) # Quick look

# Heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(
    transfer_matrix.astype(float), 
    annot=True, 
    fmt='.3f', 
    cmap='RdYlGn',
    vmin=0, 
    vmax=0.6,
    center=0.3,
    square=True,
    linewidths=1,
    cbar_kws={'label': 'R² Score'}
)
plt.title('Cross-Era Model Transfer Performance\n(Diagonal = Within-Era, Off-Diagonal = Cross-Era)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Test Era', fontsize=12)
plt.ylabel('Train Era', fontsize=12)
plt.tight_layout()
plt.savefig('cross_era_transfer_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Save matrix
transfer_matrix.to_csv('cross_era_performance.csv')

# Calculate transfer penalties
for era in eras:
    if era not in era_models:
        continue
    
    within_r2 = transfer_matrix.loc[era, era]
    
    # Average performance on OTHER eras
    other_eras = [e for e in eras if e != era and pd.notna(transfer_matrix.loc[era, e])]
    if not other_eras:
        continue
    
    cross_r2_avg = transfer_matrix.loc[era, other_eras].mean()
    penalty = within_r2 - cross_r2_avg
    penalty_pct = (penalty / within_r2) * 100 if within_r2 > 0 else 0
    
    print(f"\n{era}:")
    print(f"  Within-era R²: {within_r2:.4f}")
    print(f"  Cross-era R² (avg): {cross_r2_avg:.4f}")
    print(f"  Transfer penalty: {penalty:.4f} ({penalty_pct:.1f}% drop)")


### Feature Importance + Heatmap ###
importance_by_era = {}

for era, model in era_models.items():
    importance_by_era[era] = dict(zip(audio_features, model.feature_importances_))

# Convert to DataFrame
importance_df = pd.DataFrame(importance_by_era).T
importance_df = importance_df[audio_features]  # Reorder columns

print("\nFeature Importance by Era:")
print(importance_df.round(4)) # Quick look

# Visualize importance evolution
fig, axes = plt.subplots(1, 1, figsize=(14, 10))

# Heatmap
sns.heatmap(
    importance_df, 
    annot=True, 
    fmt='.3f', 
    cmap='YlOrRd',
    ax=axes,
    cbar_kws={'label': 'Feature Importance'}
)
axes.set_title('Feature Importance by Era', fontsize=14, fontweight='bold')
axes.set_xlabel('Audio Features', fontsize=12)
axes.set_ylabel('Era', fontsize=12)

plt.tight_layout()

# Save figure and csv data
plt.savefig('feature_importance_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
importance_df.to_csv('feature_importance_by_era.csv')