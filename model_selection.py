"""
Part 1: Compare Popularity Prediction Performance Across Multiple Algorithms
- Compares regression models using R² and RMSE
- Stores more detailed results in CSV for further analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

### Load data and initialize variables ###
df = pd.read_csv('tracks_cleaned.csv')

# Isolate features
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Prepare data
x = df[audio_features].dropna() # features
y = df.loc[x.index, 'popularity'] # target variable

# Train-test split 80/20
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Standardize for models that need it
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


### Define Models ###
models = {
    # Linear Models (need scaling)
    'Linear Regression': {
        'model': LinearRegression(),
        'scaled': True
    },
    'Ridge Regression': {
        'model': Ridge(alpha=1.0, random_state=42),
        'scaled': True
    },
    'Lasso Regression': {
        'model': Lasso(alpha=0.1, random_state=42, max_iter=10000),
        'scaled': True
    },
    
    # Tree-based (no scaling needed)
    'Decision Tree': {
        'model': DecisionTreeRegressor(max_depth=15, min_samples_split=20, random_state=42),
        'scaled': False
    },
    'Random Forest': {
        'model': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, 
                                       random_state=42, n_jobs=-1),
        'scaled': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                           random_state=42),
        'scaled': False
    },
    
    # kNN (needs scaling)
    'K-Nearest Neighbors': {
        'model': KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
        'scaled': True
    }
}



### Train and evaluate each model ###
results = []
for name, config in models.items():
    print(f"\n{name}:") # Keep track of models
    
    model = config['model']
    needs_scaling = config['scaled']
    
    # Scale where needed
    if needs_scaling:
        x_tr = x_train_scaled
        x_te = x_test_scaled
    else:
        x_tr = x_train.values
        x_te = x_test.values

    # Train
    model.fit(x_tr, y_train)

    # Predictions
    y_train_pred = model.predict(x_tr)
    y_test_pred = model.predict(x_te)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation R² (5-fold)
    cv_scores = cross_val_score(
        model, x_tr, y_train, 
        cv=5, scoring='r2', n_jobs=-1
    )
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'CV R² (mean)': cv_r2_mean,
        'CV R² (std)': cv_r2_std,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Overfitting': train_r2 - test_r2  # Positive = overfitting
    })
    
    # Print brief summary for quick results
    print(f"  Test R² = {test_r2:.4f}, Test RMSE = {test_rmse:.4f}, Test MAE = {test_mae:.4f}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('model_comparison_detailed.csv', index=False)



### Model comparison graphs (R^2, RMSE) ###
fig, axes = plt.subplots(1, 2, figsize=(16, 12))

# Plot 1: R² Comparison
ax1 = axes[0]
x_pos = np.arange(len(results_df))
width = 0.35

bars1 = ax1.barh(x_pos - width/2, results_df['Train R²'], width, 
                 label='Train R²', alpha=0.8, color='skyblue')
bars2 = ax1.barh(x_pos + width/2, results_df['Test R²'], width, 
                 label='Test R²', alpha=0.8, color='coral')

ax1.set_yticks(x_pos)
ax1.set_yticklabels(results_df['Model'])
ax1.set_xlabel('R² Score', fontsize=11)
ax1.set_title('Model Performance: R² (Coefficient of Determination)', 
              fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Add value labels
for i, (train, test) in enumerate(zip(results_df['Train R²'], results_df['Test R²'])):
    ax1.text(train, i - width/2, f'{train:.3f}', 
            va='center', ha='left', fontsize=9)
    ax1.text(test, i + width/2, f'{test:.3f}', 
            va='center', ha='left', fontsize=9)

# Plot 2: RMSE Comparison
ax2 = axes[1]
bars3 = ax2.barh(x_pos - width/2, results_df['Train RMSE'], width, 
                 label='Train RMSE', alpha=0.8, color='lightgreen')
bars4 = ax2.barh(x_pos + width/2, results_df['Test RMSE'], width, 
                 label='Test RMSE', alpha=0.8, color='salmon')

ax2.set_yticks(x_pos)
ax2.set_yticklabels(results_df['Model'])
ax2.set_xlabel('RMSE (Root Mean Squared Error)', fontsize=11)
ax2.set_title('Model Performance: RMSE (Lower is Better)', 
              fontsize=13, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
# Save model comparison
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()