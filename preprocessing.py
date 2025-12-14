"""
Spotify Bias Analysis - Part 1: Data Loading and EDA
Project: Biases in Spotify Suggestion
Authors: Avery Mattoon, Leila Buchan
"""

import pandas as pd

df_clean = pd.read_csv('tracks.csv')

# Identify audio features
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence',
                    'tempo']

# Handle missing values in audio features by replacing with median
for feature in audio_features:
    if df_clean[feature].isnull().sum() > 0:
        df_clean[feature].fillna(df_clean[feature].median(), inplace=True)

# Extract year from release_date
df_clean['year'] = (
    df_clean['release_date']
    .astype(str)
    .str.extract(r'(19\d{2}|20\d{2})', expand=False)
    .astype(float)
)

# Filter to valid years (1921-2020)
df_clean = df_clean[(df_clean['year'] >= 1921) & (df_clean['year'] <= 2020)]
print(f"After filtering to years 1921-2020: {len(df_clean):,} tracks remaining")

# Era feature
def assign_era(year):
    if year < 1950:
        return 'Pre-1950'
    elif year < 1970:
        return '1950s-1960s'
    elif year < 1990:
        return '1970s-1980s'
    elif year < 2010:
        return '1990s-2000s'
    else:
        return '2010s-2020s'

df_clean['era'] = df_clean['year'].apply(assign_era)

# Drop unneeded columns
df_clean.drop(columns=['id', 'name', 'duration_ms', 'explicit', 'artists', 'id_artists', 'release_date', 'time_signature', 'key', 'mode'], inplace=True)

# Save cleaned data
df_clean.to_csv('tracks_cleaned.csv', index=False)