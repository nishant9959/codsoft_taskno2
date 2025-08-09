import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 1. Load the Dataset ---
# Load the dataset from the uploaded CSV file.
try:
    df = pd.read_csv('IMDb Movies India.csv', encoding='ISO-8859-1')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'IMDb Movies India.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# --- 2. Data Cleaning and Preprocessing ---
print("\n--- Data Cleaning and Preprocessing ---")

# Display initial info about the dataset
print("Initial Dataset Info:")
df.info()
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop rows where the target variable 'Rating' is missing, as they are not useful for training or evaluation.
df.dropna(subset=['Rating'], inplace=True)

# Handle missing values for other key columns
# For categorical columns, we'll fill missing values with 'Unknown'.
# For numerical columns, we'll impute them later in the ML pipeline.
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col].fillna('Unknown', inplace=True)

# The 'Year' column has non-numeric values and is in object format.
# We'll extract the numeric year and convert the column to an integer.
df['Year'] = df['Year'].astype(str).str.extract('(\d{4})').astype(float)

# The 'Duration' column is an object with ' min' suffix.
# We'll remove the suffix and convert it to a numeric type.
df['Duration'] = df['Duration'].astype(str).str.replace(' min', '').astype(float)

# The 'Votes' column is an object with commas.
# We'll remove the commas and convert it to a numeric type.
df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(float)

# Drop rows where the cleaned numerical columns might still have NaNs
df.dropna(subset=['Year', 'Duration', 'Votes'], inplace=True)

# Convert columns to appropriate integer types
df['Year'] = df['Year'].astype(int)
df['Duration'] = df['Duration'].astype(int)
df['Votes'] = df['Votes'].astype(int)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nCleaned Dataset Info:")
df.info()
print("\nSample of cleaned data:")
print(df.head())


# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")

# a. Year with the best average rating
avg_rating_per_year = df.groupby('Year')['Rating'].mean().reset_index()
best_year = avg_rating_per_year.loc[avg_rating_per_year['Rating'].idxmax()]
print(f"\nYear with the best average rating: {best_year['Year']} with an average rating of {best_year['Rating']:.2f}")

plt.figure(figsize=(15, 6))
sns.lineplot(data=avg_rating_per_year, x='Year', y='Rating')
plt.title('Average Movie Rating Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()


# b. Does the length of the movie have any impact on the rating?
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Duration', y='Rating', alpha=0.5)
plt.title('Movie Duration vs. Rating')
plt.xlabel('Duration (minutes)')
plt.ylabel('Rating')
plt.grid(True)
plt.show()
print("\nAnalysis of Duration vs. Rating: The scatter plot shows the relationship. Generally, there isn't a strong linear correlation, but very short or very long movies might have different rating patterns.")


# c. Top 10 movies overall (considering movies with a decent number of votes)
# We filter for movies with more than 1000 votes to get more reliable top ratings.
top_10_overall = df[df['Votes'] > 1000].sort_values(by='Rating', ascending=False).head(10)
print("\nTop 10 Movies Overall (with >1000 votes):")
print(top_10_overall[['Name', 'Year', 'Rating', 'Votes']])


# d. Number of popular movies released each year (let's define popular as Rating > 8.0 and Votes > 1000)
popular_movies = df[(df['Rating'] > 8.0) & (df['Votes'] > 1000)]
popular_movies_per_year = popular_movies['Year'].value_counts().sort_index()
print("\nNumber of popular movies (Rating > 8.0, Votes > 1000) released each year:")
print(popular_movies_per_year)

plt.figure(figsize=(15, 6))
popular_movies_per_year.plot(kind='bar')
plt.title('Number of Popular Movies Released Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Popular Movies')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()


# e. Top Director, Actor
top_director = df['Director'].value_counts().drop('Unknown').head(1)
print(f"\nDirector who directed the most movies: {top_director.index[0]} with {top_director.values[0]} movies.")

top_actor1 = df['Actor 1'].value_counts().drop('Unknown').head(1)
print(f"Actor who starred in the most movies (as Actor 1): {top_actor1.index[0]} with {top_actor1.values[0]} movies.")


# --- 4. Feature Engineering and Model Building ---
print("\n--- Model Building: Predicting Movie Ratings ---")

# For simplicity, we'll use a subset of features.
# Using all actors/directors would create too many features (curse of dimensionality).
# We'll use the top N most frequent categories for 'Genre', 'Director', and 'Actor 1'.
top_n = 20
top_genres = df['Genre'].value_counts().nlargest(top_n).index
top_directors = df['Director'].value_counts().nlargest(top_n).index
top_actors = df['Actor 1'].value_counts().nlargest(top_n).index

# Replace less frequent categories with 'Other'
df['Genre'] = df['Genre'].apply(lambda x: x if x in top_genres else 'Other')
df['Director'] = df['Director'].apply(lambda x: x if x in top_directors else 'Other')
df['Actor 1'] = df['Actor 1'].apply(lambda x: x if x in top_actors else 'Other')

# Define features (X) and target (y)
features = ['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1']
X = df[features]
y = df['Rating']

# Identify categorical and numerical features
categorical_features = ['Genre', 'Director', 'Actor 1']
numerical_features = ['Year', 'Duration', 'Votes']

# Create a preprocessing pipeline
# This will handle imputation for numerical columns and one-hot encoding for categorical columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the full model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining model on {len(X_train)} samples and testing on {len(X_test)} samples.")

# Train the model
model_pipeline.fit(X_train, y_train)

# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"\nModel Explanation: The R² value of {r2:.2f} means that our model can explain approximately {r2*100:.0f}% of the variance in movie ratings based on the selected features (Year, Duration, Votes, Genre, Director, and Actor).")

# --- Example Prediction ---
# Let's create a sample movie to predict its rating
sample_movie = pd.DataFrame([{
    'Year': 2023,
    'Duration': 150,
    'Votes': 50000,
    'Genre': 'Action', # Use a genre that is in the top list or it will be handled as unknown
    'Director': 'David Dhawan', # Use a director in the top list
    'Actor 1': 'Amitabh Bachchan' # Use an actor in the top list
}])

predicted_rating = model_pipeline.predict(sample_movie)
print(f"\nPredicted rating for the sample movie: {predicted_rating[0]:.1f}")

