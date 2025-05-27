import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#IMPORT THE CSV FILE

from google.colab import files
upload = files.upload() 

# Load dataset
df = pd.read_csv('mobile phone price prediction.csv')

# Clean 'Price' column (remove commas, convert to numeric)
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'Name'])

# Drop rows with missing target (Price is already cleaned above)
df = df.dropna(subset=['Price'])

# Fill missing values (can be improved with more domain-specific logic)
df = df.fillna('Unknown')

# Feature/target split
X = df.drop(columns=['Price'])
y = df['Price']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Print the starting rows
df.head()

# Print the rows from the last
df.tail()

# Print the no. of rows and column
df.shape

# Print the statistical values
df.describe()

# prompt: use group by command

# Example: Group by 'Brand' and calculate the average price
average_price_by_brand = df.groupby('company')['Price'].mean()

average_price_by_brand

# You can also group by multiple columns
average_price_by_brand_and_processor = df.groupby(['company', 'Processor'])['Price'].mean()
print(average_price_by_brand_and_processor)

# You can apply multiple aggregation functions
brand_summary = df.groupby('company')['Price'].agg(['mean', 'min', 'max', 'count'])
print(brand_summary)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# prompt: generate histogram and correlation matrix (for all the fields)

import matplotlib.pyplot as plt
import seaborn as sns

# Generate histograms for numerical features
df[numerical_features].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Generate correlation matrix for all numerical fields
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# prompt: plot a graph between price and rating

import matplotlib.pyplot as plt
# Check if 'Rating' column exists before plotting
if 'Rating' in df.columns:
  # Scatter plot of Price vs Rating
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x='Rating', y='Price', data=df)
  plt.title('Price vs. Rating')
  plt.xlabel('Rating')
  plt.ylabel('Price')
  plt.show()
else:
  print("The 'Rating' column does not exist in the DataFrame.")

# prompt: make a histogram of company vs rating 

import matplotlib.pyplot as plt
# Generate a histogram of 'company' vs 'Rating' if 'Rating' exists
if 'Rating' in df.columns:
    # Assuming 'Rating' is a numerical column, we can plot its distribution grouped by 'company'
    # This might require some aggregation depending on how you want to represent it.
    # For example, plotting the distribution of ratings for each company:
    plt.figure(figsize=(15, 8))
    sns.histplot(data=df, x='Rating', hue='company', multiple='stack', bins=10)
    plt.title('Distribution of Ratings by Company')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Alternatively, if you want to see the average rating per company:
    average_rating_by_company = df.groupby('company')['Rating'].mean().sort_values(ascending=False)
    plt.figure(figsize=(15, 8))
    average_rating_by_company.plot(kind='bar')
    plt.title('Average Rating by Company')
    plt.xlabel('Company')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

else:
  print("The 'Rating' column does not exist in the DataFrame.")

y = np.log1p(df['Price'])

from sklearn.ensemble import RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R² Score: {r2:.2f}')