# Create a comprehensive Titanic dataset simulation with realistic data
# This matches the real Titanic dataset structure and patterns
np.random.seed(42)  # For reproducibility

n_passengers = 891  # Same as original Titanic dataset

# Create realistic passenger data
data = {
    'PassengerId': range(1, n_passengers + 1),
    'Survived': np.random.choice([0, 1], n_passengers, p=[0.62, 0.38]),  # ~38% survival rate
    'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.24, 0.21, 0.55]),  # Class distribution
    'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),  # Gender distribution
    'Age': np.random.normal(30, 14, n_passengers),  # Age with realistic distribution
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5, 8], n_passengers, p=[0.68, 0.23, 0.05, 0.02, 0.015, 0.003, 0.002]),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_passengers, p=[0.76, 0.13, 0.08, 0.015, 0.004, 0.001, 0.001]),
    'Fare': np.random.exponential(15, n_passengers),  # Fare with exponential distribution
    'Embarked': np.random.choice(['S', 'C', 'Q'], n_passengers, p=[0.72, 0.19, 0.09]),
    'Cabin': np.random.choice([None, 'A1', 'B2', 'C3', 'D4', 'E5'], n_passengers, p=[0.77, 0.046, 0.046, 0.046, 0.046, 0.046])
}

# Apply some realistic patterns from the actual Titanic data
for i in range(n_passengers):
    # Higher survival rates for females and first class
    if data['Sex'][i] == 'female':
        data['Survived'][i] = np.random.choice([0, 1], p=[0.26, 0.74])  # 74% survival for women
    if data['Pclass'][i] == 1:
        data['Survived'][i] = np.random.choice([0, 1], p=[0.37, 0.63])  # Higher survival in 1st class
    elif data['Pclass'][i] == 3:
        data['Survived'][i] = np.random.choice([0, 1], p=[0.76, 0.24])  # Lower survival in 3rd class
    
    # Adjust fare based on class
    if data['Pclass'][i] == 1:
        data['Fare'][i] = np.random.normal(84, 78)  # First class higher fares
    elif data['Pclass'][i] == 2:
        data['Fare'][i] = np.random.normal(20, 13)  # Second class moderate fares
    else:
        data['Fare'][i] = np.random.normal(13, 11)  # Third class lower fares
    
    # Ensure positive values
    data['Age'][i] = max(0.42, data['Age'][i])  # Min age 0.42 (as in original)
    data['Fare'][i] = max(0, data['Fare'][i])

# Create DataFrame
titanic_df = pd.DataFrame(data)

# Introduce some missing values to match real dataset patterns
missing_indices_age = np.random.choice(titanic_df.index, size=int(0.2 * len(titanic_df)), replace=False)
missing_indices_embarked = np.random.choice(titanic_df.index, size=2, replace=False)
missing_indices_cabin = np.random.choice(titanic_df.index, size=int(0.77 * len(titanic_df)), replace=False)

titanic_df.loc[missing_indices_age, 'Age'] = np.nan
titanic_df.loc[missing_indices_embarked, 'Embarked'] = np.nan
titanic_df.loc[missing_indices_cabin, 'Cabin'] = np.nan

print("Dataset created successfully!")
print(f"Dataset shape: {titanic_df.shape}")
print(f"Columns: {list(titanic_df.columns)}")
print("\nFirst 5 rows:")
print(titanic_df.head())