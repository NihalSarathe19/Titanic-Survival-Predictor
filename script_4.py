# Continue Part 1: Categorical Encoding and Feature Scaling

print("\n\n4. CATEGORICAL VARIABLE ENCODING")
print("-" * 40)

# Encode categorical variables
# Sex: Binary encoding (male=1, female=0)
df_processed['Sex_Male'] = (df_processed['Sex'] == 'male').astype(int)
print("Encoded Sex as Sex_Male (1=male, 0=female)")

# Embarked: One-hot encoding
embarked_dummies = pd.get_dummies(df_processed['Embarked'], prefix='Embarked')
df_processed = pd.concat([df_processed, embarked_dummies], axis=1)
print(f"One-hot encoded Embarked: {list(embarked_dummies.columns)}")

# Pclass: One-hot encoding (to avoid ordinality assumption)
pclass_dummies = pd.get_dummies(df_processed['Pclass'], prefix='Pclass')
df_processed = pd.concat([df_processed, pclass_dummies], axis=1)
print(f"One-hot encoded Pclass: {list(pclass_dummies.columns)}")

# Drop original categorical columns
df_processed.drop(['Sex', 'Embarked', 'Pclass'], axis=1, inplace=True)
print("Dropped original categorical columns")

print(f"\nDataset after encoding:")
print(f"Shape: {df_processed.shape}")
print(f"Columns: {list(df_processed.columns)}")

# 5. Feature Engineering
print("\n\n5. FEATURE ENGINEERING")
print("-" * 40)

# Create family size feature
df_processed['Family_Size'] = df_processed['SibSp'] + df_processed['Parch'] + 1
print("Created Family_Size feature (SibSp + Parch + 1)")

# Create age groups
df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                   bins=[0, 12, 18, 30, 50, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle_Age', 'Senior'])

# One-hot encode age groups
age_group_dummies = pd.get_dummies(df_processed['Age_Group'], prefix='Age')
df_processed = pd.concat([df_processed, age_group_dummies], axis=1)
df_processed.drop('Age_Group', axis=1, inplace=True)
print("Created and encoded Age_Group categories")

# Create fare groups
df_processed['Fare_Group'] = pd.cut(df_processed['Fare'], 
                                    bins=[0, 7.91, 14.45, 31, 1000], 
                                    labels=['Low', 'Medium', 'High', 'Very_High'])

# One-hot encode fare groups
fare_group_dummies = pd.get_dummies(df_processed['Fare_Group'], prefix='Fare')
df_processed = pd.concat([df_processed, fare_group_dummies], axis=1)
df_processed.drop('Fare_Group', axis=1, inplace=True)
print("Created and encoded Fare_Group categories")

# Create is_alone feature
df_processed['Is_Alone'] = (df_processed['Family_Size'] == 1).astype(int)
print("Created Is_Alone binary feature")

print(f"\nDataset after feature engineering:")
print(f"Shape: {df_processed.shape}")
print(f"New features added: Family_Size, Age groups, Fare groups, Is_Alone")

# 6. Feature Scaling
print("\n\n6. FEATURE SCALING/NORMALIZATION")
print("-" * 40)

# Identify numerical columns that need scaling
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Family_Size']
print(f"Columns to be scaled: {numerical_cols}")

# Initialize scaler
scaler = StandardScaler()

# Fit and transform numerical columns
df_scaled = df_processed.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

print("Applied StandardScaler to numerical features")
print(f"\nStatistics after scaling (should have mean~0, std~1):")
print(df_scaled[numerical_cols].describe().round(3))

# Prepare final dataset
# Drop PassengerId as it's not useful for prediction
df_final = df_scaled.drop('PassengerId', axis=1)

print(f"\n\nFINAL PREPROCESSED DATASET")
print("=" * 40)
print(f"Shape: {df_final.shape}")
print(f"Features: {len(df_final.columns)-1}")  # -1 for target variable
print(f"Target: Survived")
print(f"\nColumns in final dataset:")
print(list(df_final.columns))

print(f"\nFirst 3 rows of processed data:")
print(df_final.head(3))