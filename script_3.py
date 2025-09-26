# PART 1: DATA UNDERSTANDING & PREPROCESSING

print("="*60)
print("PART 1: DATA UNDERSTANDING & PREPROCESSING")
print("="*60)

# 1.1 Basic data exploration
print("\n1. BASIC DATA EXPLORATION")
print("-" * 40)

print(f"Dataset shape: {titanic_df.shape}")
print(f"Memory usage: {titanic_df.memory_usage().sum()} bytes")

print("\nData types:")
print(titanic_df.dtypes)

print("\nBasic statistics for numerical columns:")
print(titanic_df.describe())

print("\nValue counts for categorical columns:")
for col in ['Sex', 'Pclass', 'Embarked']:
    print(f"\n{col}:")
    print(titanic_df[col].value_counts())

# 1.2 Missing values analysis
print("\n\n2. MISSING VALUES ANALYSIS")
print("-" * 40)

missing_data = titanic_df.isnull().sum()
missing_percentage = (missing_data / len(titanic_df)) * 100

missing_info = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percentage': missing_percentage
})

print("Missing values summary:")
print(missing_info[missing_info['Missing_Count'] > 0])

# Visualize missing data patterns
print("\nMissing data pattern:")
for col in titanic_df.columns:
    if titanic_df[col].isnull().sum() > 0:
        print(f"{col}: {'█' * int(missing_percentage[col]/2)}░{'█' * int((100-missing_percentage[col])/2)} ({missing_percentage[col]:.1f}% missing)")

# 1.3 Handle missing values
print("\n\n3. HANDLING MISSING VALUES")
print("-" * 40)

# Create a copy for preprocessing
df_processed = titanic_df.copy()

# Handle Age: Fill with median grouped by Sex and Pclass
print("Handling Age missing values...")
age_median_by_group = df_processed.groupby(['Sex', 'Pclass'])['Age'].median()
print("Median age by Sex and Pclass:")
print(age_median_by_group)

def fill_age(row):
    if pd.isna(row['Age']):
        return age_median_by_group[row['Sex'], row['Pclass']]
    return row['Age']

df_processed['Age'] = df_processed.apply(fill_age, axis=1)

# Handle Embarked: Fill with mode (most common port)
mode_embarked = df_processed['Embarked'].mode()[0]
df_processed['Embarked'].fillna(mode_embarked, inplace=True)
print(f"Filled Embarked missing values with mode: {mode_embarked}")

# Handle Cabin: Create binary feature for cabin availability
df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)
df_processed.drop('Cabin', axis=1, inplace=True)
print("Created Has_Cabin binary feature and dropped Cabin column")

# Verify missing values are handled
print(f"\nMissing values after preprocessing:")
print(df_processed.isnull().sum())

print("\nDataset after handling missing values:")
print(f"Shape: {df_processed.shape}")
print(df_processed.head())