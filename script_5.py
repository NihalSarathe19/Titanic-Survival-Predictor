# PART 2: EXPLORATORY DATA ANALYSIS (EDA)

print("="*60)
print("PART 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# Basic insights from the dataset
print("\n1. BASIC INSIGHTS FROM THE DATASET")
print("-" * 40)

# Survival statistics
print(f"Overall survival rate: {df_final['Survived'].mean():.1%}")
print(f"Total passengers: {len(df_final)}")
print(f"Survivors: {df_final['Survived'].sum()}")
print(f"Non-survivors: {len(df_final) - df_final['Survived'].sum()}")

# Gender-based survival
print(f"\nGender-based survival:")
gender_survival = df_final.groupby('Sex_Male')['Survived'].agg(['count', 'sum', 'mean'])
gender_survival.index = ['Female', 'Male']
gender_survival.columns = ['Total', 'Survivors', 'Survival_Rate']
print(gender_survival)

# Class-based survival
print(f"\nClass-based survival:")
class_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3']
for i, col in enumerate(class_cols):
    class_passengers = df_final[df_final[col] == 1]
    if len(class_passengers) > 0:
        survival_rate = class_passengers['Survived'].mean()
        print(f"Class {i+1}: {survival_rate:.1%} survival rate ({class_passengers['Survived'].sum()}/{len(class_passengers)})")

# Age group analysis
print(f"\nAge group survival:")
age_cols = ['Age_Child', 'Age_Teen', 'Age_Adult', 'Age_Middle_Age', 'Age_Senior']
age_names = ['Children', 'Teenagers', 'Adults', 'Middle Age', 'Seniors']
for age_col, age_name in zip(age_cols, age_names):
    age_passengers = df_final[df_final[age_col] == 1]
    if len(age_passengers) > 0:
        survival_rate = age_passengers['Survived'].mean()
        print(f"{age_name}: {survival_rate:.1%} survival rate ({age_passengers['Survived'].sum()}/{len(age_passengers)})")

# Family size analysis
print(f"\nFamily size analysis:")
family_survival = df_final.groupby('Family_Size')['Survived'].agg(['count', 'mean'])
family_survival.columns = ['Count', 'Survival_Rate']
print(family_survival.head(8))

# Statistical summaries
print(f"\n2. STATISTICAL SUMMARIES")
print("-" * 40)

print("Key statistics for survivors vs non-survivors:")
survivors = df_final[df_final['Survived'] == 1]
non_survivors = df_final[df_final['Survived'] == 0]

stats_comparison = pd.DataFrame({
    'All_Passengers': df_final[['Age', 'Fare', 'Family_Size']].mean(),
    'Survivors': survivors[['Age', 'Fare', 'Family_Size']].mean(),
    'Non_Survivors': non_survivors[['Age', 'Fare', 'Family_Size']].mean()
})
print(stats_comparison.round(2))

# Outliers detection
print(f"\n3. OUTLIERS DETECTION")
print("-" * 40)

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Check for outliers in key numerical columns
numerical_columns = ['Age', 'Fare', 'Family_Size']
for col in numerical_columns:
    outlier_count, lower, upper = detect_outliers_iqr(df_final, col)
    print(f"{col}: {outlier_count} outliers detected (bounds: {lower:.2f} to {upper:.2f})")

# Correlation analysis
print(f"\n4. CORRELATION ANALYSIS")
print("-" * 40)

# Calculate correlation matrix for key numerical features
numerical_features = ['Survived', 'Age', 'Fare', 'Family_Size', 'Sex_Male', 'Has_Cabin']
correlation_matrix = df_final[numerical_features].corr()

print("Correlation with Survival (top correlations):")
survival_corr = correlation_matrix['Survived'].drop('Survived').abs().sort_values(ascending=False)
print(survival_corr.round(3))

print("\nStrong correlations (>0.3) with survival:")
strong_corr = survival_corr[survival_corr > 0.3]
for feature, corr in strong_corr.items():
    direction = "positive" if correlation_matrix.loc['Survived', feature] > 0 else "negative"
    print(f"- {feature}: {corr:.3f} ({direction})")

# Save processed data for visualization
df_for_viz = df_final.copy()
# Add back original categorical variables for better visualization
df_for_viz['Sex'] = df_processed['Sex_Male'].map({1: 'Male', 0: 'Female'})
df_for_viz['Pclass'] = (df_final['Pclass_1'] * 1 + df_final['Pclass_2'] * 2 + df_final['Pclass_3'] * 3)

print(f"\n5. DATA DISTRIBUTIONS")
print("-" * 40)

print("Age distribution:")
print(f"Mean age: {df_final['Age'].mean():.1f} years")
print(f"Median age: {df_final['Age'].median():.1f} years")
print(f"Age range: {df_final['Age'].min():.1f} to {df_final['Age'].max():.1f} years")
print(f"Standard deviation: {df_final['Age'].std():.1f} years")

print(f"\nFare distribution:")
print(f"Mean fare: {df_final['Fare'].mean():.2f}")
print(f"Median fare: {df_final['Fare'].median():.2f}")
print(f"Fare range: {df_final['Fare'].min():.2f} to {df_final['Fare'].max():.2f}")

print(f"\nFamily size distribution:")
family_size_dist = df_final['Family_Size'].value_counts().sort_index()
print(family_size_dist)