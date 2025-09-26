# PART 3: MODEL BUILDING

print("="*60)
print("PART 3: MODEL BUILDING")
print("="*60)

# Prepare features and target
X = df_final.drop('Survived', axis=1)
y = df_final['Survived']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Feature columns: {len(X.columns)} features")

# Split the dataset into train/test sets
print("\n1. SPLITTING DATA INTO TRAIN/TEST SETS")
print("-" * 40)

# Use stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# Training set survival rate
train_survival_rate = y_train.mean()
test_survival_rate = y_test.mean()
print(f"Training set survival rate: {train_survival_rate:.1%}")
print(f"Test set survival rate: {test_survival_rate:.1%}")

print("\n2. TRAINING MULTIPLE ML MODELS")
print("-" * 40)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(random_state=42, probability=True)
}

# Store results
model_results = {}
trained_models = {}

# Train each model
print("Training models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    # Store results
    model_results[name] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred_test
    }
    
    print(f"✓ {name} trained successfully")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

print("\n3. MODEL PERFORMANCE COMPARISON")
print("-" * 40)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Train_Accuracy': [results['train_accuracy'] for results in model_results.values()],
    'Test_Accuracy': [results['test_accuracy'] for results in model_results.values()],
    'Precision': [results['precision'] for results in model_results.values()],
    'Recall': [results['recall'] for results in model_results.values()],
    'F1_Score': [results['f1_score'] for results in model_results.values()]
})

print("Model Performance Summary:")
print(comparison_df.round(4))

# Find best model based on test accuracy
best_model_name = comparison_df.loc[comparison_df['Test_Accuracy'].idxmax(), 'Model']
best_model = trained_models[best_model_name]
print(f"\nBest performing model: {best_model_name}")
print(f"Test Accuracy: {model_results[best_model_name]['test_accuracy']:.4f}")

print("\n4. DETAILED EVALUATION FOR EACH MODEL")
print("-" * 40)

for name, results in model_results.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, results['predictions'])
    print(f"  Confusion Matrix:")
    print(f"    [[{cm[0,0]:3d} {cm[0,1]:3d}]")
    print(f"     [{cm[1,0]:3d} {cm[1,1]:3d}]]")
    
    # Classification Report
    print(f"  Classification Report:")
    report = classification_report(y_test, results['predictions'], output_dict=True)
    print(f"    Class 0 (Not Survived): Precision={report['0']['precision']:.3f}, Recall={report['0']['recall']:.3f}")
    print(f"    Class 1 (Survived): Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}")

print("\n5. CROSS-VALIDATION SCORES")
print("-" * 40)

# Perform cross-validation for each model
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'cv_scores': cv_scores
    }
    print(f"{name}:")
    print(f"  CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  CV Scores: {[f'{score:.4f}' for score in cv_scores]}")

print("\n6. MODEL INSIGHTS")
print("-" * 40)

# Feature importance for tree-based models
if 'Random Forest' in trained_models:
    rf_model = trained_models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features (Random Forest):")
    print(feature_importance.head(10).round(4))

# Logistic Regression coefficients
if 'Logistic Regression' in trained_models:
    lr_model = trained_models['Logistic Regression']
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lr_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 10 Most Influential Features (Logistic Regression):")
    print(coefficients.head(10).round(4))