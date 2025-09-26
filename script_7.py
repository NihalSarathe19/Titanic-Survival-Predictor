# PART 4: OPTIMIZATION - HYPERPARAMETER TUNING

print("="*60)
print("PART 4: OPTIMIZATION - HYPERPARAMETER TUNING")
print("="*60)

print("\n1. HYPERPARAMETER TUNING SETUP")
print("-" * 40)

# Since Logistic Regression performed best, we'll focus on tuning it
# But we'll also tune Random Forest for comparison
print("Performing hyperparameter tuning for:")
print("1. Logistic Regression (best performing model)")
print("2. Random Forest (ensemble method)")
print("3. Support Vector Machine (kernel method)")

# Define parameter grids for different models
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
}

print(f"\nParameter grid sizes:")
for model_name, grid in param_grids.items():
    grid_size = 1
    for param, values in grid.items():
        grid_size *= len(values)
    print(f"  {model_name}: {grid_size} combinations")

print("\n2. GRID SEARCH HYPERPARAMETER TUNING")
print("-" * 40)

# Perform GridSearchCV for each model
tuned_models = {}
grid_search_results = {}

for model_name in ['Logistic Regression', 'Random Forest']:
    print(f"\nTuning {model_name}...")
    
    # Get base model and parameter grid
    if model_name == 'Logistic Regression':
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = param_grids[model_name]
    elif model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
        # Use smaller grid for Random Forest to save time
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Store results
    tuned_models[model_name] = grid_search.best_estimator_
    grid_search_results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }
    
    print(f"✓ {model_name} tuning completed")
    print(f"  Best CV Score: {grid_search.best_score_:.4f}")
    print(f"  Best Parameters: {grid_search.best_params_}")

print("\n3. RANDOMIZED SEARCH HYPERPARAMETER TUNING")
print("-" * 40)

# Perform RandomizedSearchCV for SVM (larger parameter space)
print("Tuning Support Vector Machine with RandomizedSearchCV...")

from scipy.stats import uniform
svm_param_dist = {
    'C': uniform(0.1, 100),  # Uniform distribution between 0.1 and 100
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1]
}

svm_model = SVC(random_state=42, probability=True)
randomized_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=svm_param_dist,
    n_iter=50,  # Try 50 random combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

randomized_search.fit(X_train, y_train)

tuned_models['Support Vector Machine'] = randomized_search.best_estimator_
grid_search_results['Support Vector Machine'] = {
    'best_params': randomized_search.best_params_,
    'best_cv_score': randomized_search.best_score_,
    'best_estimator': randomized_search.best_estimator_
}

print(f"✓ SVM tuning completed")
print(f"  Best CV Score: {randomized_search.best_score_:.4f}")
print(f"  Best Parameters: {randomized_search.best_params_}")

print("\n4. PERFORMANCE COMPARISON: BEFORE VS AFTER TUNING")
print("-" * 40)

# Evaluate tuned models
tuned_results = {}
for model_name, tuned_model in tuned_models.items():
    # Make predictions
    y_pred_test_tuned = tuned_model.predict(X_test)
    
    # Calculate metrics
    accuracy_tuned = accuracy_score(y_test, y_pred_test_tuned)
    precision_tuned = precision_score(y_test, y_pred_test_tuned)
    recall_tuned = recall_score(y_test, y_pred_test_tuned)
    f1_tuned = f1_score(y_test, y_pred_test_tuned)
    
    tuned_results[model_name] = {
        'accuracy': accuracy_tuned,
        'precision': precision_tuned,
        'recall': recall_tuned,
        'f1_score': f1_tuned
    }

# Create comparison table
comparison_table = []
for model_name in tuned_models.keys():
    original_results = model_results[model_name]
    tuned_result = tuned_results[model_name]
    
    comparison_table.append({
        'Model': model_name,
        'Original_Accuracy': original_results['test_accuracy'],
        'Tuned_Accuracy': tuned_result['accuracy'],
        'Accuracy_Improvement': tuned_result['accuracy'] - original_results['test_accuracy'],
        'Original_F1': original_results['f1_score'],
        'Tuned_F1': tuned_result['f1_score'],
        'F1_Improvement': tuned_result['f1_score'] - original_results['f1_score']
    })

comparison_df = pd.DataFrame(comparison_table)
print("Before vs After Tuning Comparison:")
print(comparison_df.round(4))

# Find best tuned model
best_tuned_model_idx = comparison_df['Tuned_Accuracy'].idxmax()
best_tuned_model_name = comparison_df.loc[best_tuned_model_idx, 'Model']
best_tuned_model = tuned_models[best_tuned_model_name]

print(f"\nBest tuned model: {best_tuned_model_name}")
print(f"Best tuned accuracy: {comparison_df.loc[best_tuned_model_idx, 'Tuned_Accuracy']:.4f}")
print(f"Improvement over original: {comparison_df.loc[best_tuned_model_idx, 'Accuracy_Improvement']:+.4f}")

print("\n5. DETAILED RESULTS FOR BEST TUNED MODEL")
print("-" * 40)

best_tuned_pred = best_tuned_model.predict(X_test)
best_tuned_cm = confusion_matrix(y_test, best_tuned_pred)

print(f"Best Tuned Model: {best_tuned_model_name}")
print(f"Parameters: {grid_search_results[best_tuned_model_name]['best_params']}")
print(f"Cross-validation score: {grid_search_results[best_tuned_model_name]['best_cv_score']:.4f}")
print(f"Test accuracy: {tuned_results[best_tuned_model_name]['accuracy']:.4f}")
print(f"Test precision: {tuned_results[best_tuned_model_name]['precision']:.4f}")
print(f"Test recall: {tuned_results[best_tuned_model_name]['recall']:.4f}")
print(f"Test F1-score: {tuned_results[best_tuned_model_name]['f1_score']:.4f}")

print(f"\nConfusion Matrix:")
print(f"  [[{best_tuned_cm[0,0]:3d} {best_tuned_cm[0,1]:3d}]")
print(f"   [{best_tuned_cm[1,0]:3d} {best_tuned_cm[1,1]:3d}]]")

print(f"\nClassification Report:")
report = classification_report(y_test, best_tuned_pred, output_dict=True)
print(f"  Class 0 (Not Survived): Precision={report['0']['precision']:.3f}, Recall={report['0']['recall']:.3f}")
print(f"  Class 1 (Survived): Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}")

# Save best model for deployment
best_final_model = best_tuned_model
print(f"\n✓ Best model saved for deployment: {best_tuned_model_name}")