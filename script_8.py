# PART 4: OPTIMIZATION - HYPERPARAMETER TUNING (SIMPLIFIED)

print("="*60)
print("PART 4: OPTIMIZATION - HYPERPARAMETER TUNING")
print("="*60)

print("\n1. SIMPLIFIED HYPERPARAMETER TUNING")
print("-" * 40)

# Use smaller parameter grids for faster execution
print("Using simplified parameter grids for demonstration...")

# Tune Logistic Regression (best performing model)
print(f"\nTuning Logistic Regression...")
lr_param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}

lr_grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    param_grid=lr_param_grid,
    cv=3,  # Reduced CV folds for speed
    scoring='accuracy',
    n_jobs=-1
)

lr_grid_search.fit(X_train, y_train)
best_lr_model = lr_grid_search.best_estimator_

print(f"✓ Logistic Regression tuning completed")
print(f"  Best CV Score: {lr_grid_search.best_score_:.4f}")
print(f"  Best Parameters: {lr_grid_search.best_params_}")

# Tune Random Forest with smaller grid
print(f"\nTuning Random Forest...")
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_

print(f"✓ Random Forest tuning completed")
print(f"  Best CV Score: {rf_grid_search.best_score_:.4f}")
print(f"  Best Parameters: {rf_grid_search.best_params_}")

# Evaluate tuned models
print("\n2. PERFORMANCE COMPARISON: BEFORE VS AFTER TUNING")
print("-" * 40)

# Original Logistic Regression performance
original_lr = LogisticRegression(random_state=42, max_iter=1000)
original_lr.fit(X_train, y_train)
original_lr_pred = original_lr.predict(X_test)
original_lr_accuracy = accuracy_score(y_test, original_lr_pred)
original_lr_f1 = f1_score(y_test, original_lr_pred)

# Tuned Logistic Regression performance
tuned_lr_pred = best_lr_model.predict(X_test)
tuned_lr_accuracy = accuracy_score(y_test, tuned_lr_pred)
tuned_lr_f1 = f1_score(y_test, tuned_lr_pred)

# Original Random Forest performance
original_rf = RandomForestClassifier(random_state=42, n_estimators=100)
original_rf.fit(X_train, y_train)
original_rf_pred = original_rf.predict(X_test)
original_rf_accuracy = accuracy_score(y_test, original_rf_pred)
original_rf_f1 = f1_score(y_test, original_rf_pred)

# Tuned Random Forest performance
tuned_rf_pred = best_rf_model.predict(X_test)
tuned_rf_accuracy = accuracy_score(y_test, tuned_rf_pred)
tuned_rf_f1 = f1_score(y_test, tuned_rf_pred)

# Create comparison table
comparison_data = {
    'Model': ['Logistic Regression', 'Random Forest'],
    'Original_Accuracy': [original_lr_accuracy, original_rf_accuracy],
    'Tuned_Accuracy': [tuned_lr_accuracy, tuned_rf_accuracy],
    'Accuracy_Improvement': [tuned_lr_accuracy - original_lr_accuracy, tuned_rf_accuracy - original_rf_accuracy],
    'Original_F1': [original_lr_f1, original_rf_f1],
    'Tuned_F1': [tuned_lr_f1, tuned_rf_f1],
    'F1_Improvement': [tuned_lr_f1 - original_lr_f1, tuned_rf_f1 - original_rf_f1]
}

comparison_df = pd.DataFrame(comparison_data)
print("Before vs After Tuning Comparison:")
print(comparison_df.round(4))

# Determine best model
best_model_idx = comparison_df['Tuned_Accuracy'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']

if best_model_name == 'Logistic Regression':
    final_best_model = best_lr_model
    best_params = lr_grid_search.best_params_
    best_cv_score = lr_grid_search.best_score_
else:
    final_best_model = best_rf_model
    best_params = rf_grid_search.best_params_
    best_cv_score = rf_grid_search.best_score_

print(f"\nBest tuned model: {best_model_name}")
print(f"Best tuned accuracy: {comparison_df.loc[best_model_idx, 'Tuned_Accuracy']:.4f}")
print(f"Improvement: {comparison_df.loc[best_model_idx, 'Accuracy_Improvement']:+.4f}")

print("\n3. DETAILED EVALUATION OF BEST TUNED MODEL")
print("-" * 40)

best_pred = final_best_model.predict(X_test)
best_cm = confusion_matrix(y_test, best_pred)

print(f"Final Best Model: {best_model_name}")
print(f"Optimized Parameters: {best_params}")
print(f"Cross-validation score: {best_cv_score:.4f}")

# Detailed metrics
accuracy = accuracy_score(y_test, best_pred)
precision = precision_score(y_test, best_pred)
recall = recall_score(y_test, best_pred)
f1 = f1_score(y_test, best_pred)

print(f"\nTest Set Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"              0    1")
print(f"Actual   0   {best_cm[0,0]:3d}  {best_cm[0,1]:3d}")
print(f"         1   {best_cm[1,0]:3d}  {best_cm[1,1]:3d}")

# Classification report
print(f"\nDetailed Classification Report:")
report = classification_report(y_test, best_pred, output_dict=True)
print(f"  Not Survived (0): Precision={report['0']['precision']:.3f}, Recall={report['0']['recall']:.3f}, F1={report['0']['f1-score']:.3f}")
print(f"  Survived (1):     Precision={report['1']['precision']:.3f}, Recall={report['1']['recall']:.3f}, F1={report['1']['f1-score']:.3f}")

print("\n4. OPTIMIZATION INSIGHTS")
print("-" * 40)

print("Key findings from hyperparameter tuning:")
if best_model_name == 'Logistic Regression':
    print(f"- Best regularization strength (C): {best_params['C']}")
    print(f"- Best solver: {best_params['solver']}")
    print(f"- Penalty type: {best_params['penalty']}")
else:
    print(f"- Best number of trees: {best_params['n_estimators']}")
    print(f"- Best maximum depth: {best_params['max_depth']}")
    print(f"- Best min samples split: {best_params['min_samples_split']}")

print(f"\nPerformance improvements:")
for idx, row in comparison_df.iterrows():
    model_name = row['Model']
    acc_imp = row['Accuracy_Improvement']
    f1_imp = row['F1_Improvement']
    print(f"- {model_name}: Accuracy {acc_imp:+.4f}, F1-Score {f1_imp:+.4f}")

print(f"\n✓ Hyperparameter tuning completed successfully!")
print(f"✓ Final optimized model ready for deployment")