import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pycaret.classification import *

# Load data
data = pd.read_csv('datatraining.csv')
df = pd.DataFrame(data)
test_data = pd.read_csv('datatest.csv')
df_test = pd.DataFrame(test_data)
validation_data = pd.read_csv('datatest2.csv')
df_val = pd.DataFrame(validation_data)

# Define models
model_1_old = df[['CO2', 'Light', 'Occupancy']]
model_1_new = df[['CO2', 'Temperature', 'HumidityRatio', 'Occupancy']]
model_2_old = df[['Temperature', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']]
model_2_new = df[['Temperature', 'CO2', 'HumidityRatio', 'Humidity', 'Occupancy']]

# --- Model 1 (Old) ---
experiment_old_1 = setup(model_1_old, target='Occupancy', fix_imbalance=True)
best_old_1 = compare_models()
rf_old_1 = create_model('rf', verbose=False)
tuned_rf_old_1 = tune_model(rf_old_1, verbose=False)
final_rf_old_1 = finalize_model(tuned_rf_old_1)

# --- Model 1 (New) ---
experiment_new_1 = setup(model_1_new, target='Occupancy', fix_imbalance=True)
best_new_1 = compare_models()
gbc_new_1 = create_model('gbc', verbose=False)
tuned_gbc_new_1 = tune_model(gbc_new_1, verbose=False)
final_gbc_new_1 = finalize_model(tuned_gbc_new_1)
et_new_1 = create_model('et', verbose=False)
tuned_et_new_1 = tune_model(et_new_1, verbose=False)
final_et_new_1 = finalize_model(tuned_et_new_1)

# --- Model 2 (Old) ---
experiment_old_2 = setup(model_2_old, target='Occupancy', fix_imbalance=True)
best_old_2 = compare_models()
rf_old_2 = create_model('rf', verbose=False)
tuned_rf_old_2 = tune_model(rf_old_2, verbose=False)
final_rf_old_2 = finalize_model(tuned_rf_old_2)
et_old_2 = create_model('et', verbose=False)
tuned_et_old_2 = tune_model(et_old_2, verbose=False)
final_et_old_2 = finalize_model(tuned_et_old_2)

# --- Model 2 (New) ---
experiment_new_2 = setup(model_2_new, target='Occupancy', fix_imbalance=True)
best_new_2 = compare_models()
gbc_new_2 = create_model('gbc', verbose=False)
tuned_gbc_new_2 = tune_model(gbc_new_2, verbose=False)
final_gbc_new_2 = finalize_model(tuned_gbc_new_2)
et_new_2 = create_model('et', verbose=False)
tuned_et_new_2 = tune_model(et_new_2, verbose=False)
final_et_new_2 = finalize_model(tuned_et_new_2)

# --- Print Results ---
print("\n--- Model 1 (Old - CO2 & Light) ---")
print(pull())
print(f"Tuned Random Forest Accuracy: {round(final_rf_old_1.score(experiment_old_1.X_test, experiment_old_1.y_test), 4)}")

print("\n--- Model 1 (New - CO2, Temp & HumidityRatio) ---")
print(pull())
print(f"Tuned Gradient Boosting Classifier Accuracy: {round(final_gbc_new_1.score(experiment_new_1.X_test, experiment_new_1.y_test), 4)}")
print(f"Tuned Extra Trees Classifier Accuracy: {round(final_et_new_1.score(experiment_new_1.X_test, experiment_new_1.y_test), 4)}")

print("\n--- Model 2 (Old - CO2, Light, Temp & HumidityRatio) ---")
print(pull())
print(f"Tuned Random Forest Accuracy: {round(final_rf_old_2.score(experiment_old_2.X_test, experiment_old_2.y_test), 4)}")
print(f"Tuned Extra Trees Classifier Accuracy: {round(final_et_old_2.score(experiment_old_2.X_test, experiment_old_2.y_test), 4)}")

print("\n--- Model 2 (New - CO2, Temp, HumidityRatio & Humidity) ---")
print(pull())
print(f"Tuned Gradient Boosting Classifier Accuracy: {round(final_gbc_new_2.score(experiment_new_2.X_test, experiment_new_2.y_test), 4)}")
print(f"Tuned Extra Trees Classifier Accuracy: {round(final_et_new_2.score(experiment_new_2.X_test, experiment_new_2.y_test), 4)}")