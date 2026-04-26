# ==========================================
# MACHINE LEARNING PROJECT USING SVR
# TRAFFIC SIGNAL TIME PREDICTION
# ==========================================


# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("/content/advanced_traffic_optimization_data.csv")

print("Dataset Loaded Successfully")

print(df.head())


# ==========================================
# CHECK NULL VALUES
# ==========================================

print("\nMissing Values:")

print(df.isnull().sum())


# ==========================================
# CLEAN NULL VALUES
# ==========================================

numeric_columns = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='mean')

df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

print("\nNull Values Cleaned Successfully")


# ==========================================
# SHOW COLUMN NAMES
# ==========================================

print("\nDataset Columns:")

print(df.columns)


# ==========================================
# TARGET COLUMN
# ==========================================

# Change this according to your dataset
target_column = "Optimized_Green_Time_Sec"


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(target_column, axis=1)

y = df[target_column]

# Keep only numeric columns
X = X.select_dtypes(include=np.number)

print("\nFeatures:")

print(X.columns)


# ==========================================
# STANDARDIZE DATA
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData Standardized Successfully")


# ==========================================
# SCATTER PLOT
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(X.iloc[:,0], y)

plt.xlabel(X.columns[0])

plt.ylabel(target_column)

plt.title("Scatter Plot")

plt.grid(True)

plt.show()


# ==========================================
# IMPORTANT FEATURE SELECTION
# ==========================================

selector = SelectKBest(
    score_func=f_regression,
    k='all'
)

selector.fit(X_scaled, y)

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

feature_scores = feature_scores.sort_values(
    by='Score',
    ascending=False
)

print("\nImportant Features:")

print(feature_scores)


# ==========================================
# PCA FEATURE REDUCTION
# ==========================================

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Applied Successfully")

print("Original Shape :", X_scaled.shape)

print("Reduced Shape :", X_pca.shape)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed")


# ==========================================
# CREATE SVR MODEL
# ==========================================

svr_model = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)


# ==========================================
# TRAIN MODEL
# ==========================================

svr_model.fit(X_train, y_train)

print("\nSVR Model Trained Successfully")


# ==========================================
# PREDICTION
# ==========================================

y_pred = svr_model.predict(X_test)


# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)

print("R2 Score :", r2)


# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()


# ==========================================
# SAVE MODEL FILES
# ==========================================

joblib.dump(svr_model, "svr_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(pca, "pca.pkl")

print("\nModel Files Saved Successfully")


# ==========================================
# TEST SAMPLE PREDICTION
# ==========================================

sample_data = [[
    120,
    45,
    60,
    3,
    1
]]

sample_scaled = scaler.transform(sample_data)

sample_pca = pca.transform(sample_scaled)

prediction = svr_model.predict(sample_pca)

print("\nPredicted Green Signal Time:")

print(prediction[0])
import joblib

joblib.dump(svr_model, "svr_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("Model Saved Successfully")