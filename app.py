import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
# from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("Health data.csv")
df.sample(3)
# Convert status values
df['Status'] = df['Status'].map({0: 0, 1: 0, 2: 1})

# Drop last 3 rows if needed
df = df.iloc[:-3]

# EDA - You can expand with plots here
print(df.describe())
print(df.isnull().sum())

# # Correlation heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set up the plotting environment
# sns.set(style="whitegrid")
# plt.figure(figsize=(18, 12))

# # Distribution of Pulse
# plt.subplot(2, 2, 1)
# sns.histplot(df['pulse'], kde=True, color='black')
# plt.title("Pulse Distribution")

# # Distribution of Temperature
# plt.subplot(2, 2, 2)
# sns.histplot(df['body temperature'], kde=True, color='blue')
# plt.title("Temperature Distribution")

# # Distribution of SpO2
# plt.subplot(2, 2, 3)
# sns.histplot(df['SpO2'], kde=True, color='green')
# plt.title("SpO2 Distribution")

# # Count of each Status
# plt.subplot(2, 2, 4)
# sns.countplot(x='Status', data=df, palette='Set3')
# plt.title("Status Counts")

# plt.tight_layout()
# plt.show()

# # Pairplot
# sns.pairplot(df, hue='Status', palette='husl')
# plt.show()

# # Correlation heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Feature Correlation Heatmap")
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns

# features = ['pulse', 'body temperature', 'SpO2']
# plt.figure(figsize=(18, 5))

# for i, col in enumerate(features):
#     plt.subplot(1, 3, i+1)
#     sns.boxplot(x='Status', y=col, data=df, palette='pastel')
#     plt.title(f"{col} vs Status")
#     plt.xlabel("Status")
#     plt.ylabel(col)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(16, 5))

# # Pulse vs Temperature
# plt.subplot(1, 2, 1)
# sns.scatterplot(data=df, x='pulse', y='body temperature', hue='Status', palette='Set2')
# plt.title("Pulse vs Temperature by Status")

# # Pulse vs SpO2
# plt.subplot(1, 2, 2)
# sns.scatterplot(data=df, x='pulse', y='SpO2', hue='Status', palette='Set1')
# plt.title("Pulse vs SpO2 by Status")

# plt.tight_layout()
# plt.show()

# # Train/Test split
X = df.drop(columns=["Status"])
y = df["Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Models and metrics
# models = {
#     "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
#     "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
#     "Logistic Regression": LogisticRegression(),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# }

# results = {"Model": [], "Accuracy": [], "MSE": []}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     acc = accuracy_score(y_test, preds)
#     mse = mean_squared_error(y_test, preds)
#     results["Model"].append(name)
#     results["Accuracy"].append(acc)
#     results["MSE"].append(mse)
#     print(f"{name} Classification Report:\n{classification_report(y_test, preds)}\n")
#     print(model.predict(np.array([55,26,50]).reshape(1,-1)))

# # Results DataFrame
# results_df = pd.DataFrame(results)
# print(results_df)

# # Plot comparison
# results_df.plot(kind='bar', x='Model', y=['Accuracy', 'MSE'], figsize=(10,5), title="Model Comparison")
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# import pickle
# filename = 'rfmodel1.sav'
# pickle.dump(Gradient Boosting, open(filename, 'wb'))

model1=GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model1.fit(X_train, y_train)
preds = model1.predict(X_test)
acc = accuracy_score(y_test, preds)
mse = mean_squared_error(y_test, preds)
# results["Model"].append(name)
# results["Accuracy"].append(acc)
# results["MSE"].append(mse)
# print(f"{name} Classification Report:\n{classification_report(y_test, preds)}\n")
print(model1.predict(np.array([100,92,23]).reshape(1,-1)))

import pickle
filename = 'rfmodel_8.sav'
pickle.dump(model1, open(filename, 'wb'))