# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

# Update working directory path as per your setup
wd = "/home/cedric/repos/cassini_data/"

# Read the data
precipitation_data = pd.read_csv(wd + "precipitation_time_series.csv")
temperature_data = pd.read_csv(wd + "temperature_time_series.csv")
evapotranspiration_data = pd.read_csv(wd + "evapotranspiration_time_series.csv")
river_height_data = pd.read_csv(wd + "fake_daily_river_height_m.csv")

# Merge datasets if needed and select features
# For the purpose of this example, I'll assume all files have the same length and corresponding dates
merged_df = pd.concat(
    [precipitation_data, temperature_data, evapotranspiration_data], axis=1
)
merged_df["River_Height_m"] = river_height_data[
    "River_Height_m"
]  # Make sure the target column matches your dataset
merged_df = merged_df.drop(
    ["timestamp"], axis=1
)  # Dropping the date column as instructed
target_column = "River_Height_m"


X = merged_df.drop([target_column], axis=1)
y = merged_df[target_column]

# Split the data into training and testing sets, with temporal separation
# Assuming data is in chronological order, use the last part as test set
test_size = int(len(X) * 0.25)  # Adjust the test_size as needed, 25% for testing
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]

# If you decide to scale, uncomment the following lines
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

#%% Model training ############################################################
# Set hyperparameter space for tuning
hyperparameters = {
    "n_estimators": [10, 50, 100, 150, 200, 300, 400, 500],
    "max_features": ["auto", "sqrt", "log2"],
}

# Perform Grid Search CV to find the best parameters
rf_cv = GridSearchCV(
    RandomForestRegressor(),
    hyperparameters,
    cv=5,
    return_train_score=True,
    verbose=True,
)
rf_cv.fit(X_train, y_train)
print("Best hyperparameters:", rf_cv.best_params_)
print("Best score:", rf_cv.best_score_)

# Make predictions on the test set
y_pred = rf_cv.best_estimator_.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Feature importance
fi = pd.DataFrame(
    data=rf_cv.best_estimator_.feature_importances_,
    index=X_train.columns,
    columns=["Importance"],
).sort_values(by="Importance", ascending=False)
print(fi)

# saving the model
best_model = rf_cv.best_estimator_
dump(best_model, wd+'qi_estimator.joblib')

# Plotting is already provided in the script and can be adapted for the current prediction results.
# And visualize
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Random Forest (RF)", fontsize=15)
sns.barplot(data=fi, x="Importance", y=fi.index, ax=axs[0])
axs[0].grid()

# Plot results
axs[1].scatter(y_pred, y_test)
axs[1].set_xlabel("Predicted " + target_column, fontsize=15)
axs[1].set_ylabel("Wflow " + target_column, fontsize=15)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
text1 = "RMSLE = " + str(rmse)
text2 = "R-squared = " + str(r2)
axs[1].legend([extra, extra], [text1, text2], loc="upper left")

plt.xlim([0, 4])
plt.ylim([0, 4])
axs[1].grid()

plt.tight_layout()
plt.show()
