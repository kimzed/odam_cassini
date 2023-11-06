# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

YEAR_START_DAM = 1974
# Update working directory path as per your setup
wd = "/home/cedric/repos/cassini_data/uk_first_basin/"

# Read the data
merged_df = pd.read_csv(wd + "/merged_input_data_time_series_1970-2020.csv")
river_downstream_data = pd.read_csv(wd + "/CAMELS_GB_hydromet_timeseries_31006_19701001-20150930.csv")

# Convert timestamps to datetime
merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
river_downstream_data['date'] = pd.to_datetime(river_downstream_data['date'])

# Filter data to include only dates before the dam was constructed
merged_df = merged_df[merged_df['timestamp'].dt.year < YEAR_START_DAM]
river_downstream_data = river_downstream_data[river_downstream_data['date'].dt.year < YEAR_START_DAM]

# Perform an inner merge with only the 'discharge_vol' column from the river_downstream_data
common_days_df = pd.merge(
    merged_df,
    river_downstream_data[['date', 'discharge_vol']],
    left_on='timestamp',
    right_on='date',
    how='inner'
)

# Remove rows with NaN values created by shifting
common_days_df = common_days_df.dropna()

target_data = common_days_df["discharge_vol"]

common_days_df = common_days_df.drop(
    ["timestamp", 'discharge_vol', 'date'], axis=1
)  # Dropping the date column as instructed

# Number of days to look back
number_days_prior = 30  # for one week; adjust this as needed
df_with_lag = common_days_df.copy()
# Create lagged features for the number of days you want to look back
for day in range(1, number_days_prior + 1):
    shifted_precipitation = common_days_df[["Precipitation"]].shift(periods=day)
    shifted_precipitation.columns = [str(col) + f'_lag_{day}' for col in shifted_precipitation.columns]
    df_with_lag = pd.concat([df_with_lag, shifted_precipitation], axis=1)

# we need to drop the first 7 days

common_days_df = df_with_lag[number_days_prior:]
target_data = target_data[number_days_prior:]


X = common_days_df
y = target_data

# Instead of a percentage-based split, use an index that respects time ordering.
# Determine the split index
split_index = int(len(X) * 0.75)  # Assuming 75% of data for training

# Split the data by index to maintain the time series order
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

#%% Model training ############################################################
# Set hyperparameter space for tuning
hyperparameters = {
    'n_estimators': [150, 200, 300, 400, 500],
    'max_features': [ .8, .6, .5, .3],
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

# After training and evaluation:
# Save the model
best_model = rf_cv.best_estimator_
dump(best_model, wd+'qi_estimator.joblib')
