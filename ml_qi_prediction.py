import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data
precipitation_data = pd.read_csv('/home/cedric/repos/cassini_data/precipitation_time_series.csv')
temperature_data = pd.read_csv('/home/cedric/repos/cassini_data/temperature_time_series.csv')
evapotranspiration_data = pd.read_csv("/home/cedric/repos/cassini_data/evapotranspiration_time_series.csv")
fake_daily_river_height = pd.read_csv('/home/cedric/repos/cassini_data/fake_daily_river_height_m.csv')

# Merge the datasets on the date column, then drop the date column
df = (precipitation_data
      .merge(temperature_data, on='timestamp')
      .merge(evapotranspiration_data, on='timestamp')
      .merge(fake_daily_river_height, on='timestamp')
      .drop('timestamp', axis=1))

# Assuming 'river_height' is the target variable in fake_daily_river_height dataframe
features = df.drop('River_Height_m', axis=1)
target = df['River_Height_m']

# Split the data into training and test sets, ensuring temporal separation
# Assuming that the data is sorted by date before the date column was dropped
test_size = int(len(df) * 0.2)  # Let's take the last 20% of data as the test set
train_features = features[:-test_size]
test_features = features[-test_size:]
train_target = target[:-test_size]
test_target = target[-test_size:]

# Train the Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune these hyperparameters
rf.fit(train_features, train_target)

# Predict on the test set
predictions = rf.predict(test_features)

# Evaluate the model
rmse = sqrt(mean_squared_error(test_target, predictions))
print(f'Test RMSE: {rmse}')
