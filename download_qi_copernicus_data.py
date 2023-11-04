import ee
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from utils import get_daily_temperature_mean, get_daily_precipitation_mean


ee.Initialize()

basin_gdf = gpd.read_file("/home/cedric/repos/cassini_data/naussac_water_bassin.gpkg")
basin_polygon = basin_gdf.geometry.to_list()[0]
merged_polygon = unary_union(basin_polygon)
coords_water_basin = list(merged_polygon.exterior.coords)

ee_water_basin_polygon = ee.Geometry.Polygon(coords_water_basin)

# Load the ERA5 daily temperature at 2 meters dataset.
era5_daily_temp = ee.ImageCollection("ECMWF/ERA5/DAILY")

# construction of the dam starts in 1976. ends up in 1980
start_date = "2000-01-01"
end_date = "2001-01-01"

temp_filtered = era5_daily_temp.filterBounds(ee_water_basin_polygon).filterDate(
    start_date, end_date
)

# Map the function over the image collection.
daily_mean_temp = temp_filtered.map(
    lambda image: get_daily_temperature_mean(image, ee_water_basin_polygon)
)
daily_mean_precipitation = temp_filtered.map(
    lambda image: get_daily_precipitation_mean(image, ee_water_basin_polygon)
)

time_series_temp = (
    daily_mean_temp.reduceColumns(
        ee.Reducer.toList(2), ["system:time_start", "daily_mean_temp"]
    )
    .values()
    .get(0)
)
time_series_precipitation = (
    daily_mean_precipitation.reduceColumns(
        ee.Reducer.toList(2), ["system:time_start", "daily_mean_precipitation"]
    )
    .values()
    .get(0)
)


# Get the results as a Python list.
values_temperature = time_series_temp.getInfo()
values_precipitation = time_series_precipitation.getInfo()

# min max for the temperature is 223.6-304, in kelvin.
# conversion is: temp=kelvin-272.15
time_series_temp = pd.DataFrame(values_temperature, columns=["timestamp", "mean_temp"])
time_series_temp["timestamp"] = pd.to_datetime(time_series_temp["timestamp"], unit="ms")
time_series_temp.set_index("timestamp", inplace=True)

# min max for the precipitation is 0 - 0.02
time_series_precipitation = pd.DataFrame(
    values_precipitation, columns=["timestamp", "mean_precipitation"]
)
time_series_precipitation["timestamp"] = pd.to_datetime(
    time_series_precipitation["timestamp"], unit="ms"
)
time_series_precipitation.set_index("timestamp", inplace=True)
