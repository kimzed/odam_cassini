import ee

from ee.geometry import Geometry
from ee.imagecollection import ImageCollection


def get_daily_temperature_mean(
    image: ImageCollection, polygon: Geometry
) -> ImageCollection:
    mean = image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=polygon, scale=30
    ).get("mean_2m_air_temperature")
    return image.set("daily_mean_temp", mean)


# Define a function that takes an image and returns the mean of the image over the polygon.
def get_daily_precipitation_mean(
    image: ImageCollection, polygon: Geometry
) -> ImageCollection:
    mean = image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=polygon, scale=30
    ).get("total_precipitation")
    return image.set("daily_mean_precipitation", mean)
