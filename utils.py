import ee

from ee.geometry import Geometry
from ee.imagecollection import ImageCollection


def get_spatial_mean(
    image: ImageCollection,
    polygon: Geometry,
    band: str,
) -> ImageCollection:
    # we set best effort to false to prevent issue with max number of pixels
    mean = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=30,
        bestEffort=False,
    ).get(band)
    return image.set("daily_mean_temp", mean)
