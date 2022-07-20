#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of geometry functions to use in the naturwb Webtool app."""

##############################################################################
#               Masterarbeit Uni Freiburg Hydrologie                         #
#                                                                            #
#  Ermittlung einer naturnahen urbanen Wasserbilanz (NatUrWB)                #
#               als Zielvorgabe für deutsche Städte                          #
#                               -                                            #
#              Erstellung eines interaktiven Webtools                        #
#                                                                            #
##############################################################################

__author__ = "Max Schmit"
__copyright__ = "Copyright 2021, Max Schmit"
__version__ = "1.0.0"

# libraries
import requests
from shapely.geometry import shape
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.errors import ShapelyDeprecationWarning
import geopandas as gpd 
import rasterio as rio
import pandas as pd
import rasterio.mask as riomask
import warnings

# functions
def geoencode(name, simplified=True):
    """Make a query to Novatim to get the best Polygon.

    Parameters
    ----------
    name : str
        The name of the german town or city to look for.
    simplified : boolean
        Should the geom get simplified?

    Returns
    -------
    GEOSGeometry.Polygon or MultiPolygons
        Simplified Geometry.
    """
    query_answ = requests.get("https://nominatim.openstreetmap.org/search?q=" +
                              name +
                              ",germany&polygon_geojson=1&format=geojson")

    # get the first polygon
    for feature in query_answ.json()["features"]:
        if feature["geometry"]["type"] in ["Polygon", "MultiPolygon"]:
            geom = shape(feature["geometry"])
            break

    if simplified:
        geom = geom.simplify(0.0001)

    return geom

def circle_part(center_xy, radius, start_angle, stop_angle):
    """Create a Portion of a circle as Polygon.

    Parameters
    ----------
    center_xy : list, array or tuple of int or floats
        The X and Y coordinates of the center.
    radius : int or float
        The radius of the circle.
    start_angle : int
        The start angle of the portion of the circle in degrees.
        0 means east.
    stop_angle : int
        The stop angle of the portion of the circle in degrees.
        0 means east.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon of the partion of the circle
    """    
    # switch start/stop angle if necessary
    if start_angle > stop_angle:
        temp = stop_angle
        stop_angle = start_angle 
        start_angle = temp
    if stop_angle - start_angle >= 360:
        return Point(center_xy).buffer(radius)

    x,y = center_xy
    coords = [center_xy]
    for ang in range(start_angle, stop_angle+1, 1):
        coords.append([
            x + np.cos(np.deg2rad(ang)) * radius,
            y + np.sin(np.deg2rad(ang)) * radius
        ])
        
    return Polygon(coords)

def polar_line(center_xy, radius, angle):
    """Create a LineString with polar coodinates.

    Parameters
    ----------
    center_xy : list, array or tuple of int or floats
        The X and Y coordinates of the center.
    radius : int or float
        The radius of the circle.
    angle : int
        The angle of the portion of the circle in degrees.
        0 means east.

    Returns
    -------
    shapely.geometry.LineString
        LineString.
    """
    coords = [center_xy]
    coords.append([
            center_xy[0] + np.cos(np.deg2rad(angle)) * radius,
            center_xy[1] + np.sin(np.deg2rad(angle)) * radius
        ])
        
    return LineString(coords)

def raster2points(raster_np, transform, crs=None):
    """Polygonize raster array to GeoDataFrame.

    Until now this only works for rasters with one band.

    Parameters
    ----------
    raster_np : np.array
        The imported raster array. 
    transform : rio.Affine
        The Affine transformation of the raster.
    crs : str or crs-type, optional
        The coordinate reference system for the raster, by default None

    Returns
    -------
    geopandas.GeoDataFrame
        The raster Data is in the data column.
    """
    mask = ~np.isnan(raster_np[0])
    cols, rows =  mask.nonzero()
    coords = rio.transform.xy(transform, cols, rows)

    geoms = [Point(xy) for xy in list(zip(*coords))]

    return gpd.GeoDataFrame(
        {"data": raster_np[0][mask]}, 
        geometry=geoms, 
        crs=crs)


def get_hab(xy, radius, dem1, dem2, dem1_crs=None, dem2_crs=None, stat_h=None):
    """Get the "Horizontabschirmung" of a Point, on the basis of two DEMs.

    The "Horizontabschirmung" is defined in Richter (95) as the angle to the horizon in the western direction.

    Parameters
    ----------
    xy : tuple of floats
        The X and Y coordinates of the Point for which to generate the "Horizontabschirmung".
        The Coordinates need to be in the DGM1's coordinate system.
    radius : int
        The maximal radius from the stations Point to look for elevations.
    dem1 : pathlike object or rasterio.io.DatasetReader
        The path or an opened DatasetReader to the first DEM.
        This DEM is the main basis to calculate the "Horizontabschirmung".
    dem2 : pathlike object or rasterio.io.DatasetReader
        The path or an opened DatasetReader to the second DEM.
        This DGM is taken to fill the holes, if the first one has not enough data.
    dem1_crs : pyproj.CRS
        The coordinate system of the first DEM as a pyproj CRS.
        If None the crs is taken from the opened DGM1 raster.
        The default is None.
    dem2_crs : pyproj.CRS
        The coordinate system of the second DEM as a pyproj CRS.
        If None the crs is taken from the opened DGM2 raster.
        The default is None.
    stat_h : int, optional
        The hight above sea of the Stations Point.
        If None, then the hight is computed from the DGM1.
        The default is None.

    Returns
    -------
    float
        The "Horizontabschirmung" angle of the given Point in degrees.
    """
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

    # get missing values
    if stat_h is None:
        stat_h = list(dem2.sample(
            xy=[xy],
            indexes=1,
            masked=True))[0]
    if dem1_crs is None:
        dem1_crs = dem1.crs
    if dem2_crs is None:
        dem2_crs = dem2.crs

    if type(dem1) != rio.io.DatasetReader:
        dem1 = rio.open(dem1, "r")
        dem1_was_path = True
    else:
        dem1_was_path = False

    if type(dem2) != rio.io.DatasetReader:
        dem2 = rio.open(dem2, "r")
        dem2_was_path = True
    else:
        dem2_was_path = False

    # get horizontabschattung
    dem1_cellsize = np.max([dem1.profile["height"], dem1.profile["width"]])
    hab = pd.Series(
        index=pd.Index([], name="angle", dtype=int), 
        name="horizontabschirmung", dtype=float)
    for angle in range(90, 271, 3):
        dem1_mask = polar_line(xy, radius, angle)
        dem1_np, dem1_tr = riomask.mask(dem1, [dem1_mask], crop=True)
        dem1_np[dem1_np==dem1.profile["nodata"]] = np.nan
        dgm_gpd = raster2points(dem1_np, dem1_tr, crs=dem1_crs)
        dgm_gpd["dist"] = dgm_gpd.distance(Point(xy))
        
        # check if parts are missing and fill
        #####################################
        dgm_gpd = dgm_gpd.sort_values("dist").reset_index(drop=True)
        line_parts = pd.DataFrame(columns=["Start_point", "radius", "line"])
        # look for holes inside
        for i, j in enumerate(dgm_gpd[dgm_gpd["dist"].diff() > 10].index):
            line_parts = pd.concat(
                [line_parts, 
                        pd.DataFrame(
                        {"Start_point": dgm_gpd.loc[j-1, "geometry"], 
                            "radius": dgm_gpd.loc[j, "dist"] - dgm_gpd.loc[j-1, "dist"]}, 
                        index=[i])])

        # look for missing values at the end
        dem1_max_dist = dgm_gpd.iloc[-1]["dist"]
        if dem1_max_dist < (radius - dem1_cellsize):
            line_parts = pd.concat(
                [line_parts,
                        pd.DataFrame(
                        {"Start_point":  dgm_gpd.iloc[-1]["geometry"], 
                            "radius": radius - dem1_max_dist}, 
                        index=[line_parts.index.max()+1])])

        # check if parts are missing and fill
        if len(line_parts) > 0:
            # create the lines
            for i, row in line_parts.iterrows():
                line_parts.loc[i, "line"] = polar_line(
                        [el[0] for el in row["Start_point"].xy],
                        row["radius"],
                        angle
                )
            line_parts = gpd.GeoDataFrame(
                line_parts, geometry="line", crs=dem1_crs
                ).to_crs(dem2_crs)
            dem2_mask = MultiLineString(line_parts["line"].tolist())
            dem2_np, dem2_tr = riomask.mask(dem2, [dem2_mask], crop=True)
            dem2_np[dem2_np==dem2.profile["nodata"]] = np.nan
            dem2_gpd = raster2points(dem2_np, dem2_tr, crs=dem2_crs
                ).to_crs(dem1_crs)
            dem2_gpd["dist"] = dem2_gpd.distance(Point(xy))
            dgm_gpd = pd.concat([dgm_gpd, dem2_gpd], ignore_index=True)
        
        hab[angle] = np.max(np.degrees(np.arctan(
            (dgm_gpd["data"]-stat_h) / dgm_gpd["dist"])))

    # close rasters if opened
    if dem1_was_path:
        dem1.close()
    if dem2_was_path:
        dem2.close()

    return (0.15*hab[(hab.index>225) & (hab.index<=270)].mean()
            + 0.35*hab[(hab.index>=180) & (hab.index<=225)].mean()
            + 0.35*hab[(hab.index>=135) & (hab.index<=180)].mean()
            + 0.15*hab[(hab.index>=90) & (hab.index<135)].mean())