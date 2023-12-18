#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of functions to work with GIS data."""

__author__ = "Max Schmit"
__copyright__ = "Copyright 2021, Max Schmit"

# libraries
import rasterio as rio
from shapely.geometry import Polygon, MultiPolygon, MultiLineString
from shapely.ops import shared_paths
from shapely import wkt
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

import numpy as np
import itertools
from pathlib import Path
import multiprocessing as mp
import tempfile
from matplotlib import pyplot as plt
from progressbar import progressbar
from sklearn.linear_model import LinearRegression
import pyproj
import subprocess
import time
from packaging import version


def resample_raster(input_raster_fp, scale, output_raster_fp=""):
    """Resample a raster file by a specific scale factor.

    Parameters
    ----------
    input_raster_fp : path-like Object or String
        The filepath of the input raster file
    scale : int
        The scale factor.
        A scale > 1: decreases the cell size.
        A scale < 1 : increases the cell size.
    output_raster_fp : str or path-like Object, optional
        The filepath of the output raster file.
        If empty string (default), then the filepath of the input file is taken followed by "resampled-scale".txt
        The default is "".
    """
    if type(input_raster_fp) == str:
        input_raster_fp = Path(input_raster_fp)

    # read and resample raster
    with rio.open(input_raster_fp, nodata=9999) as input_raster:
         # resample data to target shape
        resampled_raster_values = input_raster.read(
            out_shape=(
                input_raster.count,
                int(input_raster.height * scale),
                int(input_raster.width * scale)
            ),
            resampling=rio.enums.Resampling.bilinear
        )

        # scale image transform
        transform = input_raster.transform * input_raster.transform.scale(
            (input_raster.width // resampled_raster_values.shape[-1]),
            (input_raster.height // resampled_raster_values.shape[-2])
        )

        # get profile
        profile = input_raster.profile
        band_names = input_raster.descriptions

    # output new resampled raster
    if output_raster_fp == "":
        output_raster_fp = input_raster_fp.parent.joinpath(input_raster_fp.stem + "_resampled-" + str(scale) + ".tif")
    elif type(output_raster_fp) == str:
        output_raster_fp = Path(output_raster_fp)

    profile.update({"transform": transform,
                   "width": profile["width"] * scale,
                   "height": profile["height"] * scale})

    with rio.open(output_raster_fp, "w", **profile) as output_raster:
        output_raster.write(resampled_raster_values)
        for i, band_name in enumerate(band_names, start=1):
            output_raster.set_band_description(i, band_name)


def neighboor_xys(xy_null, dist):
    """Calculate 9 coordinates in a specific distance from a base point."""
    if dist==0:
        return [xy_null]

    diag_dist = np.cos(np.pi/4)*dist
    diag_adds_1 = list(itertools.product([diag_dist, -diag_dist], [diag_dist, -diag_dist]))
    diag_adds_2 = list(itertools.product([np.cos(np.pi/8)*dist, -np.cos(np.pi/8)*dist],
                                         [np.sin(np.pi/8)*dist, -np.sin(np.pi/8)*dist]))
    diag_adds_3 = list(itertools.product([np.cos(np.pi*3/8)*dist, -np.cos(np.pi*3/8)*dist],
                                         [np.sin(np.pi*3/8)*dist, -np.sin(np.pi*3/8)*dist]))

    rect_adds = list(itertools.product([dist, 0, -dist], [dist, 0, -dist]))
    for rem in (list(itertools.product([dist, -dist], [dist, -dist])) + [(0,0)]):
        rect_adds.remove(rem)

    neigh_xys_adds = rect_adds + diag_adds_1 + diag_adds_2 + diag_adds_3
    neigh_xys = (np.array(xy_null) + np.array(neigh_xys_adds)).tolist()
    neigh_xys = [tuple(neigh_xy) for neigh_xy in neigh_xys]
    return neigh_xys


def explode(ingdf):
    """
    Explode a GeoDataFrame with MultiPolygons to a GeoDataFrame with Polygons.

    Attributes will get multiplied.

    Source: https://gist.github.com/mhweber/cf36bb4e09df9deee5eb54dc6be74d26

    better use the geoDataFrame methode explode

    Parameters
    ----------
    ingdf : geopandas.GeoDataFrame
        The GeoDataFrame with MultiPolygons.

    Returns
    -------
    outgdf : geopandas.GeoDataFrame
        The GeoDataFrame with Polygons.

    """
    outgdf = gpd.GeoDataFrame(columns=ingdf.columns, crs=ingdf.crs)
    for idx, row in ingdf.iterrows():
        if type(row.geometry) == Polygon:
            outgdf = outgdf.append(row, ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=ingdf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs, ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom, 'geometry'] = row.geometry[geom]
            outgdf = outgdf.append(multdf, ignore_index=True)
    outgdf.crs = ingdf.crs
    return outgdf


def simplify_shps(gdf_in, area_lim, cat_col, keep_cols=None, comp_raster=None):
    """
    Simplify a set of Polygons smaller than a limit.

    Polygons with smaller area than the limit get the category of their
    neighbor with the longest common border.
    The Polygons will get dissolved to this category,
    but without generating MultiPolygons

    Parameters
    ----------
    gdf_in : geopandas.GeoDataFrame
        The GeoDataFrame with the Polygons to get simplified.
        !!!The index column will be lost,
        as the shapes change and the index would be confusing.
    area_lim : int
        Polygons with an area lower than this limit will get simplified.
    cat_col : str or list of str
        The name of the column in the gdf_in on which to categorise the Polygons.
        Takes the value from this column of the neighboor.
        Dissolves over this column.
        to set for own value before merging.
    keep_cols : str or list of str, optional
        The name of the columns for which to keep the attribute unchanged.
        Those attribute won't get changed and
        shapes with different attributes won't get merged.
        If None only the cat_col will later be in the output GeoDataFrame.
        The default is None.
    comp_raster : str, filepathlike-Path, dict or None, optional
        The raster on which basis to compare the neighboor.
        If a dict is given it should contain:
            np_array: the numpy array of the raster
            crs: the crs of the raster
            transform: the affine transformation of the raster
            nodata: the value for nodata of the raster
        For every Polygon the mean value is calculated from this raster to compare the difference to the neighboor.
        If given the neighboor cell with the least difference to the inspected cell
        is taken as replacement and not the cell with the largest border.
        If None then the neighboor cell with the largest common border is taken as replacement.
        The default is None.

    Returns
    -------
    gdf_out: geopandas.GeoDataFrame.
        The simplified GeoDataFrame.

    """
    # set dissolve columns
    if type(cat_col) == str:
        cat_col = [cat_col]
    if keep_cols is None:
        dis_cols = cat_col
    else:
        dis_cols = cat_col + list(keep_cols)

    # change NA Values to have a value
    gdf_out = gdf_in.reset_index(drop=True)\
        .explode(ignore_index=True, index_parts=False)\
        .fillna("NAN")

    # get the comp_raster if needed
    if comp_raster is not None:
        if type(comp_raster) == dict:
            raster_crs = comp_raster["crs"]
            zonalstat_kwargs = dict(
                raster=comp_raster["np_array"],
                affine=comp_raster["transform"],
                stats=["mean"],
                nodata=comp_raster["nodata"],
                all_touched=True)
        else:
            with rio.open(comp_raster) as raster:
                raster_crs = raster.crs
                zonalstat_kwargs = dict(
                    raster=raster.read(1),
                    affine=raster.transform,
                    stats=["mean"],
                    nodata=raster.profile["nodata"],
                    all_touched=True)

    # prepare iteration
    mask_shps_small = gdf_out[gdf_out.to_crs(epsg=31467).area < area_lim]
    mask_shps_small = mask_shps_small.loc[mask_shps_small.area.sort_values().index] # sort smallest areas first
    len_before = 3; len_after = 2 #;num_changed = 1 # just some nonesens to enter the loop

    gdf_out["changed"] = False

    # itterate until simplified
    while ((len(mask_shps_small) > 0) &
           #(num_changed > 0) &
           (len_before != len_after)):
        # get category of biggest neighbor if polygon is too small
        for i, row in mask_shps_small.iterrows():
            geom = row.geometry
            touch = gdf_out[gdf_out.touches(geom) | gdf_out.intersects(geom)].copy()

            # filter none valid polygons
            touch = touch[(touch[cat_col] != row[cat_col]).sum(axis=1) > 0]
            if keep_cols is not None:
                for keep_col in keep_cols:
                    touch = touch[touch[keep_col] == row[keep_col]]

            # next if no neighbor or has changed in this iteration
            # if (touch["changed"].sum() > 0) | touch.empty:
            #     continue
            if (touch["changed"].sum() > 0):
                touch = touch.dissolve(dis_cols, sort=False, as_index=False
                                      ).explode(ignore_index=True, index_parts=False)
            elif touch.empty:
                continue

            # get the right neighboor to replace
            repl_id = None
            if len(touch) == 1:
                repl_id = touch.iloc[0].name

            if comp_raster is not None and repl_id is None:
                # get the raster values
                touch["raster_mean"] = pd.DataFrame(
                    zonal_stats(touch.to_crs(raster_crs)["geometry"], **zonalstat_kwargs),
                    index=touch.index)["mean"]
                row["raster_mean"] = zonal_stats(
                    mask_shps_small.loc[[i]].to_crs(raster_crs)["geometry"].iloc[0],
                    **zonalstat_kwargs)[0]["mean"]

                # get the neighboor with the least difference to the own value in comp_col column
                touch["raster_diff"] = (touch["raster_mean"] - row["raster_mean"]).abs()
                repl_id = touch["raster_diff"].idxmin()

                # check if several neighbors with same value -> bigest border
                if touch.loc[repl_id, "raster_diff"] in touch.drop(repl_id)["raster_diff"]:
                    touch = touch[touch["raster_diff"] == touch.loc[repl_id, "raster_diff"]]
                    repl_id = None

            # get biggest neighbor
            if repl_id is None:
                geom_bound = geom.boundary
                for t_i, nb_bound in zip(touch.index, touch.boundary):

                    # check for multilines in neighbor
                    if type(nb_bound) == MultiLineString: # e.g. Polygone with hole
                        for nb_bound_i in nb_bound.geoms:
                            if nb_bound_i.intersects(geom) or nb_bound_i.touches(geom):
                                nb_bound = nb_bound_i

                    # check for multilines in geom
                    if type(geom_bound) == MultiLineString:
                        lengths = [shared_paths(geom_bound_i, nb_bound).length
                                    for geom_bound_i in geom_bound.geoms]
                        touch.loc[t_i, "shared border length"] = max(lengths)
                    else:
                        touch.loc[t_i, "shared border length"] = shared_paths(
                            geom_bound, nb_bound).length

                repl_id = touch["shared border length"].idxmax()

            # replace values
            if repl_id is not None:
                gdf_out.loc[i, cat_col] = touch.loc[repl_id, cat_col]
                gdf_out.loc[i, "changed"] = True

        # num_changed = gdf_out["changed"].sum()
        # dissolve to Polygons
        len_before = len(gdf_out)
        gdf_dis = gdf_out.dissolve(by=dis_cols, sort=False, as_index=False)
        gdf_out = gdf_dis.explode(ignore_index=True, index_parts=False)
        len_after = len(gdf_out)

        # restart loop values
        gdf_out["changed"] = False
        mask_shps_small = gdf_out[gdf_out.to_crs(epsg=31467).area < area_lim]
        mask_shps_small = mask_shps_small.loc[
            mask_shps_small.area.sort_values().index] # sort smallest areas first

    # change nan back
    for colname, col in gdf_out.items():
        if col.dtype in [str, object]:
            gdf_out.loc[col == "NAN", colname] = np.nan
    gdf_out.drop("changed", axis=1, inplace=True)
    try:
        gdf_out = gdf_out.astype(gdf_in.dtypes)
    except:
        print("The dtypes of the columns couldn't get restored.")

    return gdf_out

def _simplify_shps_mp_part(temp_shp_fp, kwargs):
    temp_shp_fp = Path(temp_shp_fp)
    in_gdf = gpd.read_file(temp_shp_fp)
    del_shp(temp_shp_fp)
    out_gdf = simplify_shps(in_gdf, **kwargs)
    out_gdf.to_file(temp_shp_fp.parent.joinpath(temp_shp_fp.stem + "_simplified.shp"))

def simplify_shps_mp(
        gdf_in, dist_lim=10000,
        **kwargs):
    """
    Simplify a set of Polygons by a column and an area limit.

    Polygons with smaller area than the limit get the category of their
    neighbor with the longest common border.
    The Polygons will get dissolved to this category,
    but without generating MultiPolygons

    Parameters
    ----------
    gdf_in : geopandas.GeoDataFrame
        The GeoDataFrame with the Polygons to get simplified.
        !!!The index column will be lost,
        as the shapes change and the index would be confusing.
    dist_lim : int
        The distance in meters around the middle-lines around wich the polygons in the first simplification won't get simplified.
        The Default is 10000.
    **kwargs:
        The keyword arguments for simplify_shps() function.

    Returns
    -------
    gdf_out: geopandas.GeoDataFrame.
        The simplified GeoDataFrame.

    """
    # get bounds
    max_bounds = gdf_in.bounds.max()
    min_bounds = gdf_in.bounds.min()
    mean_x = (max_bounds["maxx"] + min_bounds["minx"])/2
    mean_y = (max_bounds["maxy"] + min_bounds["miny"])/2

    # split in 4 parts
    parts = []
    parts_border = []
    # 1. Quarter
    parts.append(gdf_in[(gdf_in.bounds["miny"] >= mean_y) & (gdf_in.bounds["maxx"] <= mean_x)].copy())
    parts_border.append(parts[-1][
        (parts[-1].bounds["miny"] < (mean_y + dist_lim)) &
        (parts[-1].bounds["maxx"] > (mean_x - dist_lim))].copy())
    # 2. Quarter
    parts.append(gdf_in[(gdf_in.bounds["miny"] >= mean_y) & (gdf_in.bounds["minx"] > mean_x)].copy())
    parts_border.append(parts[-1][
        (parts[-1].bounds["miny"] < (mean_y + dist_lim)) &
        (parts[-1].bounds["minx"] < (mean_x + dist_lim))].copy())
    # 3. Quarter
    parts.append(gdf_in[(gdf_in.bounds["maxy"] < mean_y) & (gdf_in.bounds["maxx"] <= mean_x)].copy())
    parts_border.append(parts[-1][
        (parts[-1].bounds["maxy"] > (mean_y - dist_lim)) &
        (parts[-1].bounds["maxx"] > (mean_x - dist_lim))].copy())
    # 4. Quarter
    parts.append(gdf_in[(gdf_in.bounds["maxy"] < mean_y) & (gdf_in.bounds["minx"] > mean_x)].copy())
    parts_border.append(parts[-1][
        (parts[-1].bounds["maxy"] > (mean_y - dist_lim)) &
        (parts[-1].bounds["minx"] < (mean_x + dist_lim))].copy())

    # exclude small shapes in border area
    for part, part_border in zip(parts, parts_border):
        part_excl_shps = part_border[part_border.area < kwargs["area_lim"]]
        part.drop(part_excl_shps.index, inplace=True)

    # get the excluded shapes as one GeoDataFrame
    indexes_in_parts = []
    for part in parts:
        indexes_in_parts += part.index.to_list()
    excluded_shps = gdf_in[gdf_in.index.isin(indexes_in_parts) == False].copy()


    # create temporary directory to exchange files
    # temp_dir_obj = tempfile.TemporaryDirectory()
    # temp_dir = temp_dir_obj.name
    with tempfile.TemporaryDirectory() as temp_dir:
        # save parts to disk
        temp_dir = Path(temp_dir)
        for i, part in enumerate(parts):
            part.to_file(temp_dir.joinpath("temp_part_{}.shp".format(str(i))))
        del parts, parts_border

        # start Multiproccesses
        pool = mp.Pool(processes=4)
        for i in range(4):
            res = pool.apply_async(
                _simplify_shps_mp_part,
                args=(
                    str(temp_dir.joinpath("temp_part_{}.shp".format(str(i)))),
                    kwargs)
            )
        pool.close()
        pool.join()
        pool.terminate()

        # load and merge simplified parts
        parts_simplified = []
        for i in range(4):
            part_fp = temp_dir.joinpath("temp_part_{}_simplified.shp".format(str(i)))
            parts_simplified.append(gpd.read_file(part_fp))
            del_shp(part_fp)

    # temp_dir_obj.cleanup()

    if excluded_shps.index.name in parts_simplified[0].columns:
        excluded_shps.reset_index(inplace=True)

    simpl_gdf = pd.concat(
        parts_simplified +
        [excluded_shps[parts_simplified[0].columns]]
        ).reset_index(drop=True)

    del parts_simplified, excluded_shps

    return simplify_shps(simpl_gdf, **kwargs)


def del_shp(fp):
    """Delete all the files of one ESRI-Shape file.

    fp : path-like object or string
        The file to be deleted.
    """
    fp = Path(fp) if (type(fp) == str) else fp
    if fp.suffix == ".shp":
        for fp_i in list(fp.parent.glob(fp.stem + ".*")):
            fp_i.unlink()
            pass
    else:
        raise Warning(
            "The given file ({fp}) is not a valid ESRI-Shape file.\nNo file got deleted."\
                .format(fp=fp))

def load_geo_csv(fp, crs, pd_kwargs):
    """Load a csv file with a geometry column as GeoDataFrame.

    Parameters
    ----------
    fp : path-like-Object or fileIO-object
        The filepath of the csv file.
    crs : int or str
        The coordinate reference system.
        e.g. "EPSG:4326"
    pd_kwargs : dict
        The kwargs for the pandas.read_csv() function.
        e.g. {"index_col":"id"}

    Returns
    -------
    geopandas.GeoDataFrame
        The GeoDataFrame of the csv file.
    """
    df = pd.read_csv(fp, **pd_kwargs)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, crs=crs)


def raster_to_contour_polys(raster_array, transform, crs, levels):
    """Generate contour-Polygons from a raster array.

    This function does not work when there are holes in the raster shape.

    Parameters
    ----------
    raster_array : np.Array
        A raster Array on which basis the contours should get calculated.
        create with rasterio.open(...).read(...)
    transform : rasterio.Affine
        The rasters Affine transformation
    crs : pyproj.crs
        The coordinate reference system  of the rasters input
        in a foramat that is understood by geopandas.
    levels : list of int
        The levels on which contour lines should get created.

    Returns
    -------
    geoapndas.GeoSeries
        The contour polygons.
    """
    cs = plt.contourf(raster_array[0], levels=levels)
    plt.close()
    polys = []
    categories = []
    for i in range(len(cs.collections)):
        paths = cs.collections[i].get_paths()
        if len(paths) == 0:
            continue
        for path in paths:
            if len(path)<=2:
                continue
            poly = None
            for j, path_j in enumerate(path.to_polygons()):
                x = path_j[:,0]
                y = path_j[:,1]
                if len(x)<=2:
                    continue
                new_shape = Polygon([transform * xy for xy in zip(x,y)])
                if j == 0:
                    poly = new_shape
                else:
                    poly = poly.difference(new_shape)
            if poly is not None:
                polys.append(poly)
                categories.append(i)
    return gpd.GeoSeries(polys, index=categories, crs=crs).buffer(0)

def refine_climate_grid(cl, cl_dem, cl_p, dem_fine, dem_fine_p, dem_fine_src,
        window_radius = 2,
        moving_wind_size=False, max_window_radius = 6, min_dem_std = 4,
        fix_intercept=False, scale_mean=False):
    """Refine a climate grid on the basis of a finer DEM.

    Parameters
    ----------
    cl : array
        The climate array of the input raster.
    cl_dem : array
        The digital elevation model of the climate raster.
        Must be the same CRS and shape of the climate grid.
    cl_p : dict
        The rasterio profile of the climate grid
    dem_fine : array
        The finer digital elevation model on witch basis the refinement is done.
    dem_fine_p : dict
        The rasterio profile of the dem_fine grid
    dem_fine_src : rasterio.io.DatasetReader or string
        The opened or closed DatasetReader of the finer DEM.
        Can also be the filepath to the grid file.
    window_radius : int, optional
        The number of cells in each direction around the analyzed climate cell to take into account for the regression.
        1 would mean that all the 8 direct neighbor cells + plus teh actual cell are taken into account.
        The default is 2.
    moving_wind_size : bool, optional
        Should a moving window size be applied.
        This means, that the given window_size is only a minimal number.
        If the standard deviation of the DEM cells in the window is lower than 4 meters, the window is expanded by 1 until the standard deviation is enough or more than 4 was added to the window_size.
        This methods results in better results for flat regions.
        The default is False.
    max_window_radius : int, optional
        Only used if moving_window_size is True.
        Sets the maximum radius of the window in Climate cell units.
        See window_radius for more explanations.
        The default is 6.
    min_dem_std : int, optional
        Only used if moving_window_size is True.
        Sets the minimal standard deviation of the climate DEM that is accepted in a window.
        The default is 4.
    fix_intercept : bool, optional
        Fix the intercept of the Regression line to 0.
        The default is False.
    scale_mean : bool, optional
        Rescale the refined cells to match the old cell mean.
        The default is False.

    Returns
    -------
    cl_new, quot_of_mean, cl_dem_std, used_wind_size
        cl_new : array
            The refined climate raster in the CRS and shape of the dem_fine grid.
        quot_of_mean: array
            The quotient of the newly calculated cell means of the refined raster compared to the old cell values in %.
            The grid is in the cl grid crs and shape.
         cl_dem_std: array
            The standard deviation of the dem cells in the used window for a given cell.
            The grid is in the cl grid crs and shape.
        used_wind_size: array
            The used Window size for a given climate cell.
            The grid is in the cl grid crs and shape.
    """
    # check input
    if type(dem_fine_src)==str:
        dem_fine_src = rio.open(dem_fine_src, "r")

    rowcols_bounds_raw = np.array([[-0.5,-0.5, 0.5, 0.5], [0.5,-0.5,-0.5,0.5]]).reshape((2,-1))
    tr_cl_to_dem = pyproj.Transformer.from_crs(cl_p["crs"], dem_fine_p["crs"], always_xy=True)
    cl_new = np.zeros(dem_fine.shape, int).repeat(cl_p["count"])
    cl_new[:] = cl_p["nodata"]
    cl_new = cl_new.reshape((cl_p["count"], *dem_fine.shape))
    quot_of_mean = np.zeros(cl.shape)
    quot_of_mean[:,:,:] = np.nan
    cl_dem_std = np.zeros(cl_dem.shape)
    cl_dem_std[:,:] = np.nan
    used_wind_size = np.zeros(cl_dem.shape)
    used_wind_size[:,:] = np.nan

    # itter over climate cells
    for col in progressbar(range(cl_p["width"]), line_breaks=False):
        col_min, col_max = (max(col-window_radius,0),min(col+window_radius,cl_p["width"]))
        for row in range(cl_p["height"]):
            dem_cell_mean = cl_dem[row,col]
            if cl[0,row,col] == cl_p["nodata"] or np.isnan(dem_cell_mean):
                continue
            row_min, row_max = (max(row-window_radius,0),min(row+window_radius,cl_p["height"]))
            cl_wind_all = cl[:,row_min:row_max+1,col_min:col_max+1]
            cl_dem_wind = cl_dem[row_min:row_max+1,col_min:col_max+1]

            # moving window size
            added_wind = 0
            max_add_win = max_window_radius-window_radius
            if moving_wind_size:
                col_min_bef, col_max_bef = col_min, col_max
                while np.nanstd(cl_dem_wind) < min_dem_std and added_wind<(max_add_win):
                    row_min= np.max([0, row_min- 1])
                    row_max= np.min([cl_p["height"], row_max +1])
                    col_min =np.max([0, col_min-1])
                    col_max =np.min([cl_p["width"], col_max+1])
                    added_wind +=1
                    cl_dem_wind = cl_dem[row_min:row_max+1,col_min:col_max+1]
                    cl_wind_all = cl[:,row_min:row_max+1,col_min:col_max+1]
                if added_wind >0:
                    col_min, col_max = col_min_bef, col_max_bef

            used_wind_size[row,col] = window_radius + added_wind

            for i_para in range(cl_p["count"]):
                # create linear regression
                cl_cell = cl[i_para,row,col]
                cl_wind = cl_wind_all[i_para,:,:]
                if fix_intercept:
                    regr_wind = LinearRegression(fit_intercept=False)
                else:
                    regr_wind = LinearRegression()
                mask_nas = (~np.isnan(cl_dem_wind))&(cl_wind!=cl_p["nodata"])
                regr_wind.fit(
                    cl_dem_wind[mask_nas].reshape(-1, 1) - dem_cell_mean,
                    cl_wind[mask_nas].reshape(-1, 1).astype(int)-cl_cell)

                # get boundary coordinates of actual cl cell
                cl_cell_bds_rowcols = rowcols_bounds_raw.copy()
                cl_cell_bds_rowcols[0] = cl_cell_bds_rowcols[0]+row
                cl_cell_bds_rowcols[1] = cl_cell_bds_rowcols[1]+col
                cl_cell_bds_xys = np.array(
                    rio.transform.xy(cl_p["transform"],*cl_cell_bds_rowcols))\
                    .reshape(2,-1)
                dem_cell_bds_xys = tr_cl_to_dem.transform(*cl_cell_bds_xys)

                # get dem data
                cell_poly =[Polygon(list(zip(*dem_cell_bds_xys)))]
                try:
                    cell_wind = rio.mask.geometry_window(
                        dem_fine_src, cell_poly)
                except rio.errors.WindowError:
                    break
                cell_mask = rio.mask.geometry_mask(
                    cell_poly,
                    transform=dem_fine_src.window_transform(cell_wind),
                    invert=False,
                    out_shape=(int(cell_wind.height), int(cell_wind.width)),
                    all_touched=True)
                cell_wind_slice = cell_wind.toslices()
                dem_cell = dem_fine[cell_wind_slice].copy()
                dem_cell[cell_mask]=np.nan

                # predict new climate values
                dem_cell_na_mask = np.isnan(dem_cell)
                if np.any(~dem_cell_na_mask):
                    cl_cell_pred = regr_wind\
                        .predict(dem_cell[~dem_cell_na_mask].reshape(-1,1)-dem_cell_mean)\
                        .flatten()+cl_cell
                    if scale_mean:
                        cl_cell_pred = np.round(cl_cell_pred*cl_cell/np.mean(cl_cell_pred))\
                            .astype(int)
                    cl_cell_pred = np.round(cl_cell_pred).astype(int)

                    # # replace values in cl_new
                    cl_new_cell = cl_new[i_para,cell_wind_slice[0], cell_wind_slice[1]].copy()
                    cl_new_cell = cl_new_cell.flatten()
                    cl_new_cell[~dem_cell_na_mask.flatten()] = cl_cell_pred
                    cl_new_cell = cl_new_cell.reshape(dem_cell.shape)
                    cl_new[i_para,cell_wind_slice[0], cell_wind_slice[1]] = cl_new_cell

                    quot_of_mean[i_para,row,col] = np.mean(cl_cell_pred)/cl_cell*100
                    cl_dem_std[row,col] = np.nanstd(cl_dem_wind)
    return cl_new, quot_of_mean, cl_dem_std, used_wind_size

def parquet_to_geopackage(fps_pqt:list):
    """Convert a list of parquet files to geopackages with gdal.

    Uses ogr2ogr to convert the parquet files to geopackages.

    Parameters
    ----------
    fps_pqt : list of path-like objects
        The list of parquet files to convert.
    """
    # check input
    if not isinstance(fps_pqt,list):
        fps_pqt = [fps_pqt]
    for i, fp in enumerate(fps_pqt):
        if isinstance(fp, str):
            fps_pqt[i] = Path(fp)
        if fp.suffix != ".parquet":
            raise ValueError("The filepaths must be parquet files.")

    # load libraries, ass osgeo only needed for this function
    import osgeo

    # check if ogr2ogr is available
    if isinstance(osgeo.gdal_version, tuple):
        gdal_version = version.parse(str(osgeo.gdal_version)[1:-1].replace(", ", "."))
    else:
        gdal_version = version.parse(osgeo.ver_str(osgeo.gdal_version))

    if gdal_version >= version.parse("3.5.0"):
        fp_ogr2ogr = Path(
            subprocess.run(["where", "ogr2ogr"], stdout=subprocess.PIPE
                        ).stdout.decode().replace("\r\n", ""))
        if not fp_ogr2ogr.exists():
            Path(osgeo.__path__[0]).joinpath("ogr2ogr.exe")
        use_gdal = True
    else:
        print("GDAL version must be >= 3.5.0 to convert parquets quickly")
        use_gdal = False

    # get processes and filepaths
    procs = {}
    manually = {}
    fps = {}
    for fp_pqt in fps_pqt:
        fp_gpkg = fp_pqt.parent.joinpath(fp_pqt.stem+".gpkg")
        fps[fp_pqt] = fp_gpkg
        if use_gdal:
            procs.update({
                fp_pqt:subprocess.Popen(
                        [fp_ogr2ogr.as_posix(),
                            "-f", 'GPKG',
                            '-gt', '65536',
                            fp_gpkg.as_posix(),
                            fp_pqt.as_posix(),
                            fp_pqt.stem],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)})
        else:
            manually.update({fp_pqt:fp_gpkg})

    # create geopackages
    while len(procs) > 0 or len(manually) > 0:
        drop_keys = []
        for key, proc in procs.items():
            if proc.poll() is not None:
                if proc.returncode != 0:
                    print("There was an error executing: " + " ".join(proc.args))
                    print(proc.stderr.read())
                    print("\nTherefor the geopackage is written manualy")
                    manually.update({key:fps_pqt[key]})
                drop_keys.append(key)
        for key in drop_keys:
            procs.pop(key)

        for fp_pqt in list(manually.keys()):
            manually.pop(fp_pqt)
            fp_gpkg = fps[fp_pqt]
            gdf = gpd.read_parquet(fp_pqt)
            gdf.to_file(fp_gpkg, driver="GPKG", layer=fp_gpkg.stem, chunk_size=65536)
        time.sleep(5)

def gdf_to_geopackage(gdfs, fps, remove_parquet=True, **kwargs):
    """Write a list of GeoDataFrames to geopackages.

    First writes the dataframes to parquets and then converts them to geopackages.
    Uses GDAL to write GeoPackages, as this is way faster.

    Parameters
    ----------
    gdfs : list of gpd.GeoDataFrame or gpd.GeoDataFrame
        A list of GeoDataFrames to write to parquet files.
        Can also be a single GeoDataFrame.
    fps : list of path-like objects or path-like object
        The filepaths of the geopackage files.
        Can also be a single filepath.
    remove_parquet : bool, optional
        Should the parquet files be removed after conversion.
        The default is True.
    """
    # check input
    if not isinstance(gdfs, list):
        gdfs = [gdfs]
    if not isinstance(fps, list):
        fps = [fps]
    if len(gdfs) != len(fps):
        raise ValueError("The number of GeoDataFrames and filepaths must be equal.")

    # check filepaths
    for i, fp in enumerate(fps):
        if isinstance(fp, str):
            fps[i] = Path(fp)
        if fp.suffix != ".gpkg":
            raise ValueError("The filepaths must be geopackage files.")

    # check if GeoDataframes are given
    for i, gdf in enumerate(gdfs):
        if type(gdf) != gpd.GeoDataFrame:
            raise ValueError("The input must be a GeoDataFrame or a list of GeoDataFrames.")

    # create parquets
    fps_pqt = []
    for gdf, fp in zip(gdfs, fps):
        fp_pqt = fp.parent.joinpath(fp.stem+".parquet")
        fps_pqt.append(fp_pqt)
        gdf.to_parquet(fp_pqt, **kwargs)

    print("Parquets created.")

    # create geopackages
    parquet_to_geopackage(fps_pqt)
    if remove_parquet:
        for fp_pqt in fps_pqt:
            fp_pqt.unlink()