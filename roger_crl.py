#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of functions to set, control and start RoGeR."""

__author__ = "Max Schmit"
__copyright__ = "Copyright 2021, Max Schmit"

# libraries
try:
    from import_DWD import get_dwd_data, get_dwd_meta
except ImportError:
    from .import_DWD import get_dwd_data, get_dwd_meta

import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import sys
#import progressbar
from progressbar import ProgressBar, progressbar
from progressbar.widgets import (Percentage, SimpleProgress, Bar, Timer,
                                 DynamicMessage, RotatingMarker, ETA)
from subprocess import Popen, TimeoutExpired
from multiprocessing import cpu_count
from socket import gethostname
import time
from zipfile import ZipFile
import re
import warnings
from multiprocessing import Pool
from packaging import version as pkgv


ROGER_CF_COLUMNS = {
    "index": "No.",
    "Landnutzung": "Landnutzung", 
    "Versiegelung": "Versiegelung (%)", 
    "Bodentiefe": "Bodentiefe (cm)", 
    "GWFA": "GWFA (cm)", 
    "MPD_v": "MPD_v (1/m2)",
    "MPL_v": "MPL_v (cm)", 
    "MPD_h": "MPD_h (1/m2)", 
    "TP": "TP (mm/h)", 
    "SL": "SL (%)", 
    "nFK": "nFK (%)", 
    "LK": "LK  (Vol.%)", 
    "PWP": "PWP (Vol.%)", 
    "Theta": "Theta (Vol.%)", 
    "KS": "KS (mm/h)",
    "Muldenspeicher": "Muldenspeicher (mm)", 
    "Baueme": "Baueme", 
    "Urban": "Urban", 
    "N_Wichtung": "N Wichtung (%)", 
    "N Wichtung Sommer": "N Wichtung Sommer (%)", 
    "T_Diff": "T Zuschlag (°C)",
    "ET_Wichtung": "ET Wichtung (%)",
    "N Wichtung Winter": "N Wichtung Winter (%)",
    "solar radiation factor": "SRF (%)",
    "SRF" : "SRF (%)", 
    "Flaechenanteil": "Flaechenanteil (%)",
    "ZA Tiefe": "ZA Tiefe (cm)",
}
this_dir, _ = os.path.split(__file__)
# privat functions
##################


def _fill_na(df, para, column="all"):
    """
    Fill NaN's in a df.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to correct.
    para : str
        The parameter that gets filled. Can be "N", "ET" or "T".
    column : str or list of str, optional
        The columns that get filled.
        The default is "all".

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    # set columns
    if column == "all":
        column = df.columns.to_list()

    # set na flag
    if type(column) == str:
        df["filled_na"] = df[column].isna()
    else:
        df["filled_na"] = df[column].isna().sum(axis=1).transform(bool)

    # fill nas
    if para in ["T", "ET"]:
        df[column] = df[column].interpolate(methode="linear")
    elif para == "N":
        df[column] = df[column].fillna(0)

    return df


def _fill_timeholes(df, timestep):
    """
    Check for holes in a DataFrame and correct them to get a continuous datetime series.

    Parameters
    ----------
    df : pandas.DataFrame
        Needs at least a date column named "Datum" or "MESS_DATUM".
    timestep : str or pandas.Timedelta
        The Timestep between the columns, to use for a continuous time serie.
        If string it needs to be in a format readable by pd.Timedelta.

    Raises
    ------
    ValueError
        If the input was not in a correct format.

    Returns
    -------
    new_df : pd.DataFrame
        The corrected DataFrame with a continuous datetime Serie.
        It also has a new variable new_df.filled_date_holes with the number of
        holes that got filled.

    """
    # get date column name
    if "Datum" in df.columns:
        col_date = "Datum"
    elif "MESS_DATUM" in df.columns:
        col_date = "MESS_DATUM"
    else:
        raise ValueError("Could not find a Date column. " +
                         "No column was named 'Datum' or 'MESS_DATUM'")

    # check if column is in a date format
    try:
        df[col_date] = pd.to_datetime(df[col_date])
    except:
        raise ValueError("The Date column " + col_date +
                         "is not in a valid datetime format." +
                         "\nThe column is in the format: " +
                         str(df[col_date].dtype))

    # check timestep format
    if type(timestep) == str:
        timestep = pd.Timedelta(timestep)
    elif type(timestep) != pd.Timedelta:
        raise ValueError("The given timestep is not in a valid format." +
                         "\nvalid formats are: 'str' or 'pd.Timedelta'" +
                         "\nThe format of timestep is: " + type(timestep))

    # correct timesteps in df
    holes = df[col_date].diff() > timestep
    if sum(holes) > 0:
        new_df = pd.DataFrame()
        new_date = pd.date_range(df[col_date][0],
                                 df[col_date][len(df)-1],
                                 freq=timestep)
        new_df[col_date] = new_date
        new_df = pd.merge(new_df, df, how="outer", on=col_date)
    else:
        new_df = df

    bool_new = pd.merge(df[col_date], new_df[col_date],
                        how="outer", indicator=True)._merge == "right_only"
    new_df["filled_date_holes"] = False
    new_df.loc[bool_new, "filled_date_holes"] = True

    return new_df


def _split_date(dates):
    """
    Split datetime into parts.

    Parameters
    ----------
    dates : datetime.dateime or pandas.Timestamp or
            list or pandas.DataFrame of datetime.datetime or pandas.Timestamp
        The datetime's to be split.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with 5 columns (Jahr, Monat, Tag, Stunde, Minute).

    """
    # if dates is not a list make it a list
    if type(dates) == datetime or type(dates) == pd.Timestamp:
        dates = [dates]
        index = [0]
    elif type(dates) == pd.Series:
        index = dates.index
        dates = dates.to_list()
    else:
        index = range(0, len(dates))

    # check if date is datetime or Timestamp:
    if not (type(dates[0]) == pd.Timestamp or
            type(dates[0]) == datetime):
        raise ValueError("Error: The given date is not in a datetime or " +
                         "Timestamp format.")
        return None

    # extract the parts
    years = [dates[0].year]
    months = [dates[0].month]
    days = [dates[0].day]
    hours = [dates[0].hour]
    minutes = [dates[0].minute]
    for date in dates[1:]:
        years.append(date.year)
        months.append(date.month)
        days.append(date.day)
        hours.append(date.hour)
        minutes.append(date.minute)

    return pd.DataFrame({"Jahr": years, "Monat": months, "Tag": days,
                         "Stunde": hours, "Minute": minutes},
                        dtype=int,
                        index=index)


# functions
###########

def get_cf_df_template(version="2_92_1", with_unit=False):
    """
    Get an empty DataFrame as a template for the soil configuration for RoGeR.

    Parameters
    ----------
    version : str, optional
        The RoGeR version Number for which to get the template.
        Add "+A" to add an Area column.
        Add "+SRF" to add a solar radiation column.
        Add "+ZAT" to add a ZA Tiefe column.
        E.g. "2_92_1" for 1D_WBM_roger_2_92_mx_schmit_1.exe
        The default is "2_92_1".
    with_unit : bool, optional
        Should the columns names be with their unit?
        The default is False.


    Returns
    -------
    pandas.DataFrame
        An empty DataFrame with the needed columns for the soil configuration
        for RoGeR.

    """
    # get the base columns
    columns = ["Landnutzung", "Versiegelung", "Bodentiefe", "GWFA", "MPD_v",
               "MPL_v", "MPD_h", "TP", "SL", "nFK", "LK", "PWP", "Theta", "KS",
               "Muldenspeicher", "Baueme", "Urban", "N_Wichtung", "T_Diff",
               "ET_Wichtung"]

    # check for area
    do_area = re.search("\+[Aa]", version)
    if do_area:
        version = re.sub(do_area.re, "", version)

    # check for do_srf (solar radiation factor)
    do_srf = re.search("\+((SRF)|(srf))", version)
    if do_srf:
        version = re.sub(do_srf.re, "", version)

    # check for do_srf (solar radiation factor)
    do_zat = re.search("\+((ZAT)|(zat)|(GRUND_ZA))", version)
    if do_zat:
        version = re.sub(do_zat.re, "", version)

    # parse the version
    version = pkgv.parse(version.replace("_", "."))

    # check for winter precipitation factor
    if version >= pkgv.parse("2.93.3"):
        columns[columns.index("N_Wichtung")] = "N Wichtung Sommer"
        columns.append("N Wichtung Winter")

    # add exposition factor
    if do_srf:
        columns.append("SRF")
    
    # add ZA Tiefe
    if do_zat:
        columns.append("ZA Tiefe")
    
    # add the area portion if wanted
    if do_area:
        columns.append("Flaechenanteil")

    # create the table
    table = pd.DataFrame(columns=columns)
    table.index.name = "No"
    if with_unit:
        table.rename(ROGER_CF_COLUMNS, axis=1, inplace=True)

    return table


def create_cf(cf_path, output_dir, weather_dir, cf_table,
              dir_rel_to=None, rog_ver="2_92_1", create_weather_dir=True):
    """
    Create the control file for RoGeR.

    Parameters
    ----------
    cf_path : str or Path
        Path with filename for the control file to be stored to.
    output_dir : str or Path
        Path for the directory where the results of the RoGeR simulation
        should get stored.
    weather_dir : str or Path
        Path for the directory where the weather data are stored in.
    cf_table : pandas.DataFrame
        The DataFrame with all the soil parameters.
        To get the correct format use get_cf_df_template().
    dir_rel_to: str or Path
        Path for the directory where the roger executable will be.
        The paths in the control files will be relative to this directory.
        If None the paths will be stored as absolute paths.
        The default is None.
    rog_ver: str, optional
        The version number of the roger executable.
        E.g. "2_92_1" for 1D_WBM_roger_2_92_mx_schmit_1.exe
        The default is "2_92_1".
    create_weather_dir: bool, optional
        Should the weather folder get created if it is not existing?
        The default is True.

    Raises
    ------
    ValueError
        If the given cf_path was no valid path for a csv file.
    NameError
        If the given cf table was not in a correct format
        for a specific version.

    Returns
    -------
    None.

    """
    # test directories and create them if not existant
    output_dir = Path(output_dir)
    weather_dir = Path(weather_dir)
    cf_path = Path(cf_path)
    if dir_rel_to is not None:
        dir_rel_to = Path(dir_rel_to)
    if not output_dir.is_dir():
        output_dir.mkdir()
    if create_weather_dir and not weather_dir.is_dir():
        weather_dir.mkdir()
    if not cf_path.suffix == ".csv":
        raise ValueError("The given cf_path has no .csv ending.")
    if not cf_path.parent.is_dir():
        cf_path.parent.mkdir()

    # get roger cf template
    cf_tmplt = get_cf_df_template(version=rog_ver)

    problem_cols = []
    for col in cf_table.columns:
        if (col not in cf_tmplt.columns and 
                col not in cf_tmplt.rename(ROGER_CF_COLUMNS, axis=1).columns):
            problem_cols.append(col)
    if len(problem_cols)>0:
        raise NameError(
            "The given column(s) {cols} are not valid for this RoGeR Version ({rog_ver}))".format(
                cols=', '.join(problem_cols), rog_ver=rog_ver))


    # get the header template
    tmplt_path = os.path.join(
        this_dir, "data", "roger_cf_template_header_20Zeilen.csv")
    with open(tmplt_path, "r", encoding="UTF-8") as f:
        l_header = f.read()

    # add columns to header
    add_cols = len(cf_tmplt.columns) - 20
    l_header = "\n".join(
        [l + add_cols*";" for l in l_header.split("\n")])

    # make paths absolute or relative and add them to header
    if dir_rel_to is None:
        output_dir = output_dir.resolve()
        weather_dir = weather_dir.resolve()
    else:
        output_dir = output_dir.relative_to(dir_rel_to)
        weather_dir = weather_dir.relative_to(dir_rel_to)

    l_header = l_header.replace("_output_dir_", str(output_dir) + "/")
    l_header = l_header.replace("_weather_dir_", str(weather_dir) + "/")

    # set number of areas
    l_header = l_header.replace("_num_areas_", str(len(cf_table)))

    # set table as string
    bool_index = True
    if len(cf_table.columns) > len(cf_tmplt.columns):
        bool_index = False

    # order and rename columns
    col_order = [key for key in ROGER_CF_COLUMNS.keys() 
                     if key in cf_table.columns]
    cf_table = cf_table[col_order]
    cf_table.rename(ROGER_CF_COLUMNS, axis=1, inplace=True)

    # write table lines
    if pkgv.parse(pd.__version__) > pkgv.parse("1.5.0"):
        to_csv_kwargs = dict(lineterminator="\n")
    else:
        to_csv_kwargs = dict(line_terminator="\n")
    l_table = cf_table.to_csv(
        sep=";", index=bool_index, encoding="UTF-8", **to_csv_kwargs)

    # write to file
    with open(cf_path, "w", encoding="UTF-8") as f:
        f.write(l_header + "\n" + l_table)


def create_weather_tables(n, n_dt, t, t_dt, et, et_dt,
                          folder, position=None, et_r_r0=1, name="",
                          timespan=None):
    """
    Create all the weather tables for RoGeR.

    Reduces the weather serie to overlapping Timespan.
    Writes the data in .txt files in folder.
    The files provided need to be without NAs or holes in timesteps,
    as create_weather_tables is not getting to correct them.

    Parameters
    ----------
    n : pandas.Serie or list of int or float
        The precipitation data.
    n_dt : pandas.Serie or list of datetime.datetime or pandas.Timestamp
        The DateTime for the corresponding precipitation data.
    t : pandas.Serie or list of int or float
        The temperature data. Needs to be in daily resolution.
    t_dt : pandas.Serie or list of datetime.datetime or pandas.Timestamp
        The DateTime for the corresponding temperature data.
    et : pandas.Serie or list of int or float
        The evapotranspiration data. Needs to be in daily resolution.
    et_dt : pandas.Serie or list of datetime.datetime or pandas.Timestamp
        The DateTime for the corresponding evapotranspiration data.
    folder : pathlib.Path or str
        The folder path where the tables should get stored.
        !!!This folder will get emptied!!!
    position : geometry.Point, optional
        The position of the weather station
        from which the weather data is from.
        If None no Lat and Lon in the header will be 0.
        The default is None.
    et_r_r0 : pandas.Serie or list of (int or float) or int
        The correction factor for the Evapotranpiration data.
        If only one value it will get used for all the et data.
        The default is 1.
    name : str, optional
        A name of the station or something else to name the files.
        It will get added as suffix to the files.
        The default is "".
    timespan : int or None, optional
        The timespan in years - number of years - to get precipitation and
        temperature data from DWD server.
        If timespan is None the maximum amount of years will be taken.
        The default is None.

    Returns
    -------
    None.

    """
    # test if folder exists and is empty
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir()

    for f in folder.glob("*"):
        f.unlink()

    # create filepaths
    if name != "":
        name_suffix = "_" + name

    file_n = Path(folder, "N" + name_suffix + ".txt")
    file_et = Path(folder, "ET" + name_suffix + ".txt")
    file_t = Path(folder, "Ta" + name_suffix + ".txt")

    # create headers and files
    if position == None:
        x = str(0)
        y = str(0)
    else:
        x = str(position.x)
        y = str(position.y)

    header_n = ("Name: " + name + "\t" * 5 + "\n" +
                "Lat: " + y + "   ,Lon: " + x + "\t" * 5 + "\n")
    header_et = ("Name: " + name + "\t" * 4 + "\n" +
                 "Lat: " + y + "   ,Lon: " + x + "\t" * 4 + "\n")
    header_t = ("Name: " + name + "\t" * 3 + "\n" +
                "Lat: " + y + "   ,Lon: " + x + "\t" * 3 + "\n")

    with open(file_n, "w") as f:
        f.write(header_n)
    with open(file_et, "w") as f:
        f.write(header_et)
    with open(file_t, "w") as f:
        f.write(header_t)

    # convert lists to Series
    if type(et) == list:
        et = pd.Series(et)
    if type(et_r_r0) == list:
        et_r_r0 = pd.Series(et_r_r0)
    if type(et_dt) == list:
        et_dt = pd.Series(et_dt)
    if type(n) == list:
        n = pd.Series(n)
    if type(n_dt) == list:
        n_dt = pd.Series(n_dt)
    if type(t) == list:
        t = pd.Series(t)
    if type(t_dt) == list:
        t_dt = pd.Series(t_dt)

    # get overlapping timespan and delete the rest
    min_max_cor = pd.DateOffset(hours=23, minutes=55)
    min_dt = max(et_dt.min(), t_dt.min(), n_dt.min())
    max_dt = min(et_dt.max(), t_dt.max(), n_dt.max()) + min_max_cor

    if timespan:
        timeoffset = pd.DateOffset(years=timespan)
        if min_dt < (max_dt - timeoffset):
            min_dt = max_dt - timeoffset

    n = n.loc[n_dt.between(min_dt, max_dt)]
    n_dt = n_dt.loc[n_dt.between(min_dt, max_dt)]
    t = t.loc[t_dt.between(min_dt, max_dt)]
    t_dt = t_dt.loc[t_dt.between(min_dt, max_dt)]
    et = et.loc[et_dt.between(min_dt, max_dt)]

    if type(et_r_r0) == int:
        et_r_r0 = [et_r_r0] * len(et)
    else:
        et_r_r0 = et_r_r0[et_dt.between(min_dt, max_dt)]

    et_dt = et_dt.loc[et_dt.between(min_dt, max_dt)]

    # create precipitation table
    n_out = _split_date(n_dt)
    n = pd.Series(n, name="N", index=n_out.index)
    n_out = pd.concat([n_out, n], axis=1)
    n_out.to_csv(file_n, sep="\t", decimal=".", index=False, mode="a")

    # create temperature table
    t_out = _split_date(t_dt).iloc[:, 0:3]
    t = pd.Series(t, name="Ta", index=t_out.index)
    t_out = pd.concat([t_out, t], axis=1)
    t_out.to_csv(file_t, sep="\t", decimal=".", index=False, mode="a")

    # create evapotranspiration table
    et_out = _split_date(et_dt).iloc[:, 0:3]
    et = pd.Series(et, name="ET", index=et_out.index)
    et_r_r0 = pd.Series(et_r_r0, name="R/R0", index=et_out.index)
    et_out = pd.concat([et_out, et, et_r_r0], axis=1)
    et_out.to_csv(file_et, sep="\t", decimal=".", index=False, mode="a")


def get_create_weather(station_id, folder,
                       timeres="10_minutes", timespan=None,
                       name_suffix="", make_10min=True):
    """
    Get the weather data from dwd and create tables from that for RoGeR.

    Will fill NAs and timestep holes.
    Creates a meta file in the folder to document
    how many holes and nas got filled.

    Parameters
    ----------
    station_id : int or str
        The DWD Station id.
    folder : str or pathlib.Path
        The folder path where the weather data should get stored.
    timeres : str, optional
        The time resolution to get precipitation data from DWD server.
        Afterwards the downloaded data will get prolonged to 10 minutes,
        if timeres>10 minutes.
        Possible Values are: "10_minutes", "daily", "hourly"
        The default is "10_minutes".
    timespan : int, optional
        The timespan in years - number of years - to get precipitation and
        temperature data from DWD server.
        If timespan is None the maximum amoount of years will be taken.
        The default is None.
    name_suffix : str, optional
        a suffix to add to the file names that get created.
        The default is "".
    make_10min : bool, optional
        If True: If the timeres is higher than 10 minutes,
        it will expand those to 10_minutes datasets.
        The default is True.

    Returns
    -------
    A folder with 3 timeseries for ET, T and N each and
    a meta file with additional infos.

    """
    # dict of what data to take
    dic = {"10_minutes": {"folder": "/climate_environment/CDC/observations_germany/climate/10_minutes/precipitation/historical/",
                          "n_col": "RWS_10",
                          "timestep": "10 m"},
           "hourly": {"folder": "/climate_environment/CDC/observations_germany/climate/hourly/precipitation/historical/",
                      "n_col": "R1",
                      "timestep": "1 h",
                      "hour_ofset": 0, "div": 6},
           "daily": {"folder": "/climate_environment/CDC/observations_germany/climate/daily/kl/historical/",
                     "n_col": "RSK",
                     "timestep": "1 d",
                     "hour_ofset": 23, "div": 144}}

    # import dwd data from cache if disponible
    global last_import_cache
    if "last_import_cache" in globals():
        if last_import_cache["station_id"] == station_id:
            t_df = last_import_cache["t_df"]
            et_df = last_import_cache["et_df"]
            meta = last_import_cache["meta"]
            if last_import_cache["timeres"] == timeres:
                n_df = last_import_cache["n_df"]

    # get all the data from dwd if not yet loaded from cache
    if "et_df" not in locals():
        et_df = get_dwd_data(station_id,
                             "/climate_environment/CDC/derived_germany/" +
                             "soil/daily/historical/")
        et_df = _fill_timeholes(df=et_df, timestep="1 d")
        et_df = _fill_na(df=et_df, para="ET", column="VPGB")

    if "t_df" not in locals():
        t_df = get_dwd_data(station_id, dic["daily"]["folder"])
        t_df = _fill_timeholes(df=t_df, timestep="1 d")
        t_df = _fill_na(df=t_df, para="T", column=["TMK"])

    if "meta" not in locals():
        meta = get_dwd_meta(dic["daily"]["folder"])

    if "n_df" not in locals():
        if timeres == "daily":
            n_df = t_df
        else:
            n_df = get_dwd_data(station_id=station_id,
                                ftp_folder=dic[timeres]["folder"])
        n_df = _fill_timeholes(df=n_df, timestep=dic[timeres]["timestep"])
        n_df = _fill_na(df=n_df, para="N", column=dic[timeres]["n_col"])

    # store data in cache
    last_import_cache = {"station_id": station_id, "timeres": timeres,
                         "n_df": n_df, "t_df": t_df, "et_df": et_df,
                         "meta": meta}

    # expand n timeserie if timeres > 10_minutes
    if timeres != "10_minutes" and make_10min:
        if timeres == "daily":
            # because, daily is calculated from 5.50 UTC to 5:50 UTC
            ts_off = pd.DateOffset(hours=3, minutes=50)
        else:
            ts_off = pd.DateOffset(hours=0, minutes=0)

        ts_new = pd.date_range(start=n_df["MESS_DATUM"].min(),
                               end=(n_df["MESS_DATUM"].max() +
                                    pd.DateOffset(minutes=50,
                                                  hours=dic[timeres]["hour_ofset"])),
                               freq="10 min")

        n_new = (ts_new + ts_off).to_frame(index=False, name="MESS_DATUM")
        n_new["MESS_DATUM_old"] = ts_new.floor(dic[timeres]["timestep"])
        n_new = pd.merge(n_new, n_df,
                         left_on="MESS_DATUM_old", right_on="MESS_DATUM",
                         suffixes=["", "_y"])
        n_new.drop(["MESS_DATUM_y", "MESS_DATUM_old"],
                   axis=1, inplace=True)
        n_new[dic[timeres]["n_col"]] = (n_new[dic[timeres]["n_col"]] /
                                        dic[timeres]["div"])
        n_df = n_new

    # get overlapping timespan and delete the rest
    min_max_cor = pd.DateOffset(hours=23, minutes=55)
    min_dt = max(et_df.Datum.min(),
                 t_df.MESS_DATUM.min(),
                 n_df.MESS_DATUM.min())
    max_dt = min(et_df.Datum.max(),
                 t_df.MESS_DATUM.max(),
                 n_df.MESS_DATUM.max()) + min_max_cor

    timeoffset = pd.DateOffset(years=timespan)
    if min_dt < (max_dt - timeoffset):
        min_dt = max_dt - timeoffset

    n_df = n_df.loc[n_df.MESS_DATUM.between(min_dt, max_dt)]
    t_df = t_df.loc[t_df.MESS_DATUM.between(min_dt, max_dt)]
    et_df = et_df.loc[et_df.Datum.between(min_dt, max_dt)]

    # create the tables
    et_r_r0 = pd.Series([1] * len(et_df), index=et_df.index)  # still unclear ###########################
    if name_suffix =="":
        name = str(station_id)
    else:
        name = str(station_id) + "_" + name_suffix

    create_weather_tables(n=n_df[dic[timeres]["n_col"]], n_dt=n_df.MESS_DATUM,
                          t=t_df.TMK, t_dt=t_df.MESS_DATUM,
                          et=et_df.VPGB, et_r_r0=et_r_r0, et_dt=et_df.Datum,
                          folder=folder,
                          position=meta.loc[int(station_id)].geometry,
                          name=name,
                          timespan=timespan)

    # write meta file weather with na filled
    with open(folder.joinpath("meta.txt"), "w") as meta_f:
        meta_f.write("table\tnumber_of_na_filled\tnumber_of_timeholes_filled\n")
        meta_f.write("N\t" + str(sum(n_df["filled_na"])) + "\t" +
                     str(sum(n_df["filled_date_holes"])) + "\n")
        meta_f.write("T\t" + str(sum(t_df["filled_na"])) + "\t" +
                     str(sum(t_df["filled_date_holes"])) + "\n")
        meta_f.write("ET\t" + str(sum(et_df["filled_na"])) + "\t" +
                     str(sum(et_df["filled_date_holes"])) + "\n")


def roger_run(roger_exe, cf_files, unuse_cpus=1): # copy from exe
    """
    Start RoGeR simulations in multiprocessing-mode.

    Uses all the ressources from a cpu, by starting several roger simulations.
    Best is to leave on cpu free for other stuff.
    If a run takes more than 40 minutes it is stoped and restarted.

    Parameters
    ----------
    roger_exe : str or pathlike
        Path to the RoGeR executable.
    cf_files : list of (str or Path) or str or Path
        Paths to the configuration files for RoGeR.
    unuse_cpus: int, optional
        The amount of CPU kernels that should not get used for the simulations.
        The default is 1.

    Returns
    -------
    failed_runs: list
        A list of the runs that had problems.

    """
    # check paths
    if type(roger_exe) != Path:
        roger_exe = Path(roger_exe)
    if not roger_exe.is_file():
        raise TypeError("The filepath to the roger distribution" +
                        " is not a filepath: \n" + str(roger_exe))

    if type(cf_files) == list:
        if type(cf_files[0]) != Path:
            cf_files = [Path(file) for file in cf_files]
    else:
        cf_files = [Path(cf_files)]

    # create log file
    with open(cf_files[0], "r") as f:
        output_dir = Path(f.readlines()[3].split(";")[1]).parent
        if not output_dir.is_dir():
            output_dir = roger_exe.parent.joinpath(output_dir)
        if not output_dir.is_dir():
            raise ValueError("The output folder could not get found, " +
                             "to write a log file there.")
    log = open(output_dir.joinpath("log_runs_roger_" + gethostname() + ".txt"),
               "wb")

    # create commands
    cmds = []
    for cf_file in cf_files:
        cmds.append(roger_exe.name + " " +
                    str(cf_file.relative_to(roger_exe.parent)))

    # initiate loop parameters
    bar = ProgressBar(max_value=len(cf_files), redirect_stdout=True,
                      widgets=[RotatingMarker(), ' ',
                               Percentage(), ' ',
                               SimpleProgress(format=("('%(value_s)s/" +
                                                      "%(max_value_s)s')")),
                               ' ', Bar(min_width=80), ' ',
                               Timer(format='%(elapsed)s'), ' | ',
                               ETA(), ' ',
                               DynamicMessage("run",
                                              format=("last: " +
                                                      "{formatted_value}"),
                                              precision=7)],
                      variables={"run": "None"},
                      term_width=80,
                      line_breaks=False
                      )
    bar.update(0)

    os.chdir(roger_exe.parent)
    max_proc = cpu_count() - unuse_cpus  # one cpu-core is free for other stuff
    processes = []
    failed_cmds = []
    start_times = []
    restart_cmds = []

    # loop for multiprocessing roger runs
    while True:
        bar.update(bar.value) # to rotate the rotatemarker in the progressbar
        # start a new process
        while cmds and (len(processes) < max_proc):
            task = cmds.pop()
            processes.append(Popen(task, stdout=-1, stderr=-1))
            start_times.append(time.time())
            time.sleep(3)

        # check for finished processes
        for p, st in zip(processes, start_times):
            # check for finished processes?
            if p.poll() is not None:
                try:
                    stdout, stderr = p.communicate(timeout=3)
                    if stdout is None: stdout = b""
                    if stderr is None: stderr = b""
                except TimeoutExpired:
                    stdout, stderr = (b"", b"")

                if p.returncode == 0:
                    # success
                    log.write(b"\n" + b"#" * 72 + b"\n#  run: " +
                              bytes(Path(p.args.split(" ")[1]).stem, "utf-8") +
                              b"\n" + b"#" * 72 +
                              b"\nfinish time:" +
                              bytes(str(datetime.now()), encoding="utf8") +
                              b"\n\n###### stdout: ######\n" + stdout +
                              b"\n###### stderr: ######\n" + stderr +
                              b"\n")
                    bar.variables["run"] = Path(p.args.split(" ")[1]).stem
                else:
                    # failed to execute
                    log.write(b"\n" + b"#" * 72 + b"\n#  run: " +
                              bytes(Path(p.args.split(" ")[1]).stem, "utf-8") +
                              b"\n" + b"#" * 72 +
                              b"\n ERROR: The run couldn't get simulated!" +
                              b"\n\n###### stdout: ######\n" + stdout +
                              b"\n###### stderr: ######\n" + stderr +
                              b"\n")
                    failed_cmds.append(p.args)

                p.terminate()
                processes.remove(p)
                start_times.remove(st)

                bar.update(bar.value+1)

            # process runs out of time ? more than 40 minutes
            elif (time.time() - st) > (60 * 40):
                try:
                    stdout, stderr = p.communicate(timeout=3)
                    if stdout is None: stdout = b""
                    if stderr is None: stderr = b""
                except TimeoutExpired:
                    stdout, stderr = (b"", b"")

                if p.args in restart_cmds:
                    log.write(b"\n" + b"#" * 72 + b"\n#  run: " +
                              bytes(Path(p.args.split(" ")[1]).stem, "utf-8") +
                              b"\n" + b"#" * 72 +
                              b"\nERROR: The control file did also run " +
                              b"out of time on the second try! " +
                              b"\n\tTherefor it is not retried.\n" +
                              b"\n\n###### stdout: ######\n" + stdout +
                              b"\n###### stderr: ######\n" + stderr +
                              b"\n")
                    failed_cmds.append(p.args)
                else:
                    log.write(b"\n" + b"#" * 72 + b"\n#  run: " +
                              bytes(Path(p.args.split(" ")[1]).stem, "utf-8") +
                              b"\n" + b"#" * 72 +
                              b"\nWARNING: The run got stoped and restarted, " +
                              b" because it seems to be blocked." +
                              b" Execution time > 40 Minutes." +
                              b"\n\n###### stdout: ######\n" + stdout +
                              b"\n###### stderr: ######\n" + stderr +
                              b"\n")
                    restart_cmds.append(p.args)
                    cmds.append(p.args)

                p.terminate()
                processes.remove(p)
                start_times.remove(st)

        # all processes done?
        if not processes and not cmds:
            break
        else:
            time.sleep(5)

    log.close()
    bar.finish()

    failed_runs = [Path(cmnd.split(" ")[1]).stem for cmnd in failed_cmds]

    return failed_runs

def guess_simulation_time(n_runs, runs_per_cf, years):
    """Guess the needed simulation time for a number of runs.

    Parameters
    ----------
    n_runs : int
        The number of control files to simulate
    runs_per_cf : int
        The average number of runs per control file.
    years : float or int
        The number of years that get simulated.
    """    
    # the data comes from the run_duration.xlsx file
    df_time = pd.DataFrame(
        zip(["fuhys001","fuhy1087","fuhy1030", "fuhys003-leaving 15 cores out", "bwUNICluster2.0 (80 cores)"], 
            ["00:25.858", "00:17.157", "00:19.124", "00:03.141", "00:01.382"]),
        columns=["PC","time/(10 years * 100 rows * 1 cf)"]).set_index("PC")
    df_time["time/(10 years * 100 rows * 1 cf)"] = df_time["time/(10 years * 100 rows * 1 cf)"]\
        .map(lambda x: pd.Timestamp("00:" + x) - pd.Timestamp("00:00:00"))
    df_time["estimated time for runs"] = df_time["time/(10 years * 100 rows * 1 cf)"] * n_runs/runs_per_cf * runs_per_cf/100 * years/10

    return df_time


# for testing of roger run

if False:
    file_dir = Path(os.getcwd())
    file_dir = Path("D:/Dokumenter/UNI/Master/Freiburg/2_Masterarbeit/Python/scripts/functions")
    roger_exe = file_dir.parents[2].joinpath("Modelle/RoGeR_WBM_1D_2-92_max/1D_WBM_roger_2_92__mx_schmit.exe")
    input_dir = roger_exe.parent.joinpath("runs/test_multiprocessing/input")

    cf_files = list(input_dir.glob("*.csv"))

    roger_run(roger_exe, cf_files)

    # empty output
    import shutil
    output_dir = input_dir.parent.joinpath("output")
    for d1 in output_dir.iterdir():
        if d1.is_dir():
            for d2 in d1.iterdir():
                shutil.rmtree(d2)


######################
# treat outputs
######################

def import_result_zip(zip_file, part_arcdir, with_input=True, columns="all", index_cols=["SIM_ID", "lanu_ID", "BF_ID"]):
    """
    Import a single result file from an opened Zipfile.

    Parameters
    ----------
    zip_file : zipfile.ZipFile
        The Outputs Zipfile.
    part_arcdir : pathlike Path
        The archive path to the output parts folder.
    with_input : bool, optional
        Should the Input parameters be in the output file?
        The default is True.
    columns: list of str or str, optional
        The columns to import.
        If "all", then all the columns get imported.
        The default is "all".
    index_cols: list of str or str, optional
        The name of the different parts of the id column, splited by "_".
        The default is ["SIM_ID", "lanu_ID", "BF_ID"].

    Raises
    ------
    KeyError
        If the providen columns do not exist.

    Returns
    -------
    results : pd.DataFrame
        The results of the single run.

    """
    try:
        results = pd.read_csv(
            zip_file.open(part_arcdir.joinpath(
                "Ereignisdaten/bilanz_save.csv").as_posix()),
            sep=";", skipinitialspace=True, index_col="No")
        bilanz_save_found = True
    except:
        bilanz_save_found=False

    if with_input or not bilanz_save_found:
        res_tot = pd.read_csv(
            zip_file.open(part_arcdir.joinpath(
                    "Ereignisdaten/bilanz_totalwerte.csv"
                    ).as_posix()),
            sep=";", skipinitialspace=True,
            index_col="No", encoding="ANSI", skiprows=1)
        if bilanz_save_found:
            results = res_tot.iloc[:, :21].join(results)
        else:
            results = res_tot

    # add station id, sim_id, lanu_id and Bf_id to the results
    if "STAT_ID" in results.columns:
        results["STAT_ID"] = int(part_arcdir.parent.stem)
    if len(index_cols)>0 and type(results.index[0]) == str:
        results[index_cols] = pd.DataFrame(
            results.index.str.split("_").tolist(),
            columns=index_cols, index=results.index
            ).astype(int)
        results.set_index(index_cols, inplace=True)
    else:
        results.index.name = index_cols[0]

    # select columns
    if columns != "all":
        results = results[columns]

    return results

def import_tot_zip(
        zip_fp, with_input=True, columns="all", 
        index_cols=["SIM_ID", "lanu_ID", "BF_ID"]):
    """
    Import all the results from one output-zipfile.

    Parameters
    ----------
    zip_fp : pathlike Path
        The filepath of the outputs Zip file.
    with_input : bool, optional
        Should the Input parameters be in the output file?
        The default is True.
    columns: list of str or str, optional
        The columns to import.
        If "all", then all the columns get imported.
        The default is "all".
    index_cols: list of str or str, optional
        The name of the different parts of the id column, splited by "_".
        The default is ["SIM_ID", "lanu_ID", "BF_ID"].

    Raises
    ------
    KeyError
        If the columns do not exist.

    Returns
    -------
    results : pd.DataFrame
        A DataFrame with all the results together.

    """
    with ZipFile(zip_fp, "r") as zf:
        # get list of files in Zip
        zf_list = pd.Series(zf.namelist()).apply(Path)
        zf_list_save = pd.Series(zf_list)[
            [True if re.search(r".*bilanz_save.csv", file.name) else False for file in zf_list]]
        if len(zf_list_save)==0:
            zf_list_save = pd.Series(zf_list)[
            [True if re.search(r".*bilanz_totalwerte.csv", file.name) else False for file in zf_list]]

        # create first entry in df if not existing
        results = import_result_zip(zip_file=zf,
                                    part_arcdir=zf_list_save.iloc[0].parents[1],
                                    with_input=with_input,
                                    columns=columns,
                                    index_cols=index_cols)

        # import every file and add to results
        for zf_file_save in progressbar(zf_list_save[1:], line_breaks=False):
            results_i = import_result_zip(zip_file=zf,
                                          part_arcdir=zf_file_save.parents[1],
                                          with_input=with_input,
                                          columns=columns,
                                          index_cols=index_cols)
            results = results.append(results_i)

    return results


# import functions for the monthly results
##########################################

def import_mon_para_zip_agg(zip_file, part_arcdir, para, skip_init_months):
    """
    Import the monthly results for one parameter of a single run in an opened Zipfile.

    Parameters
    ----------
    zip_file : zipfile.ZipFile
        The Outputs Zipfile.
    part_arcdir : pathlike Path
        The archive path to the output parts folder.
    para : str
        The Parameter to import.
        One of ["et", "N", "kap", "inf", "oa", "tp", "w_boden", "w_wurzel", "za"]
    skip_init_months : int
        The number of months(lines) to skip before calculating the sum.

    Returns
    -------
    results : pd.DataFrame
        The results of the single run.

    """
    fp = part_arcdir.joinpath("Ereignisdaten/monat_val_{}.csv".format(para))
    results = pd.read_csv(zip_file.open(fp.as_posix()),
                          sep=";", skipinitialspace=True)

    if para in ["w_boden", "w_wurzel"]:
        # count lines at the end with only zeros
        # normaly 24 lines, but maybe sometimes different
        sum_rows_zero = results.sum(axis=1) == 0
        skip_last_zeros = len(results) - min(24, sum_rows_zero.sum())
        results = results.iloc[skip_init_months:skip_last_zeros].mean()
    else:
        results = results.iloc[skip_init_months:].sum()
        # for the sum the 0 make no difference

    # change first header with # sign
    results_index_0 = results.index[0]
    results = results.rename({results_index_0: results_index_0.replace("# ", "")})

    return results


def import_mon_paras_zip_agg(zip_file, part_arcdir, skip_init_months, paras="all", index_cols=["SIM_ID", "lanu_ID", "BF_ID"]):
    """
    Import the monthly results for a bunch of parameters of a single run in an opened Zipfile.

    Parameters
    ----------
    zip_file : zipfile.ZipFile
        The Outputs Zipfile.
    part_arcdir : pathlike Path
        The archive path to the output parts folder.
    skip_init_months : int
        The number of months(lines) to skip before calculating the sum.
    paras : list of str or str, optional
        The Parameter to import.
        One of ["et", "N", "kap", "inf", "oa", "tp", "w_boden", "w_wurzel", "za"].
        If paras="all" then all the parameters are imported.
        The default is "all".
    index_cols: list of str or str, optional
        The name of the different parts of the id column, splited by "_".
        The default is ["SIM_ID", "lanu_ID", "BF_ID"].

    Returns
    -------
    results : pd.DataFrame
        The results of the single run.

    """
    if paras == "all":
        paras = ["et", "N", "kap", "inf", "oa",
                 "tp", "w_boden", "w_wurzel", "za"]
    elif type(paras) == str:
        paras = [paras]

    results = pd.DataFrame(columns=paras)
    for para in paras:
        ser_para = import_mon_para_zip_agg(zip_file=zip_file,
                                           part_arcdir=part_arcdir,
                                           para=para,
                                           skip_init_months=skip_init_months)
        results[para] = ser_para

    # add station ID and change Index
    results["STAT_ID"] = int(part_arcdir.parent.stem)
    results[index_cols] = pd.DataFrame(results.index.str.split("_").tolist(),
                                    columns=index_cols,
                                    index=results.index
                                    ).astype(int)
    results.set_index(index_cols, inplace=True)

    return results


def import_mon_zip_agg(zip_fp, skip_init_months, paras="all"):
    """
    Import all the aggregated monthly results for one zipfile.

    Parameters
    ----------
    zip_fp : pathlike Path
        The filepath of the outputs Zip file.
    skip_init_months : int
        The number of months(lines) to skip before calculating the sum.
    paras : list of str or str, optional
        The Parameter to import.
        One of ["et", "N", "kap", "inf", "oa", "tp", "w_boden", "w_wurzel", "za"].
        If paras="all" then all the parameters are imported.
        The default is "all".

    Raises
    ------
    KeyError
        If the providen columns do not exist.

    Returns
    -------
    results : pd.DataFrame
        A DataFrame with all the results together.

    """
    with ZipFile(zip_fp, "r") as zf:
        # get list of files in Zip
        zf_list = pd.Series(zf.namelist()).apply(Path)
        zf_list_mon = pd.Series(zf_list)[
            [True if re.search(r".*monat_val_N.csv", file.name) else False for file in zf_list]]

        # create first entry in df if not existing
        results = import_mon_paras_zip_agg(zip_file=zf,
                                           part_arcdir=zf_list_mon.iloc[0].parents[1],
                                           skip_init_months=skip_init_months,
                                           paras=paras)

        # import every file and add to results
        for zf_file_save in progressbar(zf_list_mon[1:], line_breaks=False):
            results_i = import_mon_paras_zip_agg(zip_file=zf,
                                                 part_arcdir=zf_file_save.parents[1],
                                                 skip_init_months=skip_init_months,
                                                 paras=paras)
            results = results.append(results_i)

    return results


def mute_stdout():
    sys.stdout = open(os.devnull, 'w')


def import_zips(out_zips, how, kwargs):
    """
    Import a bunch of output-zipfiles in a multiprocessing mode.

    Can Import the monthly values and aggregate or import the overall sums.

    Parameters
    ----------
    out_zips : list of pathlike Path
        A list of the filepaths of the outputs Zip files.
    how : str
        Should the ovarall sum get taken or the monthly values get aggregated?
        "mon": use the import_mon_zip_agg function to import one zip file.
        "tot": use the import_tot_zip function to import one zip file.
    kwargs : kwargs
        The keyword arguments to get handed to the import_zip... function.
        If how="tot": possible arguments are columns & with_input
        If how="mon": possible arguments are skip_init_months & paras

    Returns
    -------
    results : pd.DataFrame
        A DataFrame with all the results together.

    """
    # select the right import function
    if how == "tot":
        imp_func = import_tot_zip
    elif how == "mon":
        imp_func = import_mon_zip_agg
    else:
        raise ValueError("The how parameter has no correct value. " +
                         "Use 'mon' or 'tot'")

    # start a multiprocessing pool
    pool = Pool(processes=os.cpu_count()-1,
        initializer=mute_stdout)

    # start all the imports of the zip files
    procs = []
    for out_zip in out_zips:
        procs.append(pool.apply_async(
            func=imp_func,
            args=[out_zip,],
            kwds=kwargs
            ))

    # look and wait for finished processes
    try:
        answs = []
        bar = ProgressBar(max_value=len(out_zips), line_breaks=False)
        bar.start()
        while len(procs) != 0 :
            for proc in procs:
                if proc.ready():
                    answs.append(proc.get())
                    procs.remove(proc)
                    bar.update(bar.value + 1)
            time.sleep(1)
            bar.update(bar.value)
        bar.finish()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
        return None

    # gather all the results and merge the dataframes
    results = answs[0]

    if len(out_zips) > 0:
        for results_i in answs[1:]:
            results = results.append(results_i)

    # check if no duplicates are in the dataframe
    if results.index.has_duplicates :
        warnings.warn(
            "There are several input runs with the same Index name.")

    return results