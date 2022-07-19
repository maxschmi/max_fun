#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of functions to import raster data from the DWD-CDC Server."""

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

# import libraries
import os
if "PROJ_LIB" not in os.environ:
    os.environ["PROJ_LIB"] = "C:\\OSGeo4W64\\share\\proj"
import geopandas as gpd
from pathlib import Path
import gzip
import zipfile
import shutil
import re
import pandas as pd
from ftplib import FTP
import tarfile
import rasterio as rio
import numpy as np
from io import BytesIO
from progressbar import progressbar


# classes
# --------
class RegnieCoords:
    '''
    Class for conversion of Regnie coords.

    Source: DWD
    '''

    xdelta_grad = 1.0 /  60.0
    ydelta_grad = 1.0 / 120.0

    @staticmethod
    def convertPixelToGeographicCoordinates(cartesian_point_regnie): # (y, x)
        """ Berechnungsfunktion """

        lat = (55.0 + 10.0 * RegnieCoords.ydelta_grad) - (cartesian_point_regnie[0] - 1) * RegnieCoords.ydelta_grad
        lon = ( 6.0 - 10.0 * RegnieCoords.xdelta_grad) + (cartesian_point_regnie[1] - 1) * RegnieCoords.xdelta_grad

        return lat, lon


# functions
#----------
def create_ascii(fn_in, fn_out=None, replace=False):
    """
    Create a ASCII file from regnie data.

    Parameters
    ----------
    fn_in : str or Path
        The path of the regnie file.
    fn_out : str or Path, optional
        The path of the ASCII file to be created.
        If None the same as fn_in with an adition of ".asc" will get taken.
        The default is None.
    replace : bool, optional
        Should the input file get deleted after the conversion.
        The default is False.

    Raises
    ------
    NameError
        If the given file is not existing.

    Returns
    -------
    None.

    """
    # check the file
    fn_in = Path(fn_in)
    if not fn_in.is_file():
        raise NameError("The given file is not existing!")

    with open(fn_in, "r") as f_in:
        lines_in = f_in.readlines()

    # initiate file out
    if fn_out is None:
        fn_out = fn_in.parent.joinpath(fn_in.stem + ".asc")
    f_out = open(fn_out, "w")
    yll, xll = RegnieCoords.convertPixelToGeographicCoordinates((971, 0.75)) #971 davor Y_MAX
    header = ("ncols " + str(611 * 2) + "\n" + "nrows 971\n" +
              "xllcenter " + str(xll) + "\n" +
              "yllcenter " + str(yll) + "\n" +
              "cellsize " + str(1/120) + "\n" + "nodata_value -999\n")
    f_out.write(header)

    # convert lines to ascii
    for row in lines_in:
        new_row = []
        for i in range(0, len(row), 4):
            new_row.append(row[i:i+4])
            new_row.append(row[i:i+4]) # because the lon/lat cellsize is the double
        f_out.write(" ".join(new_row) + "\n")

    f_out.close()

    # delete file if replace is on
    if replace:
        fn_in.unlink()

def create_xyz(fn_in, fn_out=None, ignore_missings=True,
               replace=False, do_id=False):
    """
    Create a XYZ csv file from regnie data.

    Parameters
    ----------
    fn_in : str or Path
        The path of the regnie file.
    fn_out : str or Path, optional
        The path of the ASCII file to be created.
        If None the same as fn_in with an adition of ".csv" will get taken.
        The default is None.
    ignore_missings : bool, optional
        Should missing values get skiped. This results in smaller files.
        The default is True.
    replace : bool, optional
        Should the input file get deleted after the conversion.
        The default is False.
    do_id : bool, optional
        Should the output file have a ID column
        to join with the REGNIE polygone.
        The default is False.

    Raises
    ------
    NameError
        If the given file is not existing.

    Returns
    -------
    None.

    """
    # check the file
    fn_in = Path(fn_in)
    if not fn_in.is_file():
        raise NameError("The given file is not existing!")

    # open fn_out and create header
    if fn_out is None:
        fn_out = fn_in.parent.joinpath(fn_in.stem + ".xyz")
    f_out = open(fn_out, "w")
    if do_id:
        f_out.write("ID,X,Y,Z\n")
    else:
        f_out.write("X,Y,Z\n")

    # read fn_in
    with open(fn_in, "r") as f_in:
        lines_in = f_in.readlines()

    # convert lines_in to xyz values
    _id = 1
    # static variables
    y_max = 971
    x_max = 611
    for y in range(1, y_max+1):
        row_in = lines_in[y-1]
        i = 0

        for x in range(1, x_max+1):
            i_val = int(row_in[i:i+4])
            i += 4

            # treate missing values
            if i_val < 0:   # In Ursprungs-Rasterdatei: -999
                if ignore_missings:
                    continue
                i_val = -1  # weiter mit diesem Wert

            lat, lon = RegnieCoords.convertPixelToGeographicCoordinates((y, x))

            if do_id:
                f_out.write("%i,%f,%f,%d\n" % (_id, lon, lat, i_val))
                _id += 1
            else:
                f_out.write("%f,%f,%d\n" % (lon, lat, i_val))

    f_out.close()

    if replace:
        fn_in.unlink()


def unzip_convert_folder(folder, out_type="ASCII", replace=True, **kwargs):
    """
    Unzip and convert all the regnie files from a folder.

    Parameters
    ----------
    folder : str or Path
        The path were the \*.gz regnie files are stored in.
    out_type : str, optional
        The type of converter to use. Possible values are "ASCII" or "XYZ"
        The default is "ASCII".
    replace : bool, optional
        Whether to replace the original files.
        The Default is True.
    **kwargs : TYPE
        Arguments to pass to the converter.

    Raises
    ------
    NameError
        If the given folder is not a folder or if the converter doesn't exist.'

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # get the converter
    if out_type == "ASCII":
        converter = create_ascii
        ending = ".asc"
    elif out_type == "XYZ":
        converter = create_xyz
        ending = ".xyz"
    else:
        raise NameError("There is no converter defined for " + out_type +
                        "\nplease use one of those ['XYZ', 'ASCII']")

    # unzip and convert all the files in the folder
    for file in folder.glob("*.gz"):
        if ".ARC." in file.name:
            continue

        # define filenames
        fn_unzip = file.parent.joinpath(file.stem + "_orig.txt")
        fn_out = file.parent.joinpath(file.stem + ending)

        # unzip
        with gzip.open(file, 'rb') as f_in:
            with open(fn_unzip, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if replace:
            file.unlink()

        # convert file
        converter(fn_in=fn_unzip, fn_out=fn_out, replace=replace, **kwargs)


def gather_xyz_gpk(folder, merge_id=False):
    """
    Gather all the xyz files from a folder into one geopackage(shapefile).

    Creates a data frame with one column per file and the specific name.
    Save the output as csv with the points coordinates.
    Join this DataFrame with the REGNIE Polygon and save as Geopackage (.gpk).
    This function takes long and should only get used sometimes.

    Parameters
    ----------
    folder : str or Path
        The path were the \*.xyz files are stored in.
    merge_id : bool, optional
        Should the files get merged on ID. Merging on ID is way faster.
        If set to False it will get merged on spatial relations.
        The default is False.


    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # get a list of the xyz files
    col_pattern = r"(?<=(\.))\w{2,3}(?=(\.xyz))"
    files = list(folder.glob("*.xyz"))
    col_name = re.search(col_pattern, files[0].name)[0]

    # merge these files together to one dataframe
    df_union = pd.read_table(files[0], sep=",",
                             names=["X", "Y", col_name],
                             skiprows=1)
    for file in files[1:]:
        col_name = re.search(col_pattern, file.name)[0]
        df = pd.read_table(file, sep=",",
                           names=["X", "Y", col_name],
                           skiprows=1)
        if merge_id:
            df_union = pd.concat([df_union, df[col_name]],
                                 axis=1, join="outer")
        else:
            df_union = pd.merge(df_union, df, how="outer", on=["X", "Y"])

    # export as raster with coordinates of central points
    df_union.to_csv(folder.joinpath("data_merge_point.csv"), index=False)

    # get REGNIE polygone
    this_dir = Path(__file__).parent
    regnie_poly_path = this_dir.parents[2].joinpath("GIS/DWD/regnie/polygon")
    regnie_poly = gpd.read_file(regnie_poly_path)
    regnie_poly.set_crs(crs="EPSG:4326", inplace=True)
    regnie_poly.ID = regnie_poly.ID.astype(int)
    regnie_poly.set_index("ID", verify_integrity=True, inplace=True)

    # merge with regnie_polygone
    if merge_id:
        regnie_poly_join = gpd.GeoDataFrame(pd.concat([regnie_poly, df_union],
                                                      axis=1, join='inner'),
                                            crs=regnie_poly.crs)
    else:
        gdf = gpd.GeoDataFrame(df_union.iloc[:, 2:],
                       geometry=gpd.points_from_xy(df_union.X,
                                                   df_union.Y,
                                                   crs="EPSG:4326"))
        regnie_poly_join = gpd.sjoin(regnie_poly, gdf,
                                     op="contains", how="inner")
        regnie_poly_join = regnie_poly_join.drop(["index_right"], axis=1)

    regnie_poly_join.to_file(folder.joinpath("data_merge.gpk"),
                             driver="GPKG")

def gather_xyz_tif(folder, dst="input"):
    """
    Gather all the xyz files from a folder into one GeoTiff.

    The advantage is that a Tiff file is not so big.

    Parameters
    ----------
    folder : str or Path
        The path were the \*.xyz files are stored in.
    dst : str or Path, optional
        The path were the output GeoTiff file gets stored.
        If "input", the file will get stored in the input folder.
        If path is a directory the file will get named "regnie_merge.tif".
        The Default is "input".

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    if dst == "input":
        dst = folder.joinpath("regnie_merge.tif")
    else:
        dst = Path(dst)
        if dst.is_dir():
            dst = dst.joinpath("regnie_merge.tif")
        elif dst.suffix != ".tif" and dst.suffix != ".tiff":
            raise NameError("The given dst is not a file, folder or 'input'!")

    # get a list of the xyz files
    files = list(folder.glob("*.xyz"))

    # read the xyz file and get the band names
    raster_join = []
    band_names = {}
    for i, file in enumerate(files):
        raster_join.append(rio.open(file).read(fill_value=9999)[0])
        band_name = re.search(r"(?<=(\.))\w{2,3}$", file.stem)[0]
        band_names.update({i+1: band_name})
    raster_join = np.array(raster_join)

    # write the GeoTiff file
    profile = {'driver': 'GTiff', 'dtype': rio.uint16,
               'nodata': 9999, 'width': 551, 'height': 935,
               'count': len(raster_join),
               'crs': rio.crs.CRS.from_epsg(4326),
               'transform': rio.Affine(0.016666665454545457, 0.0,
                                            5.858333667272727, 0.0,
                                            -0.008333332976445396,
                                            55.06249966648822),
               'tiled': False}
    with rio.open(dst, 'w', **profile) as f_dst:
        f_dst.write(raster_join.astype(rio.uint16))

        # add the band name
        for i, file in enumerate(files):
            band_name = re.search(r"(?<=(\.))\w{2,3}$", file.stem)[0]
            f_dst.set_band_description(i+1, band_name)

def gather_asc_tif(folder, dst="input", crs="EPSG:31467",
                   dtype="uint16", band_regex=r"(?<=([_\.]))\w{2,4}$"):
    """
    Gather all the ASCII files from a folder into one GeoTiff.

    The advantage is that a Tiff file is not so big.

    Parameters
    ----------
    folder : str or Path
        The path were the \*.asc files are stored in.
    dst : str or Path, optional
        The path were the output GeoTiff file gets stored.
        If "input", the file will get stored in the input folder.
        If path is a directory the file will get named "regnie_merge.tif".
        The Default is "input".
    crs : str, optional
        The coordinate reference system of the input files.
        The output file will get the same crs.
        The parameter needs to be in a rasterio (fiona) format.
        The default is "EPSG:31467".
    dtype : str, optional
        A valid dtype for the raster field values.
        The default is "uint16".
    band_regex: str or regex, optional
        The pattern for which to look in the file.stem to find the name of the band.
        The default is r"(?<=([_\.]))\w{2,3}$", so the last 2 or 3 letters.

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        if (folder.is_file() and folder.suffix ==".asc"):
            files = [folder]
            folder = folder.parent
        else:
            raise NameError("The given folder is not a folder or valid filepath!")

    if dst == "input":
        dst = folder.joinpath("merge.tif")
    else:
        dst = Path(dst)
        if dst.is_dir():
            dst = dst.joinpath("merge.tif")
        elif dst.suffix != ".tif" and dst.suffix != ".tiff":
            raise NameError("The given dst is not a tif-file, " +
                            "folder or 'input'!")

    # set na value
    if dtype == "uint8":
        na_value = 255
    else:
        na_value = -999

    # get a list of the asc files
    if "files" not in locals():
        files = list(folder.glob("*.asc"))

    # read the asc files and get the band names
    raster_join = []
    band_names = {}
    for i, file in enumerate(files):
        try:
            with rio.open(file) as raster:
                raster_np = raster.read(fill_value=na_value)[0]
                profile = raster.profile
        except:
            with open(file) as f:
                lines = f.readlines()
            if re.match(".*header.*", lines[0]):
                last_header_count = list(filter(re.compile(".*ASCII-Raster-Format.*").match, lines))
                if len(last_header_count):
                    last_header_line = lines.index(last_header_count[0])
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as repaired_file:
                        repaired_file.writelines(
                            [bytes(line, encoding="utf8") for line in lines[last_header_line+1:]])
                    with rio.open(repaired_file.name) as raster:
                        raster_np = raster.read(fill_value=na_value)[0]
                        profile = raster.profile

                    Path(repaired_file.name).unlink() # remove temporary file
                else:
                    raise ValueError("There was a problem with the input ASC file that could not get resolved")
            else:
                raise ValueError("There was a problem with the input ASC file that could not get resolved")
                

        raster_join.append(raster_np)
        band_name = re.search(band_regex, file.stem)[0]
        band_names.update({i+1: band_name})
    raster_join = np.array(raster_join)

    # write the GeoTiff file
    profile.update({'driver': 'GTiff', 'dtype': dtype,
                    'nodata': na_value,
                    'count': len(raster_join),
                    'crs': crs
                    })

    with rio.open(dst, 'w', **profile) as f_dst:
        f_dst.write(raster_join.astype(dtype))

        # add the band name
        for i, band_name in band_names.items():
            f_dst.set_band_description(i, band_name)

def gather_nc_tif(folder, dst="input", band_regex=r"(?<=_)\w{3,4}$", crs="EPSG:3034"):
    """
    Gather all the NC files from a folder into one GeoTiff.

    The advantage is that a Tiff file is not so big.

    Parameters
    ----------
    folder : str or Path
        The path were the \*.nc files are stored in.
    dst : str or Path, optional
        The path were the output GeoTiff file gets stored.
        If "input", the file will get stored in the input folder.
        If path is a directory the file will get named "hyras.tif".
        The Default is "input".
    band_regex: str or regex, optional
        The pattern for which to look in the file.stem to find the name of the band.
        The default is r"(?<=([_\.]))\w{2,3}$", so the last 2 or 3 letters.
    crs : str, optional
        The coordinate reference system of the input files.
        The output file will get the same crs.
        The parameter needs to be in a rasterio (fiona) format.
        The default is "EPSG:3034".

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    if dst == "input":
        dst = folder.joinpath("merge.tif")
    else:
        dst = Path(dst)
        if dst.is_dir():
            dst = dst.joinpath("merge.tif")
        elif dst.suffix != ".tif" and dst.suffix != ".tiff":
            raise NameError("The given dst is not a tif-file, " +
                            "folder or 'input'!")

    # get a list of the nc files
    files = list(folder.glob("*.nc"))

    # read the nc files and get the band names
    raster_join = []
    band_names = {}
    for i, file in enumerate(files):
        with rio.open(file) as raster:
            raster_np = raster.read()[0]
            profile = raster.profile
            
        raster_join.append(raster_np)
        band_name = re.search(band_regex, file.stem)[0]
        band_names.update({i+1: band_name})
    raster_join = np.array(raster_join)

    # write the GeoTiff file
    profile.update({'driver': 'GTiff',
                    'crs':  crs,
                    'count': len(raster_join)})

    with rio.open(dst, 'w', **profile) as f_dst:
        f_dst.write(raster_join)

        # add the band name
        for i, band_name in band_names.items():
            f_dst.set_band_description(i, band_name)


def download_hyras_ma(folder, period="newest"):
    """
    Download the newest HYRAS multi_annual file and extract it to a folder.

    Parameters
    ----------
    folder : str or Path
        The folder path where the files get stored in.
    period : str, optional
        The climate period to download, e.g. "1991-2020".
        Alternativly you can enter "newest" to get the most recent climate period.

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # open ftp connection
    ftp = FTP("opendata.dwd.de")
    ftp.login()

    # get the file list
    ftp_folder = "climate_environment/CDC/grids_germany/multi_annual/hyras_de/precipitation"
    ftp_files_all = ftp.nlst(ftp_folder)
    re_last_part = r"_v3-0_de_[a-zA-Z]{3,4}\.nc$"
    comp = re.compile(r".*\d{4}_\d{4}" + re_last_part)
    ftp_files = list(filter(comp.match, ftp_files_all))

    # get the wanted period
    if period == "newest":
        comp = re.compile(r"\d{4}_\d{4}(?=" + re_last_part + ")")
        possible_periods = list(set(list(map(lambda x: comp.search(x)[0], ftp_files))))
        possible_periods.sort()
        period = possible_periods[-1]
    else:
        period = period.replace("-", "_")
        if not re.match(r"\d{4}_\d{4}", period):
            raise ValueError(f"The given period {period} is not in a valid format."+
                " \nPlease hand in in the format of e.g. \"1991_2020\".")

    # get files of given period
    comp = re.compile(f".+{period}{re_last_part}")
    dwd_hyras_fps = list(filter(comp.match, ftp.nlst(ftp_folder)))

    # download the files from dwd
    for dwd_fp in dwd_hyras_fps:
        local_fp = folder.joinpath(Path(dwd_fp).name)
        with open(local_fp, "wb") as file:
            ftp.retrbinary("RETR " + dwd_fp, file.write)

def download_regnie_daily(folder, years):
    """
    Download the newest regnie daily files and axtract them to a folder.

    Create a folder with the name of the year and extract all the daily regnie
    \*.gz files into it.

    Parameters
    ----------
    folder : str or Path
        The folder path where the files get stored in.

    years : list of str or int
        A list of the years for which to download the regnie daily files.

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # open ftp connection
    ftp = FTP("opendata.dwd.de")
    ftp.login()

    # get the newest file
    ftp_folder = "climate_environment/CDC/grids_germany/daily/regnie/"
    comp_regnie = re.compile(r".+\d{4}m\.tar$")
    regnie_fps = list(filter(comp_regnie.match, ftp.nlst(ftp_folder)))
    comp_years = re.compile(".*(" + "|".join(map(str, years)) + ").*")
    selected_regnie_fps = list(filter(comp_years.match, regnie_fps))

    for fp in progressbar(selected_regnie_fps):
        # download the tar folders from dwd
        tar_fp = folder.joinpath(fp.split("/")[-1])
        with open(tar_fp, "wb") as tar:
            ftp.retrbinary("RETR " + fp, tar.write)

        # extract tarfile to folder
        extract_dir = folder.joinpath(re.search(r"\d+", tar_fp.stem)[0])
        if extract_dir.is_dir():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir()

        with tarfile.open(tar_fp) as tar:
            tar.extractall(extract_dir)
        tar_fp.unlink()

def download_regnie_ma(folder):
    """
    Download the newest regnie multi_annual file and extract it to a folder.

    Parameters
    ----------
    folder : str or Path
        The folder path where the files get stored in.

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # open ftp connection
    ftp = FTP("opendata.dwd.de")
    ftp.login()

    # get the newest file
    ftp_folder = "climate_environment/CDC/grids_germany/multi_annual/regnie/"
    comp = re.compile(r".+\d{4}\.tar$")
    dwd_regnie_fps = list(filter(comp.match, ftp.nlst(ftp_folder)))
    dwd_regnie_fps.sort()
    newest_regnie_fp = dwd_regnie_fps[len(dwd_regnie_fps)-1]

    # download the tar folder from dwd
    tar_fp = folder.joinpath(newest_regnie_fp[-8:])
    with open(tar_fp, "wb") as tar:
        ftp.retrbinary("RETR " + newest_regnie_fp, tar.write)

    # extract tarfile to folder
    with tarfile.open(tar_fp) as tar:
        tar.extractall(folder)
    tar_fp.unlink()

def download_ma_t(folder):
    """
    Download the newest multi_annual temperature files and extract them.

    Parameters
    ----------
    folder : str or Path
        The folder path where the files get stored in.

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # create name dict
    ma_codes = {"00":"JAH", "01": "JAN", "02": "FEB", "03": "MAE", "04": "APR",
                "05": "MAI", "06": "JUN", "07": "JUL", "08": "AUG", "09": "SEP",
                "10": "OKT", "11": "NOV", "12": "DEZ",
                "13": "FR", "14": "SO", "15": "HE", "16": "WI",
                "17": "JAH"}

    # open ftp connection
    ftp = FTP("opendata.dwd.de")
    ftp.login()

    # get the newest file
    ftp_folder = "/climate_environment/CDC/grids_germany/multi_annual/air_temperature_mean/"
    comp = re.compile(r".+\d{2}\.asc\.gz$")
    dwd_fps = list(filter(comp.match, ftp.nlst(ftp_folder)))
    dwd_fps.sort(reverse=True)
    dwd_new_fps = dwd_fps[0:17]

    # download and unzip
    for dwd_fp in dwd_new_fps:
        temp = BytesIO()
        ftp.retrbinary("RETR " + dwd_fp, temp.write)
        fp = folder.joinpath(Path(dwd_fp).stem[:-6] +
                             ma_codes[Path(dwd_fp).stem[-6:-4]] + ".asc")

        with open(fp, "wb") as f:
            f.write(gzip.decompress(temp.getvalue()))

def download_ma_et(folder):
    """
    Download the newest multi_annual evapotranspiration files and axtract them.

    Parameters
    ----------
    folder : str or Path
        The folder path where the files get stored in.

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # create name dict
    ma_codes = {"00":"JAH", "01": "JAN", "02": "FEB", "03": "MAE", "04": "APR",
                "05": "MAI", "06": "JUN", "07": "JUL", "08": "AUG", "09": "SEP",
                "10": "OKT", "11": "NOV", "12": "DEZ"}

    # open ftp connection
    ftp = FTP("opendata.dwd.de")
    ftp.login()

    # get the newest file
    ftp_folder = "/climate_environment/CDC/grids_germany/multi_annual/evapo_p/"
    comp = re.compile(r".+\d{2}\.asc\.gz$")
    dwd_fps = list(filter(comp.match, ftp.nlst(ftp_folder)))
    dwd_fps.sort(reverse=True)
    dwd_new_fps = dwd_fps[0:12]

    # download and unzip
    for dwd_fp in dwd_new_fps:
        temp = BytesIO()
        ftp.retrbinary("RETR " + dwd_fp, temp.write)
        fp = folder.joinpath(Path(dwd_fp).stem[:-6] +
                             ma_codes[Path(dwd_fp).stem[-6:-4]] + ".asc")

        with open(fp, "wb") as f:
            f.write(gzip.decompress(temp.getvalue()))

def download_ma(folder, para):
    """
    Download the newest multi_annual raster files for one parameter and axtract them.

    Parameters
    ----------
    folder : str or Path
        The folder path where the files get stored in.
    para : str
        The parameter for which to download the files.
        Either 'N', 'ET', 'T', 'SUN_DUR' or 'SOL_RAD'.

    Raises
    ------
    NameError
        If the given folder is not a folder.

    Returns
    -------
    None.

    """
    # check the folder
    folder = Path(folder)
    if not folder.is_dir():
        raise NameError("The given folder is not a folder!")

    # create name dict
    dic = {"T": {"bands": 17,
                 "ftp_folder": "/climate_environment/CDC/grids_germany/multi_annual/air_temperature_mean/"},
           "ET": {"bands": 12,
                 "ftp_folder": "/climate_environment/CDC/grids_germany/multi_annual/evapo_p/"},
           "N": {"bands": 17,
                 "ftp_folder": "/climate_environment/CDC/grids_germany/multi_annual/precipitation/"},
           "SUN_DUR": {"bands": 17,
                 "ftp_folder": "/climate_environment/CDC/grids_germany/multi_annual/sunshine_duration/"},
           "SOL_RAD": {"bands": 12,
                 "ftp_folder": "/climate_environment/CDC/grids_germany/multi_annual/radiation_global/"},
           "ma_codes": {"01": "JAN", "02": "FEB", "03": "MAE", "04": "APR",
                        "05": "MAI", "06": "JUN", "07": "JUL", "08": "AUG",
                        "09": "SEP", "10": "OKT", "11": "NOV", "12": "DEZ",
                        "13": "FR", "14": "SO", "15": "HE", "16": "WI",
                        "17": "JAH"}
           }

    para = para.upper()
    if para not in dic:
        raise NameError("The given para value '" + str(para) + "' is not valid!")

    # open ftp connection
    ftp = FTP("opendata.dwd.de")
    ftp.login()

    # get the newest file
    comp = re.compile(r".+\d{2}\.asc\.gz$|.+\d{2}\.zip$")
    dwd_fps = list(filter(comp.match, ftp.nlst(dic[para]["ftp_folder"])))
    dwd_fps.sort(reverse=True)
    dwd_new_fps = dwd_fps[0:dic[para]["bands"]]
    
    # download and unzip
    for dwd_fp in dwd_new_fps:
        temp = BytesIO()
        ftp.retrbinary("RETR " + dwd_fp, temp.write)
        dwd_fp = Path(dwd_fp)
        if dwd_fp.suffix == ".gz":
            fp = folder.joinpath(Path(dwd_fp).stem[:-6] +
                                dic["ma_codes"][Path(dwd_fp).stem[-6:-4]] +
                                ".asc")
            with open(fp, "wb") as f:
                f.write(gzip.decompress(temp.getvalue()))
        else:
            fp = folder.joinpath(
                Path(dwd_fp).stem[:-2] +
                dic["ma_codes"][Path(dwd_fp).stem[-2:]] + ".asc")
            with open(fp, "wb") as f:
                f.write(zipfile.ZipFile(temp).read(Path(dwd_fp).stem+ ".asc"))


# test
# fn_regnie = "D:/Dokumenter/UNI/Master/Freiburg/2_Masterarbeit/GIS/DWD/RA701231"
# create_ascii(fn_regnie)
# unzip_convert_folder("D:/OneDrive - bwedu/Masterarbeit/GIS/DWD/regnie/multi_anual_81-10")
# unzip_convert_folder("D:/OneDrive - bwedu/Masterarbeit/GIS/DWD/regnie/multi_anual_81-10",
#                       out_type="XYZ")

# weitere ideen wenn Zeit da ist:
    # tif ohne umwege von xyz datei erstellen, alse gather_xyz_tiff, create_xyz
      # und unzip_convert_folder zusammenführen
    # REGNIE_Daten_aufbereitung Skript in einer Funktion
    # tif only select months, kick JAH, SO,WI,HE,FR
      # -> then set datatype to np.uint8