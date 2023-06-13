#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of some helpfull functions to work with climate data."""

__author__ = "Max Schmit"
__copyright__ = "Copyright 2023, Max Schmit"

import numpy as np

def et_pen_mont(ta, td, rsds, u2, doy, elev=100, lat=48 , crop="short"):
    """This is a function to calculate the daily Evapotranspiration, like defined by Penman-Monteith.

    Parameters
    ----------
    ta : 1D Array or pd.Series
        daily air temperature
    td : 1D Array or pd.Series
        daily Feuchtetemperatur
    rsds : 1D Array or pd.Series
        daily Globalstrahlung (W/m2)
    u2 : 1D Array or pd.Series
        daily wind speed (m/s)
    doy : 1D Array or pd.Series
        julian dates
    elev : int, optional
        station elevation, by default 100
    lat : float, optional
        latitude of station, by default 48
    crop : str, optional
        Defines the roughness height.
        Short for Grassland.
        The default is "short"

    Returns
    -------
    et_d : 1D Array
        daily potential Evapotranspiration in mm/day
    """   
    # Define constants
    ALPHA = 0.23 # albedo for both short and tall crop
    LAMP = 2.45  		# latent heat of evaporisation = 2.45 MJ.kg**-1 at 20 degree Celcius
    RADCON = 0.0864   # 1 W/m2 = 0.0864 MJ/m2/day.
    SIGMA = 4.903e-09    # Stefan-Boltzmann constant = 4.903*10**-9 MJ.K**-4.m**-2.day**-1,
    G = 0   # ground heat flux
    GSC = 0.0820 #  solar constant (0.0820 [MJ/(m2 min)]) 
    
    # Saturated vapour pressure
    vas = 0.6108 * np.exp(17.27 * ta / (ta + 237.3)) # Equation S2.5
    
    # Vapour pressure based on dew ppoint temperature
    vabar = 0.6108 * np.exp(17.27 * td / (td + 237.3)) 		# Equation S2.8  HESS 2013  Thomas A McMahon
    
    # incoming solar radiation in MJ m2 day
    R_s = rsds*RADCON
    
    # Calculations from data and constants for Penman-Monteith Reference Crop
    
    P = 101.3 * ((293 - 0.0065 * elev) / 293)**5.26 # atmospheric pressure (S2.10)
    delta = 4098 * (0.6108 * np.exp((17.27 * ta)/(ta+237.3))) / ((ta + 237.3)**2) # slope of vapour pressure curve (S2.4)
    gamma = 0.00163 * P / LAMP # psychrometric constant (S2.9)
    
    d_r2 = 1 + 0.033*np.cos(2*np.pi/365 * doy) # dr is the inverse relative distance Earth-Sun (S3.6)
    delta2 = 0.409 * np.sin(2*np.pi/365 * doy - 1.39) # solar dedication (S3.7)
    w_s = np.arccos(-np.tan(lat*np.pi/180) * np.tan(delta2))  # sunset hour angle (S3.8)
    N = 24/np.pi * w_s # calculating daily values
    # R_a = (1440/pi) * d_r2 * constants$Gsc * (w_s * sin(lat*pi/180) * sin(delta2) + cos(lat*pi/180) * cos(delta2) * sin(w_s)) # extraterristrial radiation (S3.5) Markus Version
    R_a = (1440/np.pi) * d_r2 * GSC * \
        (w_s * np.sin(lat*np.pi/180) * np.sin(delta2) + np.cos(lat*np.pi/180) * \
         np.cos(delta2) * np.sin(w_s)) # extraterristrial radiation (S3.5)
    R_so = (0.75 + (2*10**-5)*elev) * R_a # clear sky radiation (S3.4)
    
    R_nl = SIGMA * (0.34 - 0.14 * np.sqrt(vabar)) * \
        (ta+273.2)**4   * (1.35 * R_s / R_so - 0.35) # estimated net outgoing longwave radiation (S3.3)
    R_nsg = (1 - ALPHA) * R_s # net incoming shortwave radiation (S3.2)
    R_ng = R_nsg - R_nl # net radiation (S3.1)
    
    if (crop == "short"):
        r_s = 70 # will not be used for calculation - just informative
        CH = 0.12 # will not be used for calculation - just informative
        et_d = (0.408 * delta * (R_ng - G) + gamma * 900 * u2 * \
                (vas - vabar)/(ta + 273)) / (delta + gamma * (1 + 0.34*u2)) # FAO-56 reference crop evapotranspiration from short grass (S5.18)
    else:
        r_s = 45 # will not be used for calculation - just informative
        CH = 0.50 # will not be used for calculation - just informative
        et_d = (0.408 * delta * (R_ng - G) + gamma * 1600 * u2 * \
                (vas - vabar)/(ta + 273)) / (delta + gamma * (1 + 0.38*u2)) # ASCE-EWRI standardised Penman-Monteith for long grass (S5.19)    
    
    return et_d


def pet_hamon(N,T):
    """Calculates the potential evapotranspiration according to Hamon (1963).

    inspired by:
        van Tiel, Marit; Freudiger, Daphné; Kohn, Irene; Seibert, Jan; Weiler, Markus; Stahl, Kerstin. (2022) Hydrological modelling of the glacierized headwater catchments in the Rhine basin - technical report. Online under: https://freidok.uni-freiburg.de/fedora/objects/freidok:226492/datastreams/FILE1/content

    Parameters
    ----------
    N : 1D Array of float
        maximum possible hours of daylight on Julian day [h]
    T : 1D Array of float
        mean daily temperature in °C.

    Returns
    -------
    pET : 1D Array of float
        potential evapotranspiration in mm/day.
    """    
    return np.power(N/12, 2)*np.power(np.e, T/16)

def daylight(latitude,day):
    """This is a function to calculate the maximum possible daily amount of daylight.

    inspired by: 

    Parameters
    ----------
    latitude : float
        latitude of the station in WGS84
    day : int
        julian day of the year from 1 to 365
    
    Return
    ----------
    daylightamount : np.array with float
        maximum possible daily amount of daylight in hours
    """
    P = np.arcsin(0.39795 * np.cos(0.2163108 + 2 * np.arctan(0.9671396 * np.tan(.00860 * (day - 186)))))
    pi = np.pi
    daylightamount = 24 - (24 / pi) * np.arccos(
        (np.sin((0.8333 * pi / 180) + np.sin(latitude * pi / 180) * np.sin(P)) / (np.cos(latitude * pi / 180) * np.cos(P))))
    return daylightamount