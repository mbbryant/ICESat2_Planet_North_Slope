#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import skimage.transform as transform

def create_transect(origin, orientation, length):
    """
    Create a transect given an center, orientation and length.
    Points are spaced at 1m intervals.
    
    Source code: SDS_transects.py by Kilian Vos, Water Research Laboratory, School of Civil and 
    Environmental Engineering, UNSW Sydney 2018. Downloaded from the CoastSat
    Github repository (https://github.com/kvos/CoastSat)
    
    Modifed by Marnie Bryant(m1bryant) on Sep 17 2024 to define the origin as the center 
    of the transect instead of the start
    
    Arguments:
    -----------
    origin: np.array
        contains the X and Y coordinates of the center of the transect
    orientation: int
        angle of the transect (anti-clockwise from North) in degrees
    length: int
        length of the transect in metres
        
    Returns:    
    -----------
    transect: np.array
        contains the X and Y coordinates of the transect
        
    """   
    
    # origin of the transect
    x0 = origin[0]
    y0 = origin[1]
    # orientation of the transect
    phi = (90 - orientation)*np.pi/180 
    # create a vector with points at 1 m intervals
    x = np.linspace(-length/2,length/2,length+1)
    y = np.zeros(len(x))
    coords = np.zeros((len(x),2))
    coords[:,0] = x
    coords[:,1] = y 
    # translate and rotate the vector using the origin and orientation
    tf = transform.EuclideanTransform(rotation=phi, translation=(x0,y0))
    transect = tf(coords)
                
    return transect
