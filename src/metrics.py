#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the effect sizes metrics for mediation. 
From Preacher KJ, Kelley K. 
Effect size measures for mediation models: quantitative strategies for communicating indirect effects. 
Psychol Methods. 2011 Jun;16(2):93-115. doi: 10.1037/a0022658
"""


def r45_med(*, r2_MY, r2_YMX, r2_XY):
    return r2_MY - (r2_YMX-r2_XY)

def r46_med(*, r2_XM, r2_MY, r2_YMX, r2_XY):
    return r2_XM*(r2_YMX-r2_XY)/(1-r2_XY)

def r47_med(*, r2_XM, r2_MY, r2_YMX, r2_XY):
    return r46_med(r2_XM=r2_XM, r2_MY=r2_MY, r2_YMX=r2_YMX, r2_XY=r2_XY)/r2_YMX
    
def r2_med(xy_t, xy_p, my_t, my_p, xmy_t, xmy_p):
    from sklearn.metrics import r2_score
    
    r2_XY = r2_score(xy_t, xy_p)
    r2_MY = r2_score(my_t, my_p)
    r2_XMY = r2_score(xmy_t, xmy_p)

    
    if r2_XMY >0:
        res = r45_med(r2_MY = r2_MY, r2_YMX = r2_XMY, r2_XY = r2_XY)
    else:
        res = 0
    return res
