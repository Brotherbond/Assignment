#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:00:03 2021

@author: devs
"""

import os, re, glob
import pandas as pd
import matplotlib.pyplot as plt
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from IPython.display import IFrame
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')