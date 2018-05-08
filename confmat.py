# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:43:20 2018

@author: tchat
"""

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

conf_arr = [[94,  2,  0, 10,  0],
            [96,  6,  0, 13, 10],
            [55,  1,  0, 11,  6],
            [74,  2,  0, 20,  9],
            [61,  3,  0, 11, 12]]

df_cm = pd.DataFrame(conf_arr,range(5),range(5))

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, cmap='coolwarm', annot=True)

