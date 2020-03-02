
import warnings
warnings.filterwarnings("ignore")
import searcher
import matplotlib.pyplot as plt
import plotter
import data
import mlrose
from time import time
import pandas as pd
import numpy as np

import fourpeak as four
import onemax
import flipflop

import neural2
import variable
 
if __name__ == '__main__':

    plotter.showPlot = True
 
    onemax.run()
    flipflop.run()
    four.run()
    variable.run()
    neural2.run('heart')

 
 
 
