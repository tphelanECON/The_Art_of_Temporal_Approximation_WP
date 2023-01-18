"""
run all of the files necessary to replicate the paper
"""

import os
if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')
if not os.path.exists('../main/output'):
    os.makedirs('../main/output')

"""
Scripts for stationary analysis
"""

from .stationary import stat_figures

import stationary.stat_figures
import stationary.stat_time_accuracy
import stationary.stat_accuracy_EGM

"""
Scripts for nonstationary analysis


import nonstationary.nonstat_figures
import nonstationary.nonstat_time_accuracy
import nonstationary.nonstat_accuracy_EGM
import nonstationary.naive_seq_accuracy
"""
