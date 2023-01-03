"""
run all of the files
"""

import os
if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')
import stat_figures
import stat_time
import stat_accuracy
import nonstat_figures
import nonstat_time
