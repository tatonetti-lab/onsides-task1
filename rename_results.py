"""
When you need to go back and rename all the files to match a new naming convention....

Current naming convention:
SOURCEAPI_MODEL_PROMPTVERSION_SYSTEMVERSION_GPTPARAMS_CLEANUPFLAG_DATASET_RUN[_EVALCRITERIA_EVALTYPE].csv

New naming convention:
SOURCEAPI_MODEL_PROMPTVERSION_SYSTEMVERSION_GPTPARAMS_DATASET_RUN[_EVALCRITERIA_EVALTYPE].csv

"""

import os
import sys

for f in os.listdir('results/'):
    parts = f.strip('.csv').split('_')

    if len(parts) == 8:
        # results file
        newparts = parts[:5] + parts[6:]
    elif len(parts) == 10:
        # evaluation file
        newparts = parts[:5] + parts[6:]
    else:
        # other
        continue
    
    print(f"mv results/{'_'.join(parts)}.csv results/{'_'.join(newparts)}.csv")