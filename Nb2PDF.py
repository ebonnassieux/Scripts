#!/usr/bin/env python

import os
import sys

if __name__=="__main__":
    NbName=sys.argv[1]

    os.system("ipython nbconvert --to latex  --post PDF --SphinxTransformer.author='Etienne'  %s"%NbName)
