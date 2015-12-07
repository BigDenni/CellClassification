# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:32:31 2015

@author: dennis
"""

import tfnet

#tfnet.train_tfnet('detection', '/home/dennis/CellClassification/', 'wayback', 'default', 'working_folder', False)
tfnet.test_tfnet('detection', '/home/dennis/CellClassification/', 'wayback', 'default', 'working_folder', True, 100)