# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:32:31 2015

@author: dennis
"""

import tfnet

#tfnet.train_tfnet('detection', '/home/dennis/CellClassification/', 'wayback', 'default', 'working_folder', True, {'patchflag':False, 'patchsize':None, 'visflag':False})
tfnet.test_tfnet('detection', '/home/dennis/CellClassification/', 'wayback', 'default', 'working_folder', {'patchflag':True, 'patchsize':101, 'visflag':True})
classlabeldic = {0:0, 1:1, 2:2, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
#tfnet.train_tfnet('classification', '/home/dennis/CellClassification/', 'classy', 'default', 'Cell_classification_ROB', True, {'nouts':3, 'labelmaps':classlabeldic})