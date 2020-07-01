#GitHub: JeaustinSirias
'''
This function is made to import 
dekadals .csv files to Python.
It doesnt matter how many stations/poligons it contains.

The way to import a file is:

input_data('file_name.cvs')

'''
##########
#PACKAGES#
##########
import pandas as pd
import numpy as np

def input_data(input_d):
	data = pd.read_csv(input_d, header = None,)
	df = pd.DataFrame(data)
	return np.array(df.loc[:, 1:])


