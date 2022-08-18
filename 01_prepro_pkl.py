#!/usr/bin/env python
# coding: utf-8

#Import necessary packages

import numpy as np
from obspy import Stream, read
import pandas
from obspy import UTCDateTime
import pandas as pd
import prepro as pp
import glob
from scipy import signal
import pickle
from obspy.signal.filter import bandpass
import os,os.path

########################################
### DIRECTORY AND PARAMETER SETTING ####
########################################

#replace this with directory to where your python scripts are stored
projdir=r"E:\MANUSCRIPT1_DATAREPO"
stadir=os.path.join(projdir,"station_all")

# replace this with directory to your raw data files
rawdir=os.path.join(projdir,"rawdata_all")

#change working directory to project directory

os.chdir(stadir)

'''
Trim function:
These parameters indicate how to trim the data. For the cross-correlation function to work,
each trace in the stream must have the same number of samples 
Trim parameters described as follows:

1. trim -------   "y" to specify a start date and end date to process
                  when "y" is selected, "start_trim" and "end_trim" must be updated.
                  "n" to consider all available data (June 17 to ~July 31 2020)
                  when "n" is selected, "start_trim" and "end_trim" are not considered.
                            
2. trimtime ---   "no" (default) to trim data to 11.5 hours so that each trace has same amount of samples
                  "quiet" to trim data from 00:00 UTC to 03:00 UTC (during non-construction period)
                  "noisy" to trim data from 03:00 UTC to 18:30 UTC (during construction)
'''
#Trimming data
trim="n"                          
start_trim=["2020_06_18_15_10_0"] 
end_trim=["2020_06_28_03_0_0"]    
trimtime = "quiet"                                #if "No" then data will include 11.5 hours per trace (almost all samples)                   
trim_para={'trim':trim, 'start_trim':start_trim, 'end_trim':end_trim, 'trimtime':trimtime} 

#Filtering
freqmin=5                    #5Hz min frequency for bandpass filter
freqmax=15                   #50Hz max frequency for bandpass filter

#Decimation factor
dec_factor=5                # raw data at 500Hz, decimate by factor of 10 to 50Hz sampling

#Tapering prior to filter
max_percentage=0.05
max_length=50

#Time-domain normalization
norm_method="1bit"           #options are "1bit" or "rman"
norm_win=0.5                 #for "rman" --> length of normalization window = tr.sampling_rate*norm_win

#Cross-correlation parameters
corrwin=20                              #length of correlation window (in seconds) 
segwin=3600                             # length of segment to stack over(in seconds)
sps=100                                  # Sampling rate after decimation (in Hertz)


#group all parameters into a dictionary
prepro_para={'trim':trim,'start_trim':start_trim,'end_trim':end_trim,'trimtime':trimtime,'freqmin':freqmin,'freqmax':freqmax,'dec_factor':dec_factor,'max_percentage':max_percentage,'max_length':max_length,'norm_method':norm_method,'norm_win':norm_win, 'corrwin':corrwin, 'segwin':segwin,'trimtime':trimtime,'sps':sps}

#save parameters to text file
f = open("prepro_para.txt","w")
f.write( str(prepro_para) )
f.close()


#############################
#############################
#### PRE-PROCESSING #########
#############################
#############################

# find raw data files
#allfiles=glob.glob("?*.miniseed")

#allfiles=glob.glob("?*.miniseed")
allfiles=glob.glob(rawdir+r"\?*.miniseed")
locs=pd.read_csv('station.txt')                  # station.txt file includes all 25 stations. station_beach.txt file only includes stations along tailings beach (6). station_dam.txt file only includes stations along dam crest (19)      
#locs['station'] = locs['station'].astype('str')


nsta = len(locs)                                       # loop through one station at a time
for ista in range(nsta):
    station = locs.iloc[ista]['station']               # the station info:
    # norrow down file list by using sta/net info in the file name
    sta_files = [x for x in allfiles if station[3:] in x] #index [3:] to ignore "SS." at beginning of each station ID
    source=Stream()
    for x in sta_files:        #read in file corresponding to single station
        source+=read(x)        #append trace to stream 
    
    #here is where we would apply the trimming and preprocessing
    source.sort()
    st=source.copy()           #backup copy prior to preprocess
    st=pp.trimit(st,trim_para)
    st.decimate(dec_factor)
    ## Pre-processing Step 1 --> detrend, taper and apply bandpass
    for tr in st:
        tr.detrend('linear')                                                    # remove trends using detrend
        tr.taper(max_percentage=max_percentage,max_length=max_length)           # taper the edges (same parameters as noise_module.preprocess_raw)
        tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True) # filter data of all traces in the streams
## Pre-processing Step 2 --> normalization and spectral whitening
        tr = pp.normalize(tr, norm_win, norm_method)                           #if commented out, no time-domain normalization applied (for noisy data)
        tr = pp.whiten(tr, freqmin,freqmax)
        time=UTCDateTime(tr.stats.starttime)
        date=str(time.year)+"-"+str(time.month)+"-"+str(time.day)
        tr.write(tr.id + "_"+str(freqmin) + "-" + str(freqmax) + "Hz_" + date+".pickle", format="PICKLE") 
