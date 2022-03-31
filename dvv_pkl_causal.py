#Import necessary packages

import numpy as np
import numpy.ma as ma
from obspy import read, Stream
from obspy.clients.nrl import NRL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import stretching
from obspy import UTCDateTime
import pandas as pd
import prepro as pp
import glob
from matplotlib import cm
import pickle
import ast

############################
############################
## FIND XCORR PKL FILES ####
############################
############################
station=[]
xcorr_pkldata=glob.glob("*daily-xcorr*.pickle")
locs=pd.read_csv('stationdvv.txt')            
nsta=len(locs)

#climate_data=pd.read_csv("envcanada_rainfall.csv")

######################
######################
### PARAM SETTING ####
######################
######################

with open('prepro_para.txt') as f:
    contents = f.read()
    prepro_para = ast.literal_eval(contents)
    
segwin = prepro_para['segwin']                      # length of segment to stack over(in seconds)
corrwin=prepro_para['corrwin']                      # length of correlation window (in seconds)
freqmin=prepro_para['freqmin']                      # minimum bandpass filter frequency used in prepro_pkl.py [Hz]
freqmax=prepro_para['freqmax']                      # maximum bandpass filter frequency used in prepro_pkl.py [Hz]
trimtime=prepro_para['trimtime']                    # trimtime ("noisy" or "quiet" set in prepro_pkl.py)

sps=500                                             # Sampling rate (in Hertz)
corrlen=sps*corrwin                                 # length of correlation window (in number samples)
seglen=sps*segwin                                   #length of window to stack correlations over (in number samples)
ncorr=int(np.floor(segwin/corrwin))                 # number of correlations per segment (stack)
f.close()

#################################
#################################
### STRETCHING PARAM SETTING ####
#################################
#################################

Epsilon = .10                                       # Stretching between -Epsilon to +Epsilon (multiply by 100 to get the dv in %) STRETCHING LIMITS

#FOR CAUSAL t_ini=0.5 seconds
t_ini = 0.5                                         # Time to start computing the dv/v (in second). Zero is at the zero lag time
t_length = 3                                        # Length of the signal over which the dv/v is computed (in second)
delta=500
t_ini_d = t_ini*delta                               # start dv/v computation at t_ini_d/delta seconds from the signal begining
t_length_d = int(t_length*delta)                    # dv/v computation over t_length_d/delta seconds after t_ini_d/delta

limit=corrwin/2.                                    # set window length in plot (-10 to 10)
timevec = np.arange(-limit, limit, 1/sps)

isrc=0                                              # setting source index at 0
nsrc=25                                             # total number of sources (stations) = 25
nrec=nsrc-1                                         # total number of receivers = nsrc-1 = 24
nfile=nsrc*nrec                                     # total number of station-receiver pairs 
maxdays=60                                          # maximum number of days collecting data over all stations (estimate)
   
dv_arr_unmasked=np.zeros((nfile,maxdays))           # initialize dv/v array
dv_arr=ma.masked_equal(dv_arr_unmasked,0)           # mask zero values
dv_avg_unmasked=np.zeros((nsrc,maxdays))            # initialize average dv/v array. stacks dv/v for 1 source + all receivers
dv_avg=ma.masked_equal(dv_avg_unmasked,0)           # mask zero values
cc_arr_unmasked=np.zeros((nfile,maxdays))
cc_arr=ma.masked_equal(cc_arr_unmasked,0)
error_arr_unmasked=np.zeros((nfile,maxdays))
error_arr=ma.masked_equal(error_arr_unmasked,0)

date = np.array('2020-06-18', dtype=np.datetime64) #Based on trimming, start date should be June 18 2021
datevec=date+np.arange(maxdays)
df=pd.read_csv('stationdvv.txt')        # for calcs of source-receiver distance    

###########################
###########################
### DV/ V################## 
###########################
###########################

for ifile in range(nfile):
    with open(xcorr_pkldata[ifile], 'rb') as f:
        xcorr = pickle.load(f)
    f.close()
    source=xcorr_pkldata[ifile][28:33]               #extract station from filename. format --> daily-xcorr-5Hz-15Hz_src_SS.01313..DPZ_rec_SS.01392..DPZ.pickle
    receiver=xcorr_pkldata[ifile][46:51]             #extract station from filename. format --> daily-xcorr-5Hz-15Hz_src_SS.01313..DPZ_rec_SS.01392..DPZ.pickle
    ndays,x=xcorr.shape

    #source-receiver distance
    src_x=df.loc[df['station'].str.contains(source)].iat[0,0]
    src_y=df.loc[df['station'].str.contains(source)].iat[0,1]
    rec_x=df.loc[df['station'].str.contains(receiver)].iat[0,0]
    rec_y=df.loc[df['station'].str.contains(receiver)].iat[0,1]
    distance=np.sqrt((rec_x-src_x)**2+(rec_y-src_y)**2)
    
    ccf_ref = np.mean(xcorr,axis=0)                         
    ccf_ref /= np.max(np.abs(ccf_ref))
    
    # dvv stretching inputs
    zero_lag_ind = round(len(xcorr[1,:])/2)                         # To compute the dv/v on the causal part of the SC functions.
    window = np.arange(int(t_ini_d+zero_lag_ind), int(t_ini_d+t_length_d+zero_lag_ind))
    
    ## get rainfall data for correct dates from csv
    starttime=17                                                  #sets position in csv to starttime of June 18 2020
    endtime=starttime+maxdays
    #climate_data["date"]=pd.to_datetime(climate_data["date"])
    #rf_data=climate_data.iloc[starttime:endtime, 1].values  

    for k in range(ndays):
        dv, cc, cdp, Eps, error, C=stretching.stretching(ref = ccf_ref, cur = xcorr[k,:], t = timevec, dvmin = -0.05, dvmax = 0.05, nbtrial = 50, window = window, fmin = freqmin, fmax = freqmax, tmin = t_ini_d, tmax = t_length_d+t_ini_d)

    # to remove erroneous data points 
        if cc < 0.7:
            dv=np.nan
                 
        dv_arr[ifile,k]=dv
        cc_arr[ifile,k]=cc
        error_arr[ifile,k]=error
        
    win1=isrc*nrec
    win2=win1+nrec
    
    if ifile == win2-1:
        with open('dvv-arr-causal-%s-%s-%s.pickle' % (source,win1,win2), 'wb') as f:
            pickle.dump(dv_arr,f)
        f.close()
        with open('cc-arr-causal-%s-%s-%s.pickle' % (source,win1,win2), 'wb') as f:
            pickle.dump(cc_arr,f)
        f.close()
        with open('error-arr-causal-%s-%s-%s.pickle' % (source,win1,win2), 'wb') as f:
            pickle.dump(error_arr,f)
        f.close()
        dv_avg[isrc,:]=np.nanmean(dv_arr[win1:win2-1,:],axis=0, dtype=np.float64) 

        with open('dvv-avg-causal-%s-%s.pickle' % (source,isrc), 'wb') as f:
            pickle.dump(dv_avg,f)          #save dv_avg to pickle file
        f.close()
        isrc+=1
