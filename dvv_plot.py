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
## SET PARAMETERS ##########
############################
############################
isrc=0                                  # setting source index at 0
nsrc=25                                 # total number of sources (stations) = 25
nrec=nsrc-1                             # total number of receivers = nsrc-1 = 24
nfile=nsrc*nrec                         # total number of station-receiver pairs 
maxdays=60                              # maximum number of days collecting data over all stations (estimate)

############################
############################
### IMPORT RAINFALL DATA ###
############################
############################

climate_data=pd.read_csv("envcanada_rainfall.csv")

## get rainfall data for correct dates from csv
starttime=17                                                  #sets position in csv to starttime of June 18 2020
endtime=starttime+maxdays
climate_data["date"]=pd.to_datetime(climate_data["date"])
rf_data=climate_data.iloc[starttime:endtime, 1].values 

############################
############################
### IMPORT VWP DATA ########
############################
############################

vw49412=pd.read_csv(r'site_data\vw49412.csv')
vw49412_baro=pd.read_csv(r'site_data\vw49412_baro.csv')

vw49418=pd.read_csv(r'site_data\vw49418.csv')
vw58287=pd.read_csv(r'site_data\vw58287.csv')
vw58288=pd.read_csv(r'site_data\vw58288.csv')

vw49412["Time"]=pd.to_datetime(vw49412["Time"])

vw49412_data=vw49412.iloc[:,1].values
vw49412_baro_data=vw49412_baro.iloc[:,4].values

vw49418_data=vw49418.iloc[:,1].values
vw58287_data=vw58287.iloc[:,1].values
vw58288_data=vw58288.iloc[:,1].values

vw_date=np.array('2020-06-01 00',dtype=np.datetime64) #start date of June 1 2020
vw_nhours=len(vw49412_data)
vw_datevec=vw_date+np.arange(vw_nhours)

############################
############################
## FIND DVV PKL FILES ####
############################
############################

station=[]
dvvavg_pkldata=glob.glob("dvv-avg*.pickle")
dvvarr_pkldata=glob.glob("dvv-arr*.pickle")
ccarr_pkldata=glob.glob("cc-arr*.pickle")
errorarr_pkldata=glob.glob("error-arr*.pickle")

locs=pd.read_csv('stationdvv.txt')            
nsta=len(locs)

# nfile=len(dvvarr_pkldata)
# print(nfile)

# df=pd.read_csv('stationdvv.txt')            
# print(df.head())
# source=df[['station']]
# print(station)
    

# xcorr_pkldata=glob.glob("*daily-xcorr*.pickle")
# nfile=len(xcorr_pkldata)
# for ifile in range(nfile):
#     with open(xcorr_pkldata[ifile], 'rb') as f:
#         source=xcorr_pkldata[ifile][24:29]               #extract station from filename 
#         receiver=xcorr_pkldata[ifile][42:47]              #extract station from filename 

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
trimtime=prepro_para['trimtime']                   # trimtime ("noisy" or "quiet" set in prepro_pkl.py)
f.close()

#%% Stretching parameters

Epsilon = .10  # Stretching between -Epsilon to +Epsilon (multiply by 100 to get the dv in %) STRETCHING LIMITS
t_ini = 0.5      # Time to start computing the dv/v (in second). Zero is at the zero lag time
t_length = 3   # Length of the signal over which the dv/v is computed (in second)
sps=500                                             # Sampling rate (in Hertz)
delta=sps
t_ini_d = t_ini*delta             # start dv/v computation at t_ini_d/delta seconds from the signal begining
t_length_d = int(t_length*delta)  # dv/v computation over t_length_d/delta seconds after t_ini_d/delta


#############################
#############################
### PLOT DV/V vs RAINFALL ###
#############################
#############################

#dv/v plotting
date = np.array('2020-06-18', dtype=np.datetime64) #Based on trimming, start date should be June 18 2021
datevec=date+np.arange(maxdays)

for ifile in range(len(dvvavg_pkldata)):

    win1=isrc*nrec
    win2=win1+nrec
#for ifile in range(1):
    with open(dvvavg_pkldata[ifile],'rb') as f:
        dv_avg=pickle.load(f)
    with open(dvvarr_pkldata[ifile],'rb') as f:
        dv_arr=pickle.load(f)
    with open(ccarr_pkldata[ifile],'rb') as f:
        cc_arr=pickle.load(f)
    with open(errorarr_pkldata[ifile],'rb') as f:
        error_arr=pickle.load(f)

        source=dvvavg_pkldata[ifile][8:13]               #extract station from filename 

        plt.rcParams["figure.figsize"]=(16,16)
        fig,(ax1,ax3,ax4)=plt.subplots(nrows=3, ncols=1,sharex=True)

        for i in range(win1,win2-1,1):
            ax1.plot(datevec,dv_arr[i,:],'-k',alpha=0.2)
        ax1.plot(datevec,dv_avg[ifile,:],'-ro',label='Mean stack of all receiver pairs')
        ax1.set_title("dv/v plot for Source: %s and all Receivers, Coda window: %0.2f-%0.2f, %s - %s Hz, Trim: %s" % (source, t_ini,t_ini+t_length,freqmin, freqmax, trimtime),fontsize=18)
        ax1.set_ylim(-2,2)
        ax1.set_ylabel("dv/v (%)",fontsize=16)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_xlim('2020-06-17','2020-08-05')
        ax1.grid()
        
        ax2=ax1.twinx()
        ax2.set_ylabel("Elevation (m)",fontsize=16)
        ax2.set_ylim(332.5,333.5)
        ax2.plot(vw49412["Time"],vw49412_baro_data,'-b',alpha=0.5,label='VW49412 (corrected for baro) [m]')
        #ax2.plot(vw_datevec,vw49418_data,label="Tip elev. at 329.6 m (tailings)")
        #ax2.plot(vw_datevec,vw58287_data,label="Tip elev. at 319.9 m (GLU)")
        #ax2.plot(vw_datevec,vw58288_data,label="Tip elev. at 329.6 m (tailings)")
        ax2.legend(fontsize=14)
        ax2.tick_params(axis='y', labelsize=14) 
        ax2.set_yticks(np.arange(332.5,333.75,step=0.25))

         #Error and CC
        for i in range(win1,win2-1,1):
            ax3.plot(datevec,cc_arr[i,:],'-k',alpha=0.2)
        ax3.set_title("CC plot for Source: %s and all Receivers, Coda window: %0.2f-%0.2f, %s - %s Hz, Trim: %s" % (source, t_ini,t_ini+t_length,freqmin, freqmax, trimtime),fontsize=18)
        ax3.set_ylim(0,1)
        ax3.set_xlim('2020-06-17','2020-08-05')
        ax3.set_ylabel("Correlation Coefficient",fontsize=16)
        ax3.tick_params(axis='y', labelsize=14) 
        ax3.grid() 
        #ax3.invert_yaxis()

        #ax4=ax3.twinx()
        for i in range(win1,win2-1,1):
            ax4.plot(datevec,error_arr[i,:],'-r',alpha=0.2)
        ax4.set_ylim(0,0.00012)
        ax4.set_xlim('2020-06-17','2020-08-05')
        #ax4.legend(fontsize=14)
        ax4.grid()
        ax4.set_title("Error plot for Source: %s and all Receivers, Coda window: %0.2f-%0.2f, %s - %s Hz, Trim: %s" % (source, t_ini,t_ini+t_length,freqmin, freqmax, trimtime),fontsize=18)
        ax4.tick_params(axis='y', labelsize=14) 
        ax4.set_ylabel("Error",fontsize=16)
##        
##        #pandas rainfall data
##        ax3.bar(datevec,rf_data, label="Rainfall [mm]")
##        ax3.set_ylabel("Total Rainfall [mm]",fontsize=16)
##        ax3.set_xlabel("Date [yyyy-mm-dd]",fontsize=16)
##        ax3.grid()
##        ax3.legend(fontsize=14)
##        ax3.tick_params(axis='x', labelsize=14) 
##        ax3.tick_params(axis='y', labelsize=14) 

        plt.savefig('dvv_vwp_ccerror_%s_%s-%sHz_tr-%s.png' % (source, freqmin, freqmax, trimtime))
        #plt.savefig('dvv_vwp_ccerror_%s_5-15Hz_tr-%s.png' % (source, trimtime))

        isrc+=1
