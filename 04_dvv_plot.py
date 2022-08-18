#Import necessary packages

import numpy as np
import numpy.ma as ma
from obspy import read, Stream
from obspy.clients.nrl import NRL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
from obspy import UTCDateTime
import pandas as pd
import prepro as pp
import glob
from matplotlib import cm
import pickle
import ast
import os,os.path

#replace this with directory to where your python scripts are stored
projdir=r"E:\MANUSCRIPT1_DATAREPO"
stadir=os.path.join(projdir,"station_all")

# replace this with directory to your raw data files
rawdir=os.path.join(projdir,"rawdata_all")
figdir=os.path.join(stadir,"figures")
sitedatadir=os.path.join(projdir,"site_data")

#change working directory to project directory

os.chdir(stadir)
locs=pd.read_csv('station.txt')   # station.txt file includes all 25 stations. station_beach.txt file only includes stations along tailings beach (6). station_dam.txt file only includes stations along dam          


############################
############################
## SET PARAMETERS ##########
############################
############################
isrc=0                                  # setting source index at 0
nsrc=len(locs)                          # total number of sources (stations) = 25. For only geophones in beach, then nsrc!=25
nrec=nsrc-1                             # total number of receivers = nsrc-1 = 24
#nfile=nsrc*nrec                         # total number of station-receiver pairs 
maxdays=60                              # maximum number of days collecting data over all stations (estimate)

######################################
######################################
### IMPORT ENVIRONMENTAL SITE DATA ###
######################################
######################################

#rainfall from environment canada weather station
climate_data=pd.read_csv(sitedatadir+r'\envcanada_rainfall.csv')

#tailings pond station data
pp_station=pd.read_csv(sitedatadir+r'\pp_station.csv')       # baro, surface temp (in datalogger enclosure), pond sensor temp, pond sensor level from 2020-06 to 2020-09
pp_data=pp_station.iloc[:,4].values            # pond level data (in m)
baro=pp_station.iloc[:,1].values               # barometric pressure data (in m)
temp=pp_station.iloc[:,2].values               # temperature data (panel temp, in deg C)
surface_temp=pp_station.iloc[:,2].values
pp_station["Time"]=pd.to_datetime(pp_station["Time"])

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

vw49412=pd.read_csv(sitedatadir+r'\vw49412.csv')
vw49412_baro=pd.read_csv(sitedatadir+r'\vw49412_baro.csv')
vw49418=pd.read_csv(sitedatadir+r'\vw49418.csv')
vw58287=pd.read_csv(sitedatadir+r'\vw58287.csv')
vw58288=pd.read_csv(sitedatadir+r'\vw58288.csv')
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
dvvavg_caus_pkldata=glob.glob("dvv-avg-causal*.pickle")
dvvarr_caus_pkldata=glob.glob("dvv-arr-causal*.pickle")
ccarr_caus_pkldata=glob.glob("cc-arr-causal*.pickle")
errorarr_caus_pkldata=glob.glob("error-arr-causal*.pickle")

dvvavg_acaus_pkldata=glob.glob("dvv-avg-acausal*.pickle")
dvvarr_acaus_pkldata=glob.glob("dvv-arr-acausal*.pickle")
ccarr_acaus_pkldata=glob.glob("cc-arr-acausal*.pickle")
errorarr_acaus_pkldata=glob.glob("error-arr-acausal*.pickle")

#locs=pd.read_csv('station_beach.txt')            
nsta=len(locs)
#nfile=len(dvvavg_pkldata)

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
sps=prepro_para['sps']                              # Sampling rate (in Hertz)
f.close()

###%% Stretching parameters

Epsilon = .10  # Stretching between -Epsilon to +Epsilon (multiply by 100 to get the dv in %) STRETCHING LIMITS
t_ini = 0.5      # Time to start computing the dv/v (in second). Zero is at the zero lag time
t_length = 3   # Length of the signal over which the dv/v is computed (in second)
delta=sps
t_ini_d = t_ini*delta             # start dv/v computation at t_ini_d/delta seconds from the signal begining
t_length_d = int(t_length*delta)  # dv/v computation over t_length_d/delta seconds after t_ini_d/delta

#################################
#################################
### PLOT DV/V WITH CC AND ERR ###
#################################
#################################


#dv/v plotting
fsize=18 # fontsize


date = np.array('2020-06-18', dtype=np.datetime64) #Based on trimming, start date should be June 18 2021
datevec=date+np.arange(maxdays)
isrc=0
for ifile in range(nsta):

    #casual data -> coda window from 0.5 to 3.5 sec
    with open(dvvavg_caus_pkldata[ifile],'rb') as f:
        dv_avg_causal=pickle.load(f)
    with open(dvvarr_caus_pkldata[ifile],'rb') as f:
        dv_arr_causal=pickle.load(f)
    with open(ccarr_caus_pkldata[ifile],'rb') as f:
        cc_arr_causal=pickle.load(f)
    with open(errorarr_caus_pkldata[ifile],'rb') as f:
        error_arr_causal=pickle.load(f)
    #acausal data -> coda window from -3.5 to -0.5 sec
    with open(dvvavg_acaus_pkldata[ifile],'rb') as f:
        dv_avg_acausal=pickle.load(f)
    with open(dvvarr_acaus_pkldata[ifile],'rb') as f:
        dv_arr_acausal=pickle.load(f)
    with open(ccarr_acaus_pkldata[ifile],'rb') as f:
        cc_arr_acausal=pickle.load(f)
    with open(errorarr_acaus_pkldata[ifile],'rb') as f:
        error_arr_acausal=pickle.load(f)

    dv_avg=ma.masked_equal(np.nanmean((dv_avg_causal,dv_avg_acausal),axis=0),0)
    dv_arr=ma.masked_equal(np.nanmean((dv_arr_causal,dv_arr_acausal),axis=0),0)
    cc_arr=ma.masked_equal(np.nanmean((cc_arr_causal,cc_arr_acausal),axis=0),0)
    error_arr=ma.masked_equal(np.nanmean((error_arr_causal,error_arr_acausal),axis=0),0)


    source=dvvavg_caus_pkldata[ifile][15:20]               #extract station from filename 

    plt.rcParams["figure.figsize"]=(20,16)
    fig,(ax1,ax3)=plt.subplots(nrows=2, ncols=1,sharex=True)
    #print(win1,win2)
    win1=isrc*nrec
    win2=win1+nrec

    # AX1, UPPER PLOT --> DVV DATA
    
    for i in range(win1,win2):
        ax1.plot(datevec,dv_arr[i,:],'-k',alpha=0.5)
    ax1.plot(datevec,dv_avg[isrc,:],'-k',linewidth=4,label='Mean stack of all receiver pairs')
    ax1.set_ylim(-1,1)
    ax1.set_ylabel("dv/v (%)",fontsize=fsize)
    ax1.set_xlabel("Date [yyyy-mm-dd]",fontsize=fsize)
    ax1.set_xlim('2020-06-20','2020-08-05')
    ax1.legend(fontsize=fsize,loc=1)
    ax1.tick_params(axis='x',labelsize=18)

    ax1.set_title("dv/v plot for Source: %s and all Receivers, Coda window: %0.2f-%0.2f, %s - %s Hz, Trim: %s" % (source, t_ini,t_ini+t_length,freqmin, freqmax, trimtime),fontsize=18)
    ax1.set_ylim(-1,1)
    ax1.set_ylabel("dv/v (%)",fontsize=fsize)
    ax1.tick_params(axis='y', labelsize=fsize)
    #ax1.set_xlim('2020-06-17','2020-08-05')
    ax1.grid()
    
    # AX2, UPPER PLOT --> INVERTED POND ELEVATION
    ax2=ax1.twinx()
    ax2.set_ylabel("Inverted Pond Elevation (m)",fontsize=fsize)
    ax2.set_ylim(337.25,339.25)
    ax2.plot(pp_station['Time'],pp_station['Primary Pond Level[m]'],'-b',alpha=0.5,label='Inverted Pond Elevation (m)')
    #ax2.legend(fontsize=fsize)
    ax1.tick_params(axis='y', labelsize=fsize) 
    ax2.tick_params(axis='y', labelsize=fsize,colors='mediumblue') 
    ax1.set_xlim('2020-06-20','2020-08-05')
    ax2.spines['right'].set_color('mediumblue')
    ax2.yaxis.label.set_color('mediumblue')
    ax2.invert_yaxis()

    # AX3, LOWER PLOT --> CORRELATION COEFFICIENT (CC)

    for i in range(win1,win2-1,1):
        ax3.plot(datevec,cc_arr[i,:],'-k',alpha=0.5)
    ax3.set_title("CC and Error plot for Source: %s and all Receivers, Coda window: %0.2f-%0.2f, %s - %s Hz, Trim: %s" % (source, t_ini,t_ini+t_length,freqmin, freqmax, trimtime),fontsize=18)
    ax3.set_ylim(0,1)
    ax3.set_xlim('2020-06-20','2020-08-05')
    ax3.set_ylabel("Correlation Coefficient",fontsize=fsize)
    ax3.tick_params(axis='y', labelsize=fsize)
    ax3.tick_params(axis='x',labelsize=fsize)
    ax3.grid() 

    # AX6, LOWER PLOT, RIGHT HAND SIZE --> ERROR
    ax6=ax3.twinx()

    for i in range(win1,win2-1,1):
        ax6.plot(datevec,error_arr[i,:],'-m',alpha=0.5)
    ax6.set_ylim(0.0,0.005)
    ax6.set_xlim('2020-06-20','2020-08-05')
    ax6.legend(fontsize=14)
    ax6.grid()
    #ax6.set_title("Error plot for Source: %s and all Receivers, Coda window: %0.2f-%0.2f, %s - %s Hz, Trim: %s" % (source, t_ini,t_ini+t_length,freqmin, freqmax, trimtime),fontsize=18)
    ax6.tick_params(axis='y', labelsize=fsize) 
    ax6.tick_params(axis='x', labelsize=fsize) 
    ax6.set_ylabel("Error",fontsize=fsize)
    ax6.spines['right'].set_color('darkmagenta')
    ax6.tick_params(axis='y',labelsize=fsize,colors='darkmagenta')
    ax6.yaxis.label.set_color('darkmagenta')

    plt.legend(fontsize=fsize)
    #plt.show()
#        plt.savefig('dvv_ccerror_%s_%s-%sHz_tr-%s.png' % (source, freqmin, freqmax, trimtime))
    plt.savefig(figdir+r'\dvv_pond_ccerror_%s.png' % (source),bbox_inches='tight') #add bbox argument so axes are not cut off

    isrc+=1


##############################################
### PLOT AVERAGE DV/V WITH POND AND BARO #####

# define arrays for minimum dvv and max dvv, for plotting purposes
dv_avgmin=np.zeros(len(datevec))
dv_avgmax=np.zeros(len(datevec))

for i in range(len(datevec)):
    dv_avgmin[i]=np.nanmin(dv_avg[:,i]) # minimum dvv array
    dv_avgmax[i]=np.nanmax(dv_avg[:,i]) # maximum dvv array


#plot figure including all data (environment & dvv)

plt.rcParams["figure.figsize"]=(20,16)
fig,(ax1,ax3)=plt.subplots(nrows=2, ncols=1,sharex=True)
fsize = 18                      # font size for all labels in plott

# for i in range(nsta):
#     ax1.plot(datevec,dv_avg[i,:],'-k',alpha=0.3)
dv_avg_tot=np.nanmean(dv_avg[:],axis=0)
ax1.plot(datevec,dv_avg_tot,'-k',linewidth=6,label='Average dv/v (%) measurements over geophone array')
#ax1.plot(datevec,dv_avg_tot,'-k',linewidth=1, alpha=0.5,label='dv/v (%) measurements per source averaged over all receivers')
ax1.fill_between(datevec, dv_avgmin, dv_avgmax,color='k',alpha=0.1)
#ax1.plot(pp_station["Time"],dVsVs,'-r',linewidth=6,label='Modelled dVs/Vs changes at a depth of 10 m')
#ax1.plot(vs_array[:,0],vs_array[:,-1],'-r',linewidth=6,label='Modelled dVs/Vs changes at a depth of 10 m')

ax1.set_ylim(-1,1)
ax1.set_ylabel("dv/v (%)",fontsize=fsize)
ax1.set_xlabel("Date [yyyy-mm-dd]",fontsize=fsize)
ax1.set_xlim('2020-06-20','2020-08-05')
ax1.legend(fontsize=fsize,loc=1)
ax1.tick_params(axis='x',labelsize=18)

ax1.grid()

ax2=ax1.twinx()
ax2.set_ylabel("Inverted Pond Elevation (m)",fontsize=fsize)
ax2.set_ylim(337.25,339.25)
ax2.plot(pp_station['Time'],pp_station['Primary Pond Level[m]'],'-b',alpha=0.5,label='Inverted Pond Elevation (m)')
#ax2.legend(fontsize=fsize)
ax1.tick_params(axis='y', labelsize=fsize) 
ax2.tick_params(axis='y', labelsize=fsize,colors='mediumblue') 
ax2.set_xlim('2020-06-21','2020-08-05')
ax2.spines['right'].set_color('mediumblue')
ax2.yaxis.label.set_color('mediumblue')

ax2.invert_yaxis()

# #pandas rainfall data
ax3.bar(datevec,rf_data, label="Rainfall (mm)")
ax3.set_ylim(0,80)
ax3.set_ylabel("Daily Rainfall (mm)",fontsize=fsize)
ax3.set_xlabel("Date [yyyy-mm-dd]",fontsize=fsize)
ax3.grid()
ax3.set_xlim('2020-06-21','2020-08-05')
ax3.tick_params(axis='y', labelsize=fsize)
ax3.tick_params(axis='x',labelsize=fsize)
ax3.legend(fontsize=fsize)
# # Barometric pressure
ax6=ax3.twinx()
ax6.plot(pp_station['Time'],pp_station['Barometric Pressure[mH20]'],'-m',alpha=0.5,label="Barometric Pressure (m)")
ax6.spines['right'].set_position(("axes", 1.1))
ax6.spines['right'].set_color('darkmagenta')
ax6.tick_params(axis='y',labelsize=fsize,colors='darkmagenta')
ax6.yaxis.label.set_color('darkmagenta')

ax6.set_ylim(9,11)
ax6.yaxis.set_ticks_position('none')
ax6.set_ylabel("Barometric Pressure (m)",fontsize=fsize)
#ax6.legend(fontsize=fsize,loc=2)
ax6.tick_params(axis='x',labelsize=18)


ax4=ax3.twinx()
ax4.set_ylabel("Pond Elevation (m)",fontsize=fsize)
ax4.set_ylim(337.25,339.25)
ax4.plot(pp_station['Time'],pp_station['Primary Pond Level[m]'],'-b',alpha=0.5,label='Pond Elevation (m)')
#ax4.legend(fontsize=fsize)
ax4.tick_params(axis='y', labelsize=fsize,colors='mediumblue')
ax4.set_xlim('2020-06-21','2020-08-02')
ax4.tick_params(axis='x', labelsize=fsize) 
ax4.spines['right'].set_color('mediumblue')
ax4.yaxis.label.set_color('mediumblue')

#plt.legend(fontsize=fsize)
#plt.show()
plt.savefig(figdir+r'\dvvavg_pond_ccerror.png',bbox_inches='tight') #add bbox argument so axes are not cut off)


###########################
###########################
### PLOTTING XCORR ONLY ###
###########################
###########################

corrwin=20            # correlation window length 20 seconds
limit=corrwin/2.      # set window length in plot (-10 to 10)
freqmin=5             # minimum frequency [hz]
freqmax = 15          # maximum frequency [hz]
trimtime = 'quiet'    # trim time
timevec = np.arange(-limit, limit, 1/sps)

xcorr_pkldata=glob.glob("*daily-xcorr*.pickle")

for i in range(len(xcorr_pkldata)):
#for i in range(40):
    with open(xcorr_pkldata[i], 'rb') as f:
        xcorr = pickle.load(f)
    source=xcorr_pkldata[i][28:33]               #extract station from filename 
    receiver=xcorr_pkldata[i][46:51]              #extract station from filename 
    ndays,x=xcorr.shape
      
    ccf_ref = np.mean(xcorr,axis=0)                         
    ccf_ref /= np.max(np.abs(ccf_ref))
    timevec = np.arange(-limit, limit, 1/sps)


    #Plot Day Stack - One plot per station pair
    plt.rcParams["figure.figsize"]=(16,4)
    fig,(ax1,ax2)=plt.subplots(2)

    ax1.matshow(xcorr[:,:],cmap='seismic',extent=[-limit,limit,ndays,0],aspect='auto',vmin=-1,vmax=1)
    ax1.set_title("Daily correlation stacks between Source: SS.%s and Receiver: SS.%s, %s - %s Hz, Trim: %s" % (source, receiver, freqmin, freqmax, trimtime))
    ax1.set_ylabel("No. of days")
    ax1.set_xlim(-5,5)
    ax1.vlines(x=[-t_ini,-(t_ini+t_length),t_ini,t_ini+t_length],ymin=0,ymax=ndays,color='k', linestyle='dashed', linewidth=1)

    ax2=plt.subplot(212,sharex=ax1)
    ax2.plot(timevec, ccf_ref,'k',label='Full stack')
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylim(-1,1)
    ax2.vlines(x=[-t_ini,-(t_ini+t_length),t_ini,t_ini+t_length],ymin=-1,ymax=1,color='k', linestyle='dashed', linewidth=1)

    plt.legend()
    plt.savefig(figdir+r'\daily_xcorr_%s_%s_%s-%sHz_tr-%s.png' % (source, receiver, freqmin, freqmax, trimtime))
