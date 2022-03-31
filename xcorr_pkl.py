#Import necessary packages

import numpy as np
from obspy import read, Stream
import pandas as pd
import glob
from scipy import signal
import pickle
import ast

########################
#######################
### LOAD PICKLE #######
#######################
#######################

# Step 1 - Find pickle files - these pickles are the outputs from the prepro_pkl.py script, which relies on functions located within the prepro.py module.

allfiles=glob.glob("SS*.pickle")          # TO BE UPDATED depending on file format in root mseed folder
locs=pd.read_csv('station.txt')            
locs['station'] = locs['station'].astype('str')
#locs['network']=locs['network'].astype('str')

########################
#### XCORR PARAM #######
########################

with open('prepro_para.txt') as f:
    contents = f.read()
    prepro_para = ast.literal_eval(contents)

segwin = prepro_para['segwin']                      # length of segment to stack over(in seconds)
corrwin=prepro_para['corrwin']                      # length of correlation window (in seconds)
freqmin=prepro_para['freqmin']                      # minimum bandpass filter frequency used in prepro_pkl.py [Hz]
freqmax=prepro_para['freqmax']                      # maximum bandpass filter frequency used in prepro_pkl.py [Hz]
sps=500                                             # Sampling rate (in Hertz) --> TO-DO: Add into prepro_para.txt dictionary
corrlen=sps*corrwin                                 # length of correlation window (in number samples)
seglen=sps*segwin                                   #length of window to stack correlations over (in number samples)
ncorr=int(np.floor(segwin/corrwin))                 # number of correlations per segment (stack)


# Next, we want to scroll through the files, only selecting files with "network" and "station" included in name to extract correct files.

#############################
#### OUTER SOURCE LOOP ######
#############################

nsta = len(locs)
#network='SS'                                                          # network is 'SS' for all files

for isrc in range(nsta):                                              #this loop is for the source
  
    source_id=locs.iloc[isrc]['station']
    src_files = [x for x in allfiles if source_id in x]
    
    source=Stream()
    for x in src_files:                                            #read in file corresponding to single station
        source+=read(x)                                            #append trace to stream
    source.sort()                                                  #sort stations by chronological order
    
    # initialize nseg (necessary for np.arrays corr, stack, daystack)
    starttime = source[0].stats.starttime 
    endtime=source[0].stats.endtime
    reltime=endtime-starttime 
    nseg=int(np.floor(reltime/segwin))                     # number of segments (stacks) per trace (day)

############################
### INNER RECEIVER LOOP ####
############################

    for irec in range(nsta):
        receiver_id=locs.iloc[irec]['station']

        if source_id == receiver_id: continue                        # will not calcute autocorr

        # norrow down file list by using sta/net info in the file name
        rec_files = [y for y in allfiles if receiver_id in y]

        receiver=Stream()  
        for y in rec_files:                                         #read in file corresponding to single station
            receiver+=read(y)                                       #append trace to stream        
        receiver.sort()                                             #sort stations by chronological order
            
        #####################
        ### XCORR CALCS #####
        #####################
            
        ## Trim the start and time of each trace in a stream to match both source and receiver
        ndays_source=len(source)
        ndays_receiver=len(receiver)
            
        # Compare numdays_source with numdays_receiver. Use smallest numdays for range.

        if ndays_source < ndays_receiver:
            ndays=ndays_source
        else:
            ndays=ndays_receiver
 
        # array initialization
        corr=np.zeros((ndays,nseg,ncorr,corrlen))            # corr numpy array initialization
        stack=np.zeros((ndays,nseg,corrlen))                 # stack array initialization
        daystack=np.zeros((ndays,corrlen))
        #fullstack=np.zeros((1,corrlen))

        for i in range(ndays):                                   # loop through number of days
            win1=0
            #checks to make sure starttime and endtime of source and receiver match for each day.
            if source[i].stats.starttime < receiver[i].stats.starttime:
                starttime=receiver[i].stats.starttime
            else:
                starttime=source[i].stats.starttime
            if source[i].stats.endtime < receiver[i].stats.endtime:
                endtime=source[i].stats.endtime
            else:
                endtime=receiver[i].stats.endtime
            reltime=endtime-starttime 
            nseg=int(np.floor(reltime/segwin))                   # number of segments (stacks) per trace (day)
            
            source[i].trim(starttime,endtime)
            receiver[i].trim(starttime,endtime) 
            sig1=source[i].data
            sig2=receiver[i].data
            #corrwid=ncorr*nseg

            for j in range(nseg):                                # loop through number of segments within a day (here,segwin=3600 = 1 hr)
                for k in range(ncorr):                           # loop through number of correlations to calculate (here, corrwin=20 sec, ncorr=180 if segwin=3600).   
                    win1+=corrlen
                    win2=win1+corrlen
                    corr[i,j,k,:]=signal.correlate(sig1[win1:win2],sig2[win1:win2],method='fft',mode='same') #1 cross-correlation with length corrlen per for each segment in nseg
      
                stack[i,j,:]=np.mean(corr[i,j,:],axis=0)                                 # stack over full period
                stack[i,j,:]=stack[i,j,:]/float((np.abs(stack[i,j,:]).max()))           # normalize stack
            
            #with open('hourly-xcorr-_src_%s_rec_%s_Day%s.pickle' % (source[isrc].id, receiver[irec].id,i), 'wb') as f:
            #    pickle.dump(stack,f)                         #save stack in a pickle file
                
            daystack[i,:]=np.mean(stack[i,:],axis=0)                                 # stack over full period   
            daystack[i,:]=daystack[i,:]/float((np.abs(daystack[i,:]).max()))           # normalize stack
            #print(daystack[i,:])
        #fullstack=np.mean(daystack,axis=0)
        #fullstack=fullstack/float((np.abs(fullstack).max()))    

        with open('daily-xcorr-'+str(freqmin)+'Hz-'+str(freqmax)+'Hz'+'_src_%s_rec_%s.pickle' % (source[isrc].id, receiver[irec].id), 'wb') as f:
            pickle.dump(daystack,f)                         #save stack in a pickle file
