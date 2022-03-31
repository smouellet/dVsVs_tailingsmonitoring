#!/usr/bin/env python
# coding: utf-8

#Import necessary packages

import numpy as np
from obspy import read, read_inventory
from obspy.clients.nrl import NRL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime
from numba import jit
from scipy import signal
import pickle 

nrl = NRL()

'''
This python script contains all the necessary functions used to pre-process the data and compute cross-correlations.

Pre-processing includes:
1. Detrend
2. Taper
3. Apply bandpass filter
4. Cut start and end of trace to match start and end of station pair

Correlation includes:
1. Compute cross-correlation over specified correlation window
2. Stack over 1-hr windows
3. Daystack over multiple days
'''


# Functions
    
'''
    Source: https://github.com/scipy/scipy/blob/v0.16.0/scipy/stats/stats.py#L1963
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.
    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.
    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
'''

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def normalize(tr, norm_win, norm_method): 
    if norm_method == 'ramn':
        lwin = tr.stats.sampling_rate * norm_win
        st=int(0)                                             # starting point
        N=int(lwin)                                           # ending point
        
        while N < tr.stats.npts:
            win = tr.data[st:N]
            w = np.mean(np.abs(win)) / (2 * lwin + 1)
            
            # weight center of window
            tr.data[int(st + lwin / 2)] /= w
            
            # shift window
            st += 1
            N += 1

        # taper edges
        taper = get_window(tr.stats.npts)
        tr.data *= taper

    elif norm_method == "1bit":
        tr.data = np.sign(tr.data)
        tr.data = np.float32(tr.data)

    return tr

def get_window(N, alpha=0.2):
    #@njit
    
    window = np.ones(N)
    x = np.linspace(-1., 1., N)
    ind1 = (abs(x) > 1 - alpha) * (x < 0)
    ind2 = (abs(x) > 1 - alpha) * (x > 0)
    window[ind1] = 0.5 * (1 - np.cos(np.pi * (x[ind1] + 1) / alpha))
    window[ind2] = 0.5 * (1 - np.cos(np.pi * (x[ind2] - 1) / alpha))
    return window


def whiten(tr, freqmin, freqmax):
    #@njit
    
    nsamp = tr.stats.sampling_rate
    n = len(tr.data)
    if n == 1:
        return tr
    else: 
        frange = float(freqmax) - float(freqmin)
        nsmo = int(np.fix(min(0.01, 0.5 * (frange)) * float(n) / nsamp))
        f = np.arange(n) * nsamp / (n - 1.)
        JJ = ((f > float(freqmin)) & (f<float(freqmax))).nonzero()[0]
            
        # signal FFT
        FFTs = np.fft.fft(tr.data)
        FFTsW = np.zeros(n) + 1j * np.zeros(n)

        # Apodization to the left with cos^2 (to smooth the discontinuities)
        smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo+1))**2)
        FFTsW[JJ[0]:JJ[0]+nsmo+1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0]+nsmo+1]))

        # boxcar
        FFTsW[JJ[0]+nsmo+1:JJ[-1]-nsmo] = np.ones(len(JJ) - 2 * (nsmo+1))        * np.exp(1j * np.angle(FFTs[JJ[0]+nsmo+1:JJ[-1]-nsmo]))

        # Apodization to the right with cos^2 (to smooth the discontinuities)
        smo2 = (np.cos(np.linspace(0, np.pi/2, nsmo+1))**2)
        espo = np.exp(1j * np.angle(FFTs[JJ[-1]-nsmo:JJ[-1]+1]))
        FFTsW[JJ[-1]-nsmo:JJ[-1]+1] = smo2 * espo

        whitedata = 2. * np.fft.ifft(FFTsW).real
        
        tr.data = np.require(whitedata, dtype="float32")

        return tr


def trimit(st,trim_para):
   
    '''
    Function to trim data to select intervals and ensure source and receiver have same number samples.

    Calling this function will result in:
    - Removal of the first and last trace of each station for more consistent number samples in each trace.
    - Comparison of start and end times of each trace to confirm same number of samples in source / receiver. Otherwise
      data will be trimmed.
  
    Trim options:

    1. "all": all traces available will be included in processing.
    2. "n": data will be trimmed to dates specified in "start_trim" and "end_trim"
    3. "quiet": data will be trimmed to only include 00:00UTC to 03:00 UTC, during which no construction takes place.
    4. "noisy": data will be trimmed to only include 15:00 UTC to 00:00 UTC, during which construction takes place.

    '''
    #load parameters first
    trim = trim_para['trim']
    trimtime=trim_para['trimtime']
    start_trim = trim_para['start_trim'] #only used if trim!="all"
    end_trim = trim_para['end_trim']     #only used if trim!="all"

    stations = list(set([_i.stats.station for _i in st]))

#### PRELIMINARY TRIMMING ####

# Remove the first and last trace of each station stream.
    st.remove(st.select(station=stations[0])[0])        # first trace of stream
    st.remove(st.select(station=stations[0])[-1])       # last trace of stream 
    

#### TRIM OPTIONS #####
    start_trim=UTCDateTime(start_trim[0])
    end_trim=UTCDateTime(end_trim[0])

    if trim == "y":
        st.trim(start_trim,end_trim)

    if trimtime=="quiet":
        numdays=len(st.select(station=stations[0]))
    
        for i in range(numdays):
            t2=st.select(station=stations[0])[i].stats.endtime
            t1=t2-3*3600                                       #trim data to approximately 00:00 to 03:00 UTC
            st.select(station=stations[0])[i].trim(t1,t2)
    if trimtime=="quiet" and trim=="y":
        st.remove(st.select(station=stations[0])[-1])       # remove last trace of stream because may not be during quiet period

    if trimtime=="noisy":
        numdays=len(st.select(station=stations[0]))
  
        for i in range(numdays):
            end=st.select(station=stations[0])[i].stats.endtime
            t1=end-11.5*3600                                       #trim data to approximately 15:30 to 00:00 UTC
            t2=end-3*3600
            st.select(station=stations[0])[i].trim(t1,t2)
       
    else:
        numdays=len(st.select(station=stations[0]))
   
        for i in range(numdays):
            t2=st.select(station=stations[0])[i].stats.endtime
            t1=t2-11.5*3600                                       #trim data to approximately 00:00 to 03:00 UTC
            st.select(station=stations[0])[i].trim(t1,t2)
    return st

def corr(st,corrwin,para):
    '''
    Function to calculate cross-correlations between data in stream, then stack over each day of data.
    corrwin = length of correlation window (in seconds)
    ndays = number of traces for each station in stream
    corrlen=length of correlation window (in number samples)
    
    '''
    
    #load parameters first
    freqmin= para['freqmin']
    freqmax= para['freqmax']
    corrwin=para['corrwin']
    segwin=para['segwin']
    trimtime=para['trimtime']

    stations = list(set([_i.stats.station for _i in st]))
    
    # Trim the start and time of each trace in a stream to match both source and receiver
    numdays_source=len(st.select(station=stations[0]))
    numdays_receiver=len(st.select(station=stations[1]))
   
    # Compare numdays_source with numdays_receiver. Use smallest numdays for range.
    if numdays_source < numdays_receiver:
        ndays=numdays_source
    else:
        ndays=numdays_receiver
    starttime = st.select(station=stations[1])[0].stats.starttime 
    endtime=st.select(station=stations[1])[0].stats.endtime
    reltime=endtime-starttime 

    sps=int(st[0].stats.sampling_rate)                   # sampling rate (should be 500Hz if no decimation)
    corrlen=sps*corrwin                                   # length of correlation window (in number samples)
    seglen=sps*segwin                                     #length of window to stack correlations over (in number samples)

    ncorr=int(np.floor(segwin/corrwin))                   # number of correlations per segment (stack)
    nseg=int(np.floor(reltime/segwin))                     # number of segments (stacks) per trace (day)
    corrwid=ncorr*nseg

    corr=np.zeros((ndays,nseg,ncorr,corrlen))
    stack=np.zeros((ndays,nseg,corrlen))
    daystack=np.zeros((ndays,corrlen))
    fullstack=np.zeros((1,corrlen))

    for i in range(ndays):
        sig1=st.select(station=stations[0])[i].data
        sig2=st.select(station=stations[1])[i].data
        #sig1=st.select(station=stations[1])[i].data #THIS IS A TEST TO MAKE SURE SOURCE & RECEIVER ARRIVAL TIME IS REVERSED
        #sig2=st.select(station=stations[0])[i].data
        win1=0
        for j in range(nseg):
            for k in range(ncorr):
                win1+=corrlen
                win2=win1+corrlen
                corr[i,j,k,:]=signal.correlate(sig1[win1:win2],sig2[win1:win2],mode='same') #1 cross-correlation with length corrlen per for each segment in nseg
            stack[i,j,:]=np.sum(corr[i,j,:],axis=0)                                 # stack over full period
            stack[i,j,:]=stack[i,j,:]/float((np.abs(stack[i,j,:]).max()))           # normalize stack
        daystack[i,:]=np.sum(stack[i,:],axis=0)                                 # stack over full period   
        daystack[i,:]=daystack[i,:]/float((np.abs(daystack[i,:]).max()))           # normalize stack
    fullstack=np.sum(daystack,axis=0)
    fullstack=fullstack/float((np.abs(fullstack).max()))

     #***************************************************************************************************************
    #Plot Day Stack - One plot per station pair

    limit=corrwin/2.
    plt.matshow(daystack[:,:],cmap='seismic',extent=[-limit, limit,ndays,0],aspect='auto')
    timevec = np.arange(-limit, limit, st[0].stats.delta)
    plt.plot(timevec, fullstack,'k',label='Full stack')
    plt.xlabel("Time (sec)")
    plt.ylabel("No. of days")
    plt.title("Daily correlation stacks between %s and %s, %s - %s Hz, Trim: %s" % (stations[0], stations[1], freqmin, freqmax, trimtime))
    plt.xlim(-2,2)
    plt.legend()

    #save figure as png file
    plt.savefig('daily_stacks_%s_%s_%s-%sHz_tr-%s.png' % (stations[0], stations[1], freqmin, freqmax, trimtime))

    #****************************************************************************************************************

    #Plot Hourly Stacks - One plot per day per station pair
    limit=corrwin/2.
    for i in range(ndays):
        plt.matshow(stack[i,:,:],cmap='seismic',extent=[-limit, limit,nseg,0],aspect='auto')
        timevec = np.arange(-limit, limit, st[0].stats.delta)
        plt.plot(timevec, daystack[i,:],'k',label='Day stack')
        plt.xlabel("Time (sec)")
        plt.ylabel("No. of hours")
        plt.title("Stacked correlation between %s and %s, %s - %s Hz, Day %s, Trim: %s" % (stations[0], stations[1], freqmin, freqmax, i, trimtime))
        plt.xlim(-2,2)

        #save figure as png file
        plt.savefig('hourlystacks_%s_%s_%s-%sHz_Day%s_tr-%s.png' % (stations[0], stations[1], freqmin, freqmax, i,trimtime))   
  
    #Plot the summed stack over the entire dataset for one station pair.

    limit=corrwin/2.
    timevec = np.arange(-limit, limit, st[0].stats.delta)
    plt.figure(figsize=(16,4))
    plt.plot(timevec, fullstack, 'k-',label='Full stack over %s days' %(ndays))
    plt.title("Stacked correlation between %s and %s, %s - %s Hz, Trim: %s" % (stations[0], stations[1], freqmin, freqmax, trimtime))
    plt.legend()
    plt.xlim(-2,2)

    #save figure as png file
    plt.savefig('Full_stack_%s_%s_%s-%sHz_tr-%s.png' % (stations[0], stations[1], freqmin, freqmax, trimtime))

    with open('hourlystacks_%s_%s_%s-%sHz_Day%s_tr-%s.pickle' % (stations[0], stations[1], freqmin, freqmax, i,trimtime), 'wb') as f:
        pickle.dump(stack,f)                         #save stack in a pickle file
    with open('Day_stack_%s_%s_%s-%sHz_tr-%s.pickle' % (stations[0], stations[1], freqmin, freqmax, trimtime), 'wb') as f:
        pickle.dump(daystack,f)                         #save stack in a pickle file
   
    return stack, daystack, fullstack
   