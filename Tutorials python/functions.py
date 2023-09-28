def LoaddataNav(path, params = None):
    """Loads the behavioral data into a MATLAB structure, using parameters
    defined in loadparams to find the raw data file and preprocess the data.
    Typically, the output structure should contain a field called sampleTimes
    containing the timestamps of each time sample and a set of fields
    containing data which will be used as independent/explanatory variables 
    along which neural responses are investigated.
     
    INPUT:
    - loadparams: a structure whose fields contain the parameters necessary
    to load the behavioral data data and preprocess them.
    See SetLoadParams.m for a description of these parameters.
     
    OUTPUT:
    - Nav: a MATLAB structure whose fields contain different types of
    behavioral data resampled at the desired sampling rate (defined in
    loadparams.samplingRate).
     
    Fields of Nav are the following:
    * sampleTimes: time stamps of the samples for behavioral data
    * X: positions of the animal on the X axis
    * Y: position of the animal on the Y axis
    * Xpos: position along the X axis in percentage of the track length.
    * XDir: direction of movement along the X axis (+1 for left to right, -1
      for right to left)
    * Spd: speed of movement
    * smthSpd: speed smoothed by a box car window
    * Condition: experimental condition corresponding to different 
      subsessions (1:preprun, 2:presleep, 3:run, 4:postsleep, 5:postrun)
    * laptype: +1 if the animal went from the left to the right platform;
      -1 if it went from the right one to the left one;
      0 if it went back to the same platform before reaching the
      end of the track
    * uturn: +1 on trial where the animal returned to the same platform; 0 otherwise
    * trialID: trial number. Trial were defined as any continuous period of
      time where the animal was on the track
    * reward: +1 when a reward was delivered; 0 otherwise.
    * airpuff: +1 when an air puff was delivered; 0 otherwise.
    * state: +1 for awake; 0 for drowsy; -1 for REM sleep; -2 for slow wave sleep
    * acc: 3-axis accelerometer data (ntimes x 3 array)

    All fields of Nav have time along the first dimension.
    
    USAGE:
    datadirpath = <path to the directory containing your data>
    loadparams = SetLoadParams(datadirpath);
    Nav = LoaddataNav(loadparams);
     
    See also: SetLoadParams, LoadEvents, LoaddataSpk, LoaddataLfp
     
    Written by J.Fournier in 08/2023 for the Summer school "Advanced
    computational analysis for behavioral and neurophysiological recordings"
    Adapted by Tulio Almeida """
    import os
    import mat73
    import scipy.io
    import numpy as np
    import pandas as pd
    from scipy import signal,stats,sparse,interpolate

    if params is None:
        params =  DefineLoadParams()

    # files paths
    evt_files = {filename.split('.')[-2]:os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.evt')}
    mat_files = {filename.split('.')[-2]:os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.mat')}

    # loading position data
    pos = scipy.io.loadmat(mat_files['Positions'])

    # Converting positions from pixels to cm
    pos_dict = {'x':pos['positions'][:,1] * params['pix2cm'],
                'y':pos['positions'][:,2] * params['pix2cm']}

    # Original timestamps of sampling for the tracked positions
    pos_dict['sampleTimes'] = pos['positions'][:,0]

    # loading subsessions indices (pre-run, run etc).
    # 'timestamps' contains the start and end indices of each subsession;
    # 'description' contains a txt descriptor of the type of subsession.
    cat = pd.read_table(evt_files['cat'], header = None)

    # Information about types of trial, i.e. whether the animal is going for the right platform or the left one.
    LapType = mat73.loadmat(mat_files['LapType'])

    # air puffs timestamps
    puf = pd.read_table(evt_files['puf'], header = None)
    puf = puf[0].values/1000

    # loading right/left reward times
    rrw = pd.read_table(evt_files['rrw'], header = None)
    rrw = rrw[0].values/1000
    lrw = pd.read_table(evt_files['lrw'], header = None)
    lrw = lrw[0].values/1000

    # Filling in Nav.Condition vector with 1 for preprun, 2 for presleep, 3 for run, 4 for postsleep, 5 for postrun
    pos_dict['Condition'] = np.empty((len(pos['positions'][:,1])))*np.nan
    condID = np.repeat(np.arange(1,6,1),2)
    events = {info.split(' ')[-1].split('-')[-1]+'-'+info.split(' ')[1]:[float(info.split(' ')[0])/1000,condID[idx]] for idx,info in enumerate(cat[0])}
    for idx in range(0,len(events),2):
        condidx = (pos_dict['sampleTimes'] >= list(events.values())[idx][0]) & (pos_dict['sampleTimes'] <= list(events.values())[idx+1][0])
        pos_dict['Condition'][condidx] = list(events.values())[idx][1]

    # Computing Speed and direction of mvt.

    # Initializing the corresponding fields
    pos_dict['Spd'] = np.empty((len(pos['positions'][:,1])))*np.nan # Movement speed
    pos_dict['smthSpd'] = np.empty((len(pos['positions'][:,1])))*np.nan # Smoothed speed
    pos_dict['XDir'] =  np.empty((len(pos['positions'][:,1])))*np.nan # +1 for Left to Right; -1 for Right to Left.

    # Sampling rate of the tracking data.
    sampleRate = 1/np.mean(np.diff(pos_dict['sampleTimes']))

    # Smoothing window (used for smoothing speed and estimating direction along X).
    smthwin = np.round(sampleRate * .5).astype(int)

    # Speeds and directions need to be computed for each sessions seperately
    # because recordings from different subsessions are discontinous.

    for icond in set(condID): # pre-run, run, etc
        # Running speed
        end = np.array([pos_dict['sampleTimes'][pos_dict['Condition']==icond][-1] - pos_dict['sampleTimes'][pos_dict['Condition']==icond][-2]])
        Xdiff = np.concatenate((np.diff(pos_dict['x'][pos_dict['Condition']==icond]), np.array([np.nan])))
        Ydiff = np.concatenate((np.diff(pos_dict['y'][pos_dict['Condition']==icond]), np.array([np.nan])))
        Tdiff = np.concatenate((np.diff(pos_dict['sampleTimes'][pos_dict['Condition']==icond]),end))
        spd_temp = np.divide(np.sqrt(Xdiff**2 + Ydiff**2),Tdiff)
        pos_dict['Spd'][pos_dict['Condition']==icond] = spd_temp

        # Running speed smoothed by a box car window of length smthwin bins
        pos_dict['smthSpd'][pos_dict['Condition']==icond] = signal.savgol_filter(spd_temp,smthwin,1)

        # Speed along X
        Xspd =  signal.savgol_filter(np.divide(Xdiff,Tdiff),smthwin,1)
        XDirLtoR = signal.savgol_filter(Xspd > 0, smthwin,0) > 0
        XDirRtoL = signal.savgol_filter(Xspd < 0, smthwin,0) > 0
        pos_dict['XDir'][pos_dict['Condition']==icond] = (XDirLtoR*1 - XDirRtoL*1 > 0)*1 - (XDirLtoR*1 - XDirRtoL*1 < 0)*1

    # Interpolating Nav.XDir = 0 values to nearest non-zero value for convenience.
    pos_dict['XDir'] = interpolate.interp1d(pos_dict['sampleTimes'][pos_dict['XDir'] !=0],
                                            pos_dict['XDir'][pos_dict['XDir'] !=0],
                                            kind='nearest')(pos_dict['sampleTimes'])
    
    # Resampling behavioral data to the final resolution

    # original timestamps
    sampleTimes_orig = pos_dict['sampleTimes']

    # new timestamps corresponding to a sampling rate of loadparams.sampleRate
    sampleTimes_new = np.arange(min(sampleTimes_orig),max(sampleTimes_orig),1/params['sampleRate'])

    # Interpolation at new query time points of X, Y, Spd, etc
    pos_dict['x'] = interpolate.interp1d(sampleTimes_orig,pos_dict['x'],kind='linear')(sampleTimes_new)
    pos_dict['y'] = interpolate.interp1d(sampleTimes_orig,pos_dict['y'],kind='linear')(sampleTimes_new)
    pos_dict['Spd'] = interpolate.interp1d(sampleTimes_orig,pos_dict['Spd'],kind='linear')(sampleTimes_new)
    pos_dict['XDir'] = interpolate.interp1d(sampleTimes_orig,pos_dict['XDir'],kind='nearest')(sampleTimes_new)
    pos_dict['smthSpd'] = interpolate.interp1d(sampleTimes_orig,pos_dict['smthSpd'],kind='linear')(sampleTimes_new)
    pos_dict['Condition'] = interpolate.interp1d(sampleTimes_orig,pos_dict['Condition'],kind='nearest')(sampleTimes_new)

    # Replacing the pos_dict.sampleTime with the new ones.
    pos_dict['sampleTimes'] = sampleTimes_new

    # Filling in pos_dict.airpuff with 1 when there is an air puff
    pos_dict['airpuff'] = np.zeros(pos_dict['x'].shape)

    for idx in range(len(puf)):
        rewidx = np.argmin(abs(pos_dict['sampleTimes'] - puf[idx]))
        pos_dict['airpuff'][rewidx] = 1

    # Filling in pos_dict.reward with 1 for right reward and -1 for left reward
    pos_dict['reward'] = np.zeros(pos_dict['x'].shape)

    for idx in range(len(lrw)):
        rewidx = np.argmin(abs(pos_dict['sampleTimes'] - lrw[idx]))
        pos_dict['reward'][rewidx] = -1

    for idx in range(len(rrw)):
        rewidx = np.argmin(abs(pos_dict['sampleTimes'] - rrw[idx]))
        pos_dict['reward'][rewidx] = 1

    # pos_dict.laptype equals 1 or -1 for left to right and right to left trials respectively.
    pos_dict['laptype'] = np.zeros(pos_dict['x'].shape)
    pos_dict['uturn'] = np.zeros(pos_dict['x'].shape)

    for idx in range(len(LapType['LtoRlaps'])):
        idx = (pos_dict['sampleTimes'] >= LapType['LtoRlaps'][idx,0]) & (pos_dict['sampleTimes'] <= LapType['LtoRlaps'][idx,1])
        pos_dict['laptype'][idx] = 1

    for idx in range(len(LapType['RtoLlaps'])):
        idx = (pos_dict['sampleTimes'] >= LapType['RtoLlaps'][idx,0]) & (pos_dict['sampleTimes'] <= LapType['RtoLlaps'][idx,1])
        pos_dict['laptype'][idx] = -1

    # Defining start and end positions of the linear track (Nav.laptype = 0 when the animal is not on the track).
    Xtrackstart = min(pos_dict['x'][pos_dict['laptype'] != 0])
    Xtrackend = max(pos_dict['x'][pos_dict['laptype'] != 0])

    # Calculating Xpos as a percentage of the track length.
    pos_dict['Xpos'] = 100 * (pos_dict['x'] - Xtrackstart) / (Xtrackend - Xtrackstart)

    # Replacing with NaNs values where the animal is out of the linear track.
    pos_dict['Xpos'][(pos_dict['Xpos'] < 0) | (pos_dict['Xpos'] > 100)] = np.nan

    # New - check - TODO
    pos_dict['Xpos'][~np.isin(pos_dict['Condition'], [1,3,5])] = np.nan

    # Defining some trials indices for whenever the animal is on the track and
    # spend more than 0.5 second there.
    # Looking for potential start and end of trials
    trialduration_th = 0.5
                
    trialStart = np.argwhere(np.sign(np.diff(~np.isnan(pos_dict['Xpos'][:-1])*1).astype(int)) > 0) +1
    trialEnd = np.argwhere(np.sign(np.diff(~np.isnan(pos_dict['Xpos'])*1).astype(int))<0)

    # if the animal is already on the track at the beginning, we modify trialStart accordingly
    if trialEnd[0] < trialStart[0]:
        trialStart = np.concatenate((np.zeros((1,1)),trialStart)).astype(int)

    # %if the recording is stopped while the animal is on the track, we modify trialEnd accordingly.
    if len(trialEnd) < len(trialStart):
        trialEnd = np.concatenate((trialEnd , len(pos_dict['Xpos']))).astype(int)

    trialStart = trialStart.reshape(-1)
    trialEnd = trialEnd.reshape(-1)

    # Initializing the vector of trialIDs
    pos_dict['trialID'] = np.empty(pos_dict['Xpos'].shape) * np.nan

    # Checking that the trials are valid (i.e. longer than 1 s)
    trialnum = 0
    for k in range(len(trialStart)):
        if pos_dict['sampleTimes'][trialEnd[k]] - pos_dict['sampleTimes'][trialStart[k]] > trialduration_th:
            trialnum = trialnum + 1
            pos_dict['trialID'][trialStart[k]:trialEnd[k]] = trialnum

    # loading eeg states timestamps (wake, rem, nrem) and filling in Lfp.eegstate with 1 for wake, 0 for drowsy,
    # -1 for REM sleep and -2 for non-REM sleep.

    pos_dict['state'] = np.zeros(pos_dict['x'].shape)
    state = scipy.io.loadmat(mat_files['States'])

    for idx in range(len(state['wake'])):
        sidx = (pos_dict['sampleTimes'] >= state['wake'][idx,0]) & (pos_dict['sampleTimes'] <= state['wake'][idx,1])
        pos_dict['state'][sidx] = 1

    for idx in range(len(state['drowsy'])):
        sidx = (pos_dict['sampleTimes'] >= state['drowsy'][idx,0]) & (pos_dict['sampleTimes'] <= state['drowsy'][idx,1])
        pos_dict['state'][sidx] = 0

    for idx in range(len(state['sws'])):
        sidx = (pos_dict['sampleTimes'] >= state['sws'][idx,0]) & (pos_dict['sampleTimes'] <= state['sws'][idx,1])
        pos_dict['state'][sidx] = -2

    for idx in range(len(state['Rem'])):
        sidx = (pos_dict['sampleTimes'] >= state['Rem'][idx,0]) & (pos_dict['sampleTimes'] <= state['Rem'][idx,1])
        pos_dict['state'][sidx] = -1

    # Loading accelerometer data

    acc = mat73.loadmat(mat_files['LFP'], only_include='acc')
    acc = acc['acc']

    # Sample times for the accelerometer

    accsampleTimes = acc[:,0]
    sampleTimes_orig = accsampleTimes
    sampleTimes_new = pos_dict['sampleTimes']

    # Resampling the 3 accelerometer channels at the final sampling rate

    pos_dict['acc_x'] = interpolate.interp1d(sampleTimes_orig,acc[:,1],kind='linear')(sampleTimes_new)
    pos_dict['acc_y'] = interpolate.interp1d(sampleTimes_orig,acc[:,2],kind='linear')(sampleTimes_new)
    pos_dict['acc_z'] = interpolate.interp1d(sampleTimes_orig,acc[:,3],kind='linear')(sampleTimes_new)

    return pos_dict

def LoaddataSpk(path, Nav, params = None):
    """
    Load spiking data into Spk structure, resampled according to timestamps
    provided in sampleTimes and other parameters defined in loadparams. See
    DefineLoadParams.m for a description of those parameters.

    INPUT:
    - loadparams:  a structure whose fields contain the parameters necessary
    to load the spiking data data and resample them.
    See DefineLoadParams.m for a description of these parameters.
    - Nav: navigation data

    OUTPUT:
    - Spk: a matlab structure whose fields contain the different types of
    behavioral data resampled at the desired sampling rate (defined in
    loadparams.samplingRate).

    Fields of Nav are the following:
    - sampleTimes: time stamps of the resampled spike trains
    - spikeTimes: a ntimes x 1 array of all spike times
    - spikeID: cluster IDs of spikes in spikeTimes
    - spikeTrain: a nTimes x nCells array of spike counts in bins centered
    around sample times of Spk.sampleTimes
    - shankID: 1 x nCells array of ID of the shank where each cluster was recorded
    - PyrCell: 1 x nCells logical array. true if the cluster is a putative Pyramidal neuron
    - IntCell: 1 x nCells logical array. true if the cluster is a putative Interneuron
    - hpcCell: 1 x nCells logical array. true if the cluster is in hpc
    - blaRCell: 1 x nCells logical array. true if the cluster is in right bla
    - blaLCell: 1 x nCells logical array. true if the cluster is in left bla
    - ripple: ntimes x 1 array with ones wherever there is a ripple.
    - ripplepeak: ntimes x 1 array with ones for ripple peaks.
    - rippleTimes: timestamps of the detected ripple peaks (in seconds)

    USAGE:
    Spk = LoaddataSpk(path, Nav)

    written by J.Fournier 08/2023 for the iBio Summer school
    Adapted by Tulio Almeida"""
    import os
    import mat73
    import scipy.io
    import numpy as np
    import pandas as pd
    from scipy import signal,stats,sparse,interpolate

    if params is None:
        params =  DefineLoadParams()

    # files paths
    evt_files = {filename.split('.')[-2]:os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.evt')}
    mat_files = {filename.split('.')[-2]:os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.mat')}

    # loading spike times and cluster ID from the prepared .mat file
    spk = scipy.io.loadmat(mat_files['AllSpikes'])

    # Removing spikes that are before or after behavior started
    extraspk = (spk['AllSpikes'][:,0] < Nav['sampleTimes'][0]) | (spk['AllSpikes'][:,0] > Nav['sampleTimes'][-1])
    spk['AllSpikes'] = np.delete(spk['AllSpikes'],extraspk,0)

    # keep only cells that were either in hpc or bla
    # Saving some cluster info into the Spk structure
    spkinfo = scipy.io.loadmat(mat_files['IndexType'])
    spkinfo = spkinfo['IndexType']
    hpcblaClustidx = np.isin(spkinfo[:,2],params['ShankList']) #  params['ShankList'] + params['ShankList_blaL'] + params['ShankList_blaR']

    # Saving spike times and cluster IDs.
    spk_dict = {'spikeTimes':spk['AllSpikes'][:,0],
                'spikeID':spk['AllSpikes'][:,1]}

    # Keeping only cells in hpc or bla
    goodspkidx = np.isin(spk_dict['spikeID'],np.argwhere(hpcblaClustidx))
    spk_dict['spikeTimes'] = spk_dict['spikeTimes'][goodspkidx]
    spk_dict['spikeID'] = spk_dict['spikeID'][goodspkidx]

    # convert spike times into an array of spike trains, sampled according to sampleTimes.
    clustList = np.unique(spk_dict['spikeID']).astype(int)
    ncells = len(clustList)
    nTimeSamples = len(Nav['sampleTimes'])
    sampleRate = 1 / np.mean(np.diff(Nav['sampleTimes']))

    spk_dict['spikeTrain'] = np.zeros((nTimeSamples, ncells))
    spk_dict['spikeTimes'] = [[] for cell in range(ncells)]
    binEdges = np.concatenate((Nav['sampleTimes'],np.array([max(Nav['sampleTimes']) + 1/sampleRate]))) 

    for icell in range(len(clustList)):
        spk_dict['spikeTimes'][icell].extend(spk['AllSpikes'][spk['AllSpikes'][:,1] == clustList[icell]][:,0])
        spk_dict['spikeTrain'][:,icell] = np.histogram(spk_dict['spikeTimes'][icell],binEdges)[0]

    spk_dict['sampleTimes'] = Nav['sampleTimes']

    # Saving some cluster info into the Spk structure
    # HippoClustidx = np.isin(spkinfo[:,2],params['ShankList'])
    spk_dict['shankID'] = spkinfo[hpcblaClustidx,2]
    spk_dict['PyrCell'] = (spkinfo[hpcblaClustidx,5] == 1)*1
    spk_dict['IntCell'] = (spkinfo[hpcblaClustidx,5] == 2)*1
    spk_dict['hpcCell'] = np.isin(spk_dict['shankID'],params['ShankList'])
    spk_dict['blaLCell'] = np.isin(spk_dict['shankID'],params['ShankList_blaL'])
    spk_dict['blaRCell'] = np.isin(spk_dict['shankID'],params['ShankList_blaR'])

    # loading ripples timestamps and filling in Spk.ripple with 1 when there
    # is a ripple and Spk.ripplepeak with 1 for the peak of each ripple
    spk_dict['ripple'] = np.zeros((len(Nav['sampleTimes']),1))
    spk_dict['ripplepeak'] = np.zeros((len(Nav['sampleTimes']),1))

    rip = pd.read_table(evt_files['rip'],header = None)
    ripstart = rip[np.isin(rip[1].values, 'Ripple start 23')][0].values/1000
    ripstop = rip[np.isin(rip[1].values, 'Ripple stop 23')][0].values/1000
    rippeak = rip[np.isin(rip[1].values, 'Ripple peak 23')][0].values/1000

    for idx in range(len(ripstart)):
        rippleidx = (Nav['sampleTimes'] >= ripstart[idx]) & (Nav['sampleTimes'] <= ripstop[idx])
        spk_dict['ripple'][rippleidx] = 1
        rippeakidx = np.argmin(abs(Nav['sampleTimes'] - rippeak[idx]))
        spk_dict['ripplepeak'][rippeakidx] = 1

    spk_dict['rippleTimes'] = rippeak

    return spk_dict

def SetMapsParams(Nav, Spk):
    """Define a set of parameters needed to compute place fields.
     
    INPUTS:
    - Nav: a structure with at least a field called sampleTimes containing
      sample times of the data from which maps will be computed.
    - Spk: a structure with at least a field called spikeTrain containing the
      time sereis of responses that will be mapped.
     
    OUTPUT:
    - mapsparams: a structure whose fields contain parameters to run
      MapsAnalysis1D and MapsAnalysis2D.
     
     Fields of mapsparams are the following:
     condition: experimental conditions over which place fields will be estimated.
     dir: An array representing the directions over which place fields will be estimated.
    laptype: An array defining lap types over which place fields will be estimated.
    spdthreshold: The minimum speed threshold (in units of velocity) over 
        which place fields will be computed.
    cellidx: A logical array indicating a subset of cells for which place 
        fields will be computed.
    sampleRate: The sampling rate of the data (in Hz).
    scalingFactor: A scaling factor applied to the response data, typically 
        set to 1 / samplingRate to convert spiking data to spikes per second.
    Xvariablename: The name of the independent variable used to map the 
        response along the X-axis.
    Xrange: The range of X values over which place fields will be estimated.
    Xbinsize: The size of the X-axis bins.
    Xsmthbinsize: The size of the Gaussian window for smoothing along the X-axis.
    XsmthNbins: The number of bins used for smoothing place fields along the X-axis.
    Xbinedges: The edges of position bins used to discretize the X-axis.
    Yvariablename: The name of the independent variable used to map the 
        response along the Y-axis.
    Yrange: The range of Y values over which place fields will be estimated.
    Ybinsize: The size of the Y-axis bins.
    Ysmthbinsize: The size of the Gaussian window for smoothing place fields along the Y-axis.
    YsmthNbins: The number of bins used for smoothing place fields along the Y-axis.
    Ybinedges: The edges of position bins used to discretize the Y-axis.
    occ_th: An occupancy threshold in seconds above which positions are included in the place field estimate.
    nspk_th: The minimal number of spikes required to consider a cell.  
    nShuffle: The number of shuffle controls performed for randomization.
    kfold: The number of folds considered for cross-validation. 
    
    USAGE:
    mapsparams = DefineMapsParams(Nav,Spk)
 
    written by J.Fournier 08/2023 for the iBio Summer school
    Adapted by Tulio Almeida"""

    import numpy as np
    # Conditions over the fields of Nav for which place fields will be estimated
    # mapsparams.subset should be a structure where fields have names of the 
    # fields of Nav to which the condition should apply to.
    mapsparams = {'subset':{}}

    # For instance, for the example data set, we define the following fields
    mapsparams['subset']['Condition'] = [1,3,5]
    mapsparams['subset']['Condition_op'] = np.isin

    mapsparams['subset']['XDir'] = [-1,1]
    mapsparams['subset']['XDir_op'] = np.isin

    mapsparams['subset']['laptype'] = [-1,0,1]
    mapsparams['subset']['laptype_op'] = np.isin

    mapsparams['subset']['Spd'] =  2.5
    mapsparams['subset']['Spd_op'] = np.greater_equal

    # Subset of cells for which place fields will be computed
    mapsparams['cellidx'] = np.ones((1, Spk['spikeTrain'].shape[1])).astype(bool)[0]

    # Sampling rate of the data
    mapsparams['sampleRate'] = 1 / np.nanmean(np.diff(Nav['sampleTimes']))

    # Scaling factor on the response data (default is 1 / samplingRate so that
    # spiking data are returned in spike / s)
    mapsparams['scalingFactor'] = 1 / mapsparams['sampleRate']

    # Name of the independent variable used to map the response along X. Default is Xpos
    mapsparams['Xvariablename'] = 'Xpos'

    # Edges of position bins used to discretize X
    mapsparams['Xbinedges'] = np.arange(0,104,4)

    # Size of the gaussian window for smoothing place fields along X (in bins).
    mapsparams['XsmthNbins'] = 1

    # Name of the independent variable used to map the response along Y. Default is XDir.
    mapsparams['Yvariablename'] = 'XDir'

    # Size of the gaussian window for smoothing place fields along Y (in bins).
    mapsparams['YsmthNbins'] = 0

    # Edges of Y bins used to discretize Y
    mapsparams['Ybinedges'] = [-2,0,2]

    # Occupancy threshold above which positions are included in the place field estimate (in seconds)
    mapsparams['occ_th'] = 0

    # Minimal number of spikes to consider a cell
    mapsparams['nspk_th'] = 0

    # Number of shuffle controls to perform for randomization
    mapsparams['nShuffle'] = 100

    # Number of folds to consider for cross-validation
    mapsparams['kfold'] = 10

    return mapsparams

def DefineLoadParams():
    params = {'sampleRate': 50,
            'sampleRate_rawLfp': 600,
            'pix2cm': 0.4300,
            'ShankList': [1,2,3,4],
            'ShankList_blaL': [5,6,7,8],
            'ShankList_blaR': [13,14,15,16,17,18,19],
            'LfpChannel_Hpc': 2,
            'LfpChannel_Bla': 2,
            'LfpChannel_Acc': 2,
            'ThetaBand': [6,9]}

    return params

def Compute1DMap(Xd,Z, nbins):
    """Compute 1D map efficiently. Xd is the binned independent varaible and Z is
    the dependent variable.

    Written by J. Fournier in August 2023 for the iBio Summer School.
    adapted by Tulio Almeida"""

    import numpy as np
    from scipy import sparse  
    # Selecting valid indices
    valididx = ((np.invert(np.isnan(Xd))) & (np.invert(np.isnan(Z))))
    Xd = Xd[valididx]
    Z = Z[valididx]

    # Summing Z within indices of Xd.
    # Converting into a full matrix again for accessing elements more conveniently

    return sparse.csr_matrix( ( Z, (np.zeros(Xd.shape),Xd-1) ),shape = (1,nbins) ).toarray()

def Compute2DMap(Xd, Yd, Z, nXbins, nYbins):
    """Compute 2D map efficiently. Xd is the binned independent varaible and Z is
    the dependent variable.

    Written by J. Fournier in August 2023 for the iBio Summer School.
    adapted by Tulio Almeida"""

    import numpy as np
    from scipy import sparse  

    # Selecting valid indices
    valididx = ((np.invert(np.isnan(Xd))) & (np.invert(np.isnan(Yd))))
    Xd = Xd[valididx]
    Yd = Yd[valididx]
    Z = Z[valididx]

    # Summing Z within indices of Xd.
    # Converting into a full matrix again for accessing elements more conveniently

    return sparse.csr_matrix( ( Z ,( Xd-1, Yd-1 ) ), shape = (nXbins,nYbins) ).toarray()

def ComputeMap(Xd = None,Yd = None, Z = None, nXbins = None, nYbins = None):
    """ComputeMap - Compute a map efficiently by accumulating Z into values of Xd and Yd.

    map = ComputeMap(Xd, Yd, Z, nXbins, nYbins) efficiently computes a two-dimensional
    map by accumulating values of the dependent variable Z into the bins of the binned
    independent variables Xd and Yd. It utilizes sparse matrix operations for faster computation.
    
    INPUTS:
    - Xd: The binned independent variable along the X-axis. If empty, computes a 1D map along Y.
    - Yd: The binned independent variable along the Y-axis. If empty, computes a 1D map along X.
    - Z: The dependent variable to be accumulated into Xd and Yd bins.
    - nXbins: Scalar, number of bins in Xd for accumulating Z values.
    - nYbins: Scalar, number of bins in Yd for accumulating Z values.
    
    OUTPUT:
    - map: An nYbins x nXbins array of Z values summed into bins of Xd and Yd.
    
    USAGE:
    map = ComputeMap(Xd, Yd, Z, nXbins, nYbins);
    
    SEE ALSO:
    GaussianSmooth, MapsAnalyses
    
    Written by J. Fournier in 08/2023 for the Summer school 
    Advanced computational analysis for behavioral and neurophysiological recordings
    Adapted by Tulio Almeida"""

    # If Yd is empty, we will compute a 1D map along X
    if Yd is None and nYbins is None:
        map = Compute1DMap(Xd, Z, nXbins)
    # If Xd is empty, we will compute a 1D map along Y
    elif Xd is None and nXbins is None:
        map = Compute1DMap(Yd, Z, nYbins)
    # Else, compute the 2D map
    else:
        map = Compute2DMap(Xd, Yd, Z, nXbins,nYbins)

    return map
    

def ComputeTriggeredAverage(R, S, idxwin, w = None):
    """Computes a triggered average of vector R based on timestamps in S over a
    window of indices idxwin, weighted by w.

    INPUTS:
    - R: 1D array from which to compute the average.
    - S: list of indices from which to extract value of R.
    - idxwin: list of indices around values of S.
    - w (optional): list of the same size as S to weight R's snippets before averaging.

    OUTPUTS:
    - m: average of R triggered on indices in S on a window defined by idxwin.
    - sd: standard deviation of the average m.
    - r: snippets of R triggered on S. Each line correspond to one snippet.
    
    written by J.Fournier 08/2023 for the iBio Summer school
    adapted by Tulio Almeida """

    import numpy as np
    # Padding R
    idxmax = max(abs(idxwin))
    Rc = np.concatenate( (np.empty((idxmax))*np.nan, R, np.empty((idxmax))*np.nan) )

    # %Extracting snippets of R around timestamps in S
    r = np.array([Rc[x + idxwin] for idx,x in enumerate(S + idxmax)])
    if w is not None:
        r = np.multiply(r , w)

    # Average across snippets
    m = np.mean(r,axis=0)

    # s.d. across snippets
    sd = np.std(r,axis=0, ddof=1)

    return m, sd, r

def GaussianSmooth1D(inputs, smthNbins):
    """GaussianSmooth1D - Smooth a 1D vector with a Gaussian kernel.

    output = GaussianSmooth1D(input, smthNbins)
    GaussianSmooth1D function applies Gaussian smoothing to a 1D input array using a
    Gaussian kernel.

    INPUTS:
    input:      1D array to be smoothed.
    smthNbins:  Standard deviation of the Gaussian kernel.

    OUTPUT:
    output:     Smoothed 1D array of the same size as the input.

    USAGE:
    output = GaussianSmooth1D(input, smthNbins);

    SEE ALSO:
    GaussianSmooth, Compute1DMap, Compute2DMap, MapsAnalyses1D,MapsAnalyses2D

    Written by J. Fournier in August 2023 for the iBio Summer School.
    adapted by Tulio Almeida """
    import numpy as np
    from scipy import signal

    # Saving sise of input to reshape it at the end
    sz = inputs.shape

    # Building the gaussian function that'll be used to smooth the data
    npts = 5
    x = np.arange(-(npts * smthNbins),(npts * smthNbins)+1)
    Smoother = np.exp((-1*np.power(x,2) / (np.power(smthNbins,2)) /2))
    Smoother = Smoother / np.sum(Smoother)

    # Detecting NaN values to exclude them from the convolution
    valididx = np.invert(np.isnan(inputs))

    # Replacing NaN values with 0 so they don't count when convolving
    inputs[np.isnan(inputs)] = 0

    # Convolving the input vector with the gaussian
    output = signal.convolve(inputs.reshape(-1), Smoother, mode = 'same')

    # Counting the actual number of valid points smoothed (i.e. excluding NaN values).
    flat = signal.convolve(valididx.reshape(-1), Smoother, mode = 'same')

    # Normalizing the convolved vector by the number of valid points
    output = np.divide(output,flat)

    # Replacing back NaN values in the output vector.
    output[np.invert(valididx).reshape(-1)] = np.nan

    # Reshaping the output vector as originally.
    output = output.reshape(sz)

    return output

def GaussianSmooth2D(input, smthNbins):
    import numpy as np
    from scipy.signal import convolve

    # If input is 1D, ensure it is a column vector
    sz0 = input.shape

    if np.sum(len(sz0) > 1) == 1:
        input = input.reshape(-1, 1)
        smthNbins = [max(smthNbins), 0]

    # Building the Gaussian function that will be used to smooth the data
    Ndim = input.ndim
    npts = np.full(Ndim, 5)
    npts[smthNbins == 0] = 0

    Smoother = np.ones(np.round(2 * npts * np.maximum(smthNbins, 1) + 1).astype(int))
    for k in range(Ndim):
        x = np.arange(-npts[k] * max(1, smthNbins[k]), npts[k] * max(1, smthNbins[k]) + 1)
        if smthNbins[k] > 0:
            Smoother_1D = np.exp(-x**2 / (2 * smthNbins[k]**2))
        else:
            Smoother_1D = np.double(x == 0)
        Smoother_1D = Smoother_1D.reshape(-1, 1)
        vperm = np.arange(Ndim)
        vperm[[0, k]] = vperm[[k, 0]]
        Smoother_1D = np.transpose(Smoother_1D, vperm)
        sz = np.asarray(Smoother.shape)
        sz[k] = 1
        Smoother = Smoother * np.tile(Smoother_1D, sz)

    Smoother = Smoother / np.sum(Smoother)

    # Detecting NaN values to exclude them from the convolution
    valididx = ~np.isnan(input)

    # Replacing NaN values with 0 so they don't count when convolving
    input[np.isnan(input)] = 0

    # Convolving the input vector with the Gaussian
    output = convolve(input, Smoother, mode='same')

    # Counting the actual number of valid points smoothed (excluding NaN values)
    flat = convolve(valididx.astype(np.double), Smoother, mode='same')

    # Normalizing the convolved vector by the number of valid points
    output = output / flat

    # Replacing back NaN values in the output vector
    output[~valididx] = np.nan

    # Reshaping output to the original size of input
    output = output.reshape(sz0)
    
    return output

def GaussianSmooth(arr, smthbins = list):
    """GaussianSmooth - Smooth a nD array with a Gaussian kernel.
    
       output = GaussianSmooth(input, smthNbins)
       GaussianSmooth applies Gaussian smoothing to an n-dimensional input array
       using a Gaussian kernel.
    
     INPUTS:
     - input: n-dimensional array to be smoothed.
     - smthNbins: Standard deviations of the Gaussian kernel for each dimension in
     a list.
    
     OUTPUT:
     - output: Smoothed n-dimensional array of the same size as the input.
    
     USAGE:
     output = GaussianSmooth(input, smthNbins);
    
     SEE ALSO:
     ComputeMap, MapsAnalyses
    
     Written by J. Fournier in 08/2023 for the Summer school
     Advanced computational analysis for behavioral and neurophysiological recordings
     Adapted by Tulio Almeida"""
    import numpy as np
    if np.isin(0,smthbins):
        output = GaussianSmooth1D(arr,max(smthbins))
    else:
        output = GaussianSmooth2D(arr,smthbins)

    return output

def getSpatialinfo(t, o):
    """getSpatialinfo computes the spatial information in a tuning curve.

    SInfoperspike = getSpatialinfo(t, o) computes the spatial information in a tuning curve t,
    considering an occupancy o. The result is returned in bits per spike.

    INPUTS:
    - t: Tuning curve (often representing firing rates).
    - o: Occupancy map corresponding to the tuning curve.

    OUTPUT:
    - SInfoperspike: Spatial information in bits per spike.
    - SInfo: Spatial information in bits per second.

    USAGE:
    SInfoperspike = getSpatialinfo(t, o);

    Written by J. Fournier in 08/2023 for the Summer school
    Advanced computational analysis for behavioral and neurophysiological recordings
    Adapted by Tulio Almeida"""
    import numpy as np
    # Computes the spatial information in tuning curve t, considering an
    # occupancy o. This is returned in bits per spike.
    # Excluding values where either t or o are missing
    valididx = (~np.isnan(t)) & (~np.isnan(o))
    t = t[valididx]
    try:
        o = o[valididx]
    except:
        o = o[valididx[0]]

    # Mean rate of the cell
    meanRate = np.nansum( np.multiply( t, (o/np.nansum(o)) ) )

    # Spatial information in bits per seconds.
    SInfo = np.multiply( (o/np.nansum(o)),t )
    SInfo = np.nansum( np.multiply( SInfo, np.log2(t / meanRate ) ) )

    # Converting in bits per spike.
    SInfoperspike = SInfo/meanRate

    return SInfoperspike

def getSparsity(t, o):
    """getSparsity Compute the sparsity index of a tuning curve.

    sparsityIndex = getSparsity(t, o) computes the selectivity of a tuning curve t
    given an occupancy o. The sparsity index is defined as 1 minus the squared sum
    of (t.*o)/sum(o) divided by the squared sum of (t.^2.*o)/sum(o), as proposed by
    Jung, Wiener, and McNaughton (JNS, 1994). A higher value indicates greater sparsity
    and selectivity in the tuning curve.

    INPUTS:
    - t: Tuning curve representing firing rates.- o: Occupancy map corresponding to the tuning curve.

    OUTPUT:
    - sparsityIndex: Sparsity index of the tuning curve.

    USAGE:
    sparsityIndex = getSparsity(t, o);

    Written by J. Fournier in 08/2023 for the Summer school
    Advanced computational analysis for behavioral and neurophysiological recordings
    Adapted by Tulio Almeida"""
    import numpy as np

    valididx = (~np.isnan(t)) & (~np.isnan(o))
    t = t[valididx]
    try:
        o = o[valididx]
    except:
        o = o[valididx[0]]

    return 1 - np.power(np.nansum(np.multiply(t,o/np.nansum(o))),2)/np.nansum(np.multiply(np.power(t,2),o/np.nansum(o)))

def getSelectivity(t):
    """getSelectivity computes the selectivity index of a tuning curve.
    
    selectivityIndex = getSelectivity(t) computes the selectivity index of a
    tuning curve t as the difference between the maximal and minimal values
    normalized by the mean value of the tuning curve.
    
    INPUT:
    - t: Tuning curve representing firing rates.
    
    OUTPUT:
    - selectivityIndex: Selectivity index of the tuning curve.
    
    USAGE:
    selectivityIndex = getSelectivity(t);
    %
    
    Written by J. Fournier in 08/2023 for the Summer school
    "Advanced computational analysis for behavioral and neurophysiological recordings"
    Adapted by Tulio Almeida"""
    import numpy as np

    return (np.nanmax(t)-np.nanmin(t)) / np.nanmean(t)

def getDirectionality(t1, t2):
    """getDirectionality Compute the directionality index between two tuning curves.
    
    DirectionalityIndex = getDirectionality(t1, t2) computes the directionality
    index between two tuning curves t1 and t2. The directionality index measures
    the discrepancy between the tuning curves as the absolute difference
    divided by the sum
    
    INPUTS:
    - t1: First tuning curve.
    - t2: Second tuning curve.
    
    OUTPUT:
    - DirectionalityIndex: Directionality index between the two tuning curves.
    
    USAGE:
    DirectionalityIndex = getDirectionality(t1, t2);
    
    Written by J. Fournier in 08/2023 for the Summer school
    "Advanced computational analysis for behavioral and neurophysiological recordings"
    Adapted by Tulio Almeida"""
    import numpy as np

    first = abs(np.sum(t1-t2))
    second = np.sum(t1+t2)
    DirectionalityIndex = np.divide(first,second)

    return DirectionalityIndex

def crossvalPartition(n,kfold):
    """cv = crossvalPartition(n, kfold) returns a structure containing k-fold cross-validation
    partition sets where training and test sets are contiguous within each partition.
    
    INPUTS:
    - n: Total number of data points.
    - kfold: Number of folds for cross-validation.
    
    OUTPUT:
    - cv: Struct with fields trainsets and testsets, each containing kfold cell arrays
      defining the training and testing indices for each fold.
    
    USAGE:
    cv = crossvalPartition(n, kfold);
    
    Written by J. Fournier in 08/2023 for the Summer school
    "Advanced computational analysis for behavioral and neurophysiological recordings"
    Adapted by Tulio"""
    import numpy as np

    # Creates a partition of 1:n indices into k-fold.

    trainsets = {k:np.zeros((n)) for k in range(kfold)}
    testsets = {k:np.zeros((n)) for k in range(kfold)}

    # Edges of the subparts, taken as contiguous.

    kidx = np.multiply(np.floor(np.divide(n,kfold)),np.arange(kfold+1))
    kidx[-1] = n
    kidx = kidx.astype(int)

    for k in range(kfold):
        testsets[k][kidx[k]:kidx[k+1]] = 1
        trainsets[k][np.invert(testsets[k].astype(bool))] = 1

    cv = {'trainsets':trainsets,
          'testsets':testsets}

    return cv

def computeEV(y,ypred):
    """computeEV computes the Explained Variance (EV) of a model's prediction.
    
    EV = computeEV(y, ypred) computes the Explained Variance (EV) of a model's prediction
    given the original data (y) and its prediction (ypred).
    
    INPUTS:
    - y: Original data.
    - ypred: Model prediction.
    
    OUTPUT:
    - EV: Explained Variance.
    
    USAGE:
    EV = computeEV(y, ypred);
    
    Written by J. Fournier in 08/2023 for the Summer school
    "Advanced computational analysis for behavioral and neurophysiological recordings"
    Adapted by Tulio Almeida"""
    import numpy as np

    # Calculate Residual Sum of Squares (RSS)
    RSS = np.nansum(np.power((y - ypred),2))

    # Calculate Mean of the original data (y)
    m = np.nanmean(y)

    # Calculate Total Sum of Squares (TSS)
    TSS = np.nansum(np.power((y - m),2))

    # Calculate Explained Variance (EV)
    return 1 - (RSS/TSS)

def normpdf_python(x, mu, sigma):
    """Function to calculate the probability density function 
    for a normal distribution with std = s.
    
    ref: https://stackoverflow.com/questions/19913613/matlab-to-python-stat-equation
    """
    import numpy as np
    
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(x-mu)**2/2*sigma**2)

def computeLLH_normal(y, ypred, k = None):
    """computeLLH_normal Compute log likelihood, Bayesian Information Criterion (BIC),
    and Akaike Information Criterion (AIC) for a Gaussian model.
    
    [LLH, BIC, AIC] = computeLLH_normal(y, ypred, k) computes the log likelihood (LLH)
    for a Gaussian model given the original signal (y) and its model prediction (ypred).
    The total number of model parameters (k) is optionally provided for calculating BIC and AIC.
    
    INPUTS:
    - y: Original signal.
    - ypred: Model prediction.
    - k: Total number of model parameters (optional for BIC and AIC).
    
    OUTPUTS:
    - LLH: Log likelihood of the Gaussian model.
    - BIC: Bayesian Information Criterion.
    - AIC: Akaike Information Criterion.
    
    USAGE:
    [LLH, BIC, AIC] = computeLLH_normal(y, ypred, k);
    
    Written by J. Fournier in 08/2023 for the Summer school
    "Advanced computational analysis for behavioral and neurophysiological recordings"
    Adapted by Tulio Almeida"""
    import numpy as np

    # std of the model

    s = np.nanstd((y - ypred))

    # probability density function for a normal distribution with std = s.

    pdfun = normpdf_python(y, ypred, s)
    pdfun[pdfun == 0] = 1e-16

    # Log likelihood is the sum of the log of the probabilities

    LLH = np.nansum(np.log(pdfun))

    # if k is provided, we also compute BIC and AIC of the model.

    if k is None:
        BIC = np.nan
        AIC = np.nan
    else:
        N = len(y)
        BIC = k * np.log(N) - 2 * LLH
        AIC = 2 * k - 2 * LLH

    return LLH,BIC,AIC

def lratiotest(llmax,llmin,dof):
    """ likelihood ratio test in python
    
    Created by Tulio Almeida"""
    from scipy.stats.distributions import chi2

    LR = 2*(llmax-llmin)
    try:
        p = chi2.sf(LR, max(dof))
    except:
        p = chi2.sf(LR, dof)

    return LR,p

def MapsAnalysis(Nav, Spk, mapsparams):    
    """MapsAnalysis - Estimates two-dimensional place fields and their significance.
    
      Maps = MapsAnalysis(Nav, Spk, mapsparams)
    
      This function estimates two-dimensional maps and their significance using either
      shuffling or model comparison on cross-validated predictions.
    
      INPUTS:
      - Nav: A structure containing at least a field called 'sampleTimes' with
      the sample times of the data and some additional fields with the
      explanatory variables
      - Spk: spike data
      - mapsparams: Struct containing parameters for place field estimation.
    
      OUTPUT:
      - Maps: Struct containing place field analysis results, including fields such as:
        * map: Two-dimensional place fields (ncells x nYbins x nXbins array)
        * map_cv: Two-dimensional place fields estimated using k-fold 
          cross-validation (ncells x nYbins x nXbins x k-fold array)
        * map_SE: Jackknife estimate of standard error for place fields, 
          (ncells x nYbins x nXbins array)
        * mapsparams: Structure of parameters used for analysis
        * Xbincenters: Bin centers along X-axis
        * Ybincenters: Bin centers along Y-axis
        * occmap: Occupancy map, a nYbins x nXbins arrays (scaled by mapsparams.scalingFactor)
        * SI: Spatial information for each cell (ncells x 1 array).
        * SparsityIndex: Sparsity index for each cell.
        * SelectivityIndex: Selectivity index for each cell.
        * DirectionalityIndex: Directionality index for each cell.
        * SI_pval: P-values for spatial information, based on shuffle controls
        * SparsityIndex_pval: P-values for sparsity index, based on shuffle controls
        * SelectivityIndex_pval: P-values for selectivity index, based on shuffle controls
        * DirectionalityIndex_pval: P-values for directionality index, based on shuffle controls
        * EV: Cross-validated percentage of explained variance from place field model
        * EV_cst: Cross-validated percentage of explained variance from constant mean model
        * LLH: Cross-validated Log likelihood from place field model
        * LLH_cst: Cross-validated Log likelihood from constant mean model
        * LLH_pval: P-values for model comparison from likelihood ratio test
    
      USAGE:
       Nav = LoaddataNav(loadparams)
       Spk = LoaddataSpk(loadparams, Nav['sampleTimes'])
       mapsparams = SetMapsParams(Nav,Spk)
    
       # Change parameters of mapsparams here if needed. For instance
       mapsparams.Yvariablename = []; # compute 1D maps along X variable.
    
       Maps = MapsAnalysis(Nav, Spk.spikeTrain, mapsparams);
    
      SEE ALSO:
      ComputeMap, GaussianSmooth, computeEV, computeLLH_normal, crossvalPartition,
      getSpatialinfo, getSparsity, getSelectivity, getDirectionality,
      lratiotest (requires econometrics toolbox).
    
    Written by J. Fournier in 08/2023 for the Summer school
    "Advanced computational analysis for behavioral and neurophysiological recordings"
    Adapted by Tulio Almeida
    """
    smth = 1

    # If no Y variable are indicated in mapsparams, we'll just compute a 1D map
    if len(mapsparams['Xvariablename']) != 0:
        X = Nav[mapsparams['Xvariablename']]
    else:
        X = np.ones(Nav['sampleTimes'].shape).astype(bool)
        mapsparams['Xbinedges'] = 1
        mapsparams['XsmthNbins'] = 0

    # If no Y variable are indicated in mapsparams, we'll just compute a 1D map
    if len(mapsparams['Yvariablename']) != 0:
        Y = Nav[mapsparams['Yvariablename']]
    else:
        Y = np.ones(Nav['sampleTimes'].shape).astype(bool)
        mapsparams['Ybinedges'] = 1
        mapsparams['YsmthNbins'] = 0

    # Selecting time indices over which to compute maps, according to parameters defined in mapsparams.subset
    tidx = np.ones(X.shape).astype(bool)
    pnames = list(mapsparams['subset'].keys())
    fnames = list(Nav.keys()) 
    for i in range(len(pnames)):
        if pnames[i] in fnames:
            fn = mapsparams['subset'][pnames[i] + '_op']
            tidx = tidx & fn(Nav[pnames[i]], mapsparams['subset'][pnames[i]])
        elif not pnames[i][-3:] == '_op':
            print("Some fields of mapsparams.subset are not matching fields of Nav")

    # Selecting time indices where X and Y are in the range of bins
    try:
        tidx = (tidx) & \
            (X >= mapsparams['Xbinedges'][0]) & (X <= mapsparams['Xbinedges'][-1]) & \
            (Y >= mapsparams['Ybinedges'][0]) & (Y <= mapsparams['Ybinedges'][-1]) & \
            (~np.isnan(X)) & (~np.isnan(Y))
    except:
        tidx = (tidx) & \
            (X >= mapsparams['Xbinedges'][0]) & (X <= mapsparams['Xbinedges'][-1]) & \
            (Y >= mapsparams['Ybinedges']) & (Y <= mapsparams['Ybinedges']) & \
            (~np.isnan(X)) & (~np.isnan(Y))

    # Selecting cell indices for which to compute maps
    cellidx = np.argwhere((mapsparams['cellidx'].T & np.nansum(Spk['spikeTrain'][tidx,:], 0).astype(bool)) 
                        > mapsparams['nspk_th'])

    # Subsetting spikeTrain, X and Y.
    # after this step the cell became the X and the idx the Y
    # when I tried to fix this the colab RAM ran OOF
    spikeTrain = Spk['spikeTrain'][tidx,cellidx] 
    # spikeTrain = Spk['spikeTrain'][tidx,:]
    # spikeTrain = spikeTrain[:,cellidx]
    X = X[tidx]
    Y = Y[tidx]

    # number of bins along X
    nXbins = max(1,len(mapsparams['Xbinedges']) - 1)

    # number of bins along Y
    try:
        nYbins = max(1,len(mapsparams['Ybinedges']) - 1) 
    except:
        nYbins = max(1,mapsparams['Ybinedges'])

    # If no Y variable are indicated, we'll just compute a 1D maps so bining parameters are changed accordingly
    if len(mapsparams['Yvariablename']) == 0:
        # number of bins along Y
        nYbins = 1

    # number of selected cells
    ncells = spikeTrain.shape[0]

    # number of data points
    ntimepts = spikeTrain.shape[1]

    # Discretizing X position vectors according to mapsparams['Xbinedges']
    try:
        X_discrete = np.digitize(X, mapsparams['Xbinedges'])
        X_discrete = X_discrete.astype(np.float32)
        X_discrete[np.isnan(X)] = np.nan
        X_discrete[X_discrete>nXbins] = nXbins
    except:
        X_discrete = X
    # Discretizing Y position vectors according to mapsparams['Ybinedges'] 
    try:
        Y_discrete = np.digitize(Y, mapsparams['Ybinedges'])
        Y_discrete = Y_discrete.astype(np.float32)
        Y_discrete[np.isnan(Y)] = np.nan
        Y_discrete[Y_discrete>nYbins] = nYbins
    except:
        Y_discrete = Y

    # Computing occupancy map (same for all cells)
    flat =  mapsparams['scalingFactor'] * np.ones(X_discrete.shape)

    if sum(Y_discrete) == sum(Y):
        occmap = ComputeMap(Xd = X_discrete, Yd = None, Z = flat,
                            nXbins = nXbins, nYbins = None)
    elif sum(X_discrete) == sum(X):
        occmap = ComputeMap(Xd = None, Yd = Y_discrete, Z = flat,
                            nXbins = None, nYbins = nYbins)
    else:
        occmap = ComputeMap(Xd = X_discrete, Yd = Y_discrete, Z = flat,
                            nXbins = nXbins, nYbins = nYbins) 
        
    # Removing occupancy for bins below the occupancy threshold
    occmap[occmap <= mapsparams['occ_th']] = np.nan 

    # Smoothing the occupancy map with a 2D gaussian window
    occmap = GaussianSmooth(occmap, [mapsparams['XsmthNbins'] + smth,mapsparams['YsmthNbins'] + smth]) # +1 ? smth

    # Computing and smoothing spike count map for each cell
    scmap = np.empty((ncells, nYbins, nXbins)) * np.nan
    for icell in range(ncells):
        if sum(Y_discrete) == sum(Y):
            scmapcell = ComputeMap(Xd = X_discrete, Yd = None, Z = spikeTrain[icell,:],
                                nXbins = nXbins, nYbins = None)
        elif sum(X_discrete) == sum(X):
            scmapcell = ComputeMap(Xd = None, Yd = Y_discrete, Z = spikeTrain[icell,:],
                                nXbins = None, nYbins = nYbins)
        else:
            scmapcell = ComputeMap(Xd = X_discrete, Yd = Y_discrete, Z = spikeTrain[icell,:],
                                nXbins = nXbins, nYbins = nYbins)  
            
        scmapcell[np.isnan(occmap)] = np.nan
        scmapcell = GaussianSmooth(scmapcell, [mapsparams['YsmthNbins'] + smth, mapsparams['XsmthNbins']+ smth]) # +1 ? smth
        try:
            scmap[icell,:,:] = scmapcell
        except:
            scmap[icell,:,:] = scmapcell.T

    # Calculating the maps by dividing scmap and occmap
    try:
        permutes = np.transpose( np.expand_dims(occmap, axis=occmap.ndim), (2, 0, 1) )
        mapXY = np.divide(scmap , permutes)
    except:
        permutes = np.transpose( np.expand_dims(occmap, axis=occmap.ndim), (2, 1, 0) )
        mapXY = np.divide(scmap , permutes)
    occmap = np.squeeze(occmap)

    # Quantifying selectivity by computing the spatial information (SI),
    # the sparsity index, the selectivity index and the directionality index.
    SI = np.empty((ncells, 1)) * np.nan
    SparsityIndex = np.empty((ncells, 1)) * np.nan
    SelectivityIndex = np.empty((ncells, 1)) * np.nan
    DirectionalityIndex = np.empty((ncells, 1)) * np.nan
    for icell in range(ncells):
        try:
            SI[icell] = getSpatialinfo(mapXY[icell,:], occmap)
            SparsityIndex[icell] = getSparsity(mapXY[icell,:], occmap)
        except:
            SI[icell] = getSpatialinfo(mapXY[icell,:].T, occmap)
            SparsityIndex[icell] = getSparsity(mapXY[icell,:].T, occmap)
        SelectivityIndex[icell] = getSelectivity(mapXY[icell,:])
        if nYbins == 2:
            DirectionalityIndex[icell] = getDirectionality(mapXY[icell,0,:], mapXY[icell,1,:])

    # Computing shuffle controls by randomly shifting time bins of positions and
    # calculate the selectivity metrics for each shuffle control
    SI_Shf = np.empty((ncells, mapsparams['nShuffle'])) * np.nan
    SparsityIndex_Shf = np.empty((ncells, mapsparams['nShuffle'])) * np.nan
    SelectivityIndex_Shf = np.empty((ncells, mapsparams['nShuffle'])) * np.nan
    DirectionalityIndex_Shf = np.empty((ncells, mapsparams['nShuffle'])) * np.nan

    # Initializing the random number generator for reproducibility purposes
    nShf = mapsparams['nShuffle']

    # Calculating the place field for each shuffle permutation
    # Initializing the random number generator for reproducibility purposes
    rng = np.random.Generator(np.random.MT19937(seed = 0))

    for icell in range(ncells): 
        for iperm in range(nShf): 
            # Shifting X and Y by a random amount larger than 1 second
            tshift = int(rng.integers(ntimepts - 2 * mapsparams['sampleRate']) + 1 * mapsparams['sampleRate'])
            X_discrete_shf = np.roll(X_discrete, tshift)
            Y_discrete_shf = np.roll(Y_discrete, tshift)

            # Computing maps after shuffling
            if sum(Y_discrete_shf) == sum(Y):
                scmap_shf = ComputeMap(Xd = X_discrete_shf, Yd = None, Z = spikeTrain[icell,:],
                                    nXbins = nXbins, nYbins = None)
            elif sum(X_discrete_shf) == sum(X):
                scmap_shf = ComputeMap(Xd = None, Yd = Y_discrete_shf, Z = spikeTrain[icell,:],
                                    nXbins = None, nYbins = nYbins)
            else:
                scmap_shf = ComputeMap(Xd = X_discrete_shf, Yd = Y_discrete_shf, Z = spikeTrain[icell,:],
                                    nXbins = nXbins, nYbins = nYbins)
            try:      
                scmap_shf[np.isnan(occmap)] = np.nan
            except:
                scmap_shf[0][np.isnan(occmap)] = np.nan
            scmap_shf = GaussianSmooth(scmap_shf, [mapsparams['YsmthNbins']+ smth, mapsparams['XsmthNbins']+ smth])
            mapX_shf = np.divide(scmap_shf , occmap)

            # saving only the spatial selectivity metrics for each permutation
            SI_Shf[icell,iperm] = getSpatialinfo(mapX_shf[:], occmap)
            SparsityIndex_Shf[icell,iperm] = getSparsity(mapX_shf[:], occmap)
            SelectivityIndex_Shf[icell,iperm] = getSelectivity(mapX_shf[:])
            if nYbins == 2:
                DirectionalityIndex_Shf[icell,iperm] = getDirectionality(mapX_shf[:,0], mapX_shf[:,1])

    # Computing p-values from the distribution of selectivity measures obtained from the shuffle controls
    SI_pval = np.nansum(SI_Shf> SI,1) / mapsparams['nShuffle']
    SparsityIndex_pval = np.nansum(SparsityIndex_Shf > SparsityIndex, 1) / mapsparams['nShuffle']
    SelectivityIndex_pval = np.nansum(SelectivityIndex_Shf > SelectivityIndex, 1) / mapsparams['nShuffle']
    DirectionalityIndex_pval = np.nansum(DirectionalityIndex_Shf > DirectionalityIndex, 1) / mapsparams['nShuffle']

    # Defining a partition of the data for k-fold cross-validation
    cv = crossvalPartition(ntimepts, mapsparams['kfold'])

    # Computing the spike train predicted from the place field using k-fold  cross-validation
    mapXY_cv = np.empty((ncells, nYbins, nXbins, mapsparams['kfold'])) * np.nan
    Ypred = np.empty((ntimepts,ncells)) * np.nan

    for i in range(mapsparams['kfold']):
        # Subsetting X and spiketrain according to the train set of the current fold
        Xtraining = X_discrete[cv['trainsets'][i].astype(bool)]
        Ytraining = Y_discrete[cv['trainsets'][i].astype(bool)]
        Spktraining = spikeTrain[:,cv['trainsets'][i].astype(bool)]

        # Computing occupancy map for the current fold
        flat = mapsparams['scalingFactor'] * np.ones(Xtraining.shape)
        if sum(Ytraining) == sum(Y[cv['trainsets'][i].astype(bool)]):
            occmap_cv = ComputeMap(Xd = Xtraining, Yd = None, Z = flat,
                                nXbins = nXbins, nYbins = None)
        elif sum(Xtraining) == sum(X[cv['trainsets'][i].astype(bool)]):
            occmap_cv = ComputeMap(Xd = None, Yd = Ytraining, Z = flat,
                                nXbins = None, nYbins = nYbins)
        else:
            occmap_cv = ComputeMap(Xd = Xtraining, Yd = Ytraining, Z = flat,
                                nXbins = nXbins, nYbins = nYbins)  

        occmap_cv[occmap_cv <= mapsparams['occ_th']] = np.nan
        occmap_cv = GaussianSmooth(occmap_cv, [mapsparams['YsmthNbins'] + smth, mapsparams['XsmthNbins'] + smth])

        # Computing the spike count map and place field of each cell for the current fold
        for icell in range(ncells):
            if sum(Ytraining) == sum(Y[cv['trainsets'][i].astype(bool)]):
                scmap_cv = ComputeMap(Xd = Xtraining, Yd = None, Z = Spktraining[icell,:],
                                    nXbins = nXbins, nYbins = None)
            elif sum(Xtraining) == sum(X[cv['trainsets'][i].astype(bool)]):
                scmap_cv = ComputeMap(Xd = None, Yd = Ytraining, Z = Spktraining[icell,:],
                                    nXbins = None, nYbins = nYbins)
            else:
                scmap_cv = ComputeMap(Xd = Xtraining, Yd = Ytraining, Z = Spktraining[icell,:],
                                    nXbins = nXbins, nYbins = nYbins) 
            try:
                scmap_cv[np.isnan(occmap)] = np.nan
            except:
                scmap_cv[0][np.isnan(occmap)] = np.nan
            scmap_cv = GaussianSmooth(scmap_cv, [mapsparams['YsmthNbins'] + smth, mapsparams['XsmthNbins'] + smth])
            try:
                mapXY_cv[icell,:,:,i] = np.divide(scmap_cv,occmap_cv)
            except:
                mapXY_cv[icell,:,:,i] = np.divide(scmap_cv,occmap_cv).T

        # Subsetting X and Y according to the test set of the current fold
        Xtest = X_discrete[cv['testsets'][i].astype(bool)]
        Ytest = Y_discrete[cv['testsets'][i].astype(bool)]

        # Computing the spike train predicted on the test set from the place computed from the train set
        for icell in range(ncells):
            temp_arr = (icell*np.ones(Xtest.shape).astype(int),
                        Ytest.astype(int) - 1,
                        Xtest.astype(int) - 1,
                        i*np.ones(Xtest.shape).astype(int))
            temp_idx = (ncells, nYbins, nXbins, mapsparams['kfold'])
            XYlinidx = np.ravel_multi_index(temp_arr, temp_idx, mode = 'wrap')
            Ypred[cv['testsets'][i].astype(bool),icell] = mapXY_cv.flatten()[XYlinidx] * mapsparams['scalingFactor']

    # Now computing the spike train predicted from the mean firing rate of the 
    # cell using the same k-fold partition as above
    Ypred_cst = np.empty((ntimepts,ncells)) * np.nan
    for i in range(mapsparams['kfold']):
        for icell in range(ncells):
            Ypred_cst[cv['testsets'][i].astype(bool),icell] = np.nanmean(spikeTrain[icell,cv['testsets'][i].astype(bool)])

    # Computing the percentage of explained variance and the log likelihood from the spike trains predicted by the place field model
    EV = np.empty((ncells,1))*np.nan
    LLH = np.empty((ncells,1))*np.nan

    for icell in range(ncells):
        # Percentage of explained variance
        EV[icell] =  computeEV(spikeTrain[icell,:], Ypred[:,icell])

        # Log likelihood
        LLH[icell],BIC,AIC = computeLLH_normal(spikeTrain[icell,:], Ypred[:,icell])

    # Same for the spike train predicted by the mean constant model

    EV_cst = np.empty((ncells,1))*np.nan
    LLH_cst = np.empty((ncells,1))*np.nan

    for icell in range(ncells):
        # Percentage of explained variance
        EV_cst[icell] =  computeEV(spikeTrain[icell,:], Ypred_cst[:,icell])

        # Log likelihood
        LLH_cst[icell],_,_ = computeLLH_normal(spikeTrain[icell,:], Ypred_cst[:,icell])

    # Comparing the place field model to the constant mean model by performing a likelihood ratio test
    LLH_pval = np.empty((ncells,1)) * np.nan
    goodidx = LLH > LLH_cst
    LLH_pval[~goodidx] = 1
    dof = sum(occmap > 0)-1
    if sum(goodidx) > 0:
        _,LLH_pval[goodidx] = lratiotest(LLH[goodidx],LLH_cst[goodidx],dof)

    # Computing a Jacknife estimate of the standard error
    for i in range(mapXY_cv.shape[3]):
        mapXY_cv[:,:,:,i] = np.power(mapXY_cv[:,:,:,i] - mapXY,2)
    mapXY_SE = np.sqrt( (np.divide(mapsparams['kfold'] - 1,mapsparams['kfold'])) * np.sum(mapXY_cv,3) )

    # Populate the output structure with results to be saved
    mapsparams['tidx'] = tidx
    Maps = {'mapsparams':mapsparams}

    if nXbins > 1:
        Maps['Xbincenters'] = mapsparams['Xbinedges'][:-1] + np.diff(mapsparams['Xbinedges']) / 2
    else:
        Maps['Xbincenters'] = 1

    if nYbins > 1:
        Maps['Ybincenters'] = mapsparams['Ybinedges'][:-1] + np.diff(mapsparams['Ybinedges']) / 2
    else:
        Maps['Ybincenters'] = 1

    # Interpolating in case there are infinite values among the bin edges
    # (probably at the beginiing or end in order to clamp de signal
    if nXbins > 1 and np.sum(~np.isinf(Maps['Xbincenters'])) > 0:
        xb = np.arange(nXbins)
        Maps['Xbincenters'] = interpolate.interp1d(xb[~np.isinf(Maps['Xbincenters'])], 
                                                Maps['Xbincenters'][~np.isinf(Maps['Xbincenters'])], 
                                                kind = 'linear', fill_value="extrapolate")(xb)

    if nYbins > 1 and np.sum(~np.isinf(Maps['Ybincenters'])) > 0:
        yb = np.arange(nYbins)
        Maps['Xbincenters'] = interpolate.interp1d(yb[~np.isinf(Maps['Ybincenters'])], 
                                                Maps['Ybincenters'][~np.isinf(Maps['Ybincenters'])], 
                                                kind = 'linear', fill_value="extrapolate")(yb)

    Maps['map'] = mapXY
    Maps['map_cv'] = mapXY_cv
    Maps['map_SE'] = mapXY_SE
    Maps['occmap'] = occmap
    Maps['SI'] = SI
    Maps['SparsityIndex'] = SparsityIndex
    Maps['SelectivityIndex '] = SelectivityIndex
    Maps['DirectionalityIndex'] = DirectionalityIndex
    Maps['SI_pval']= SI_pval
    Maps['SparsityIndex_pval']= SparsityIndex_pval
    Maps['SelectivityIndex_pval'] = SelectivityIndex_pval
    Maps['DirectionalityIndex_pval'] = DirectionalityIndex_pval
    Maps['EV'] = EV
    Maps['EV_cst'] = EV_cst
    Maps['LLH'] = LLH
    Maps['LLH_cst'] = LLH_cst
    Maps['LLH_pval'] = LLH_pval

    return Maps