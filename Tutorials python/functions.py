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
                
    trialStart = np.argwhere(np.sign(np.diff(~np.isnan(pos_dict['Xpos'][:-1])))>0) + 1
    trialEnd = np.argwhere(np.sign(np.diff(~np.isnan(pos_dict['Xpos'])))<0)

    # if the animal is already on the track at the beginning, we modify trialStart accordingly
    if trialEnd[0] < trialStart[0]:
        trialStart = np.concatenate((np.zeros((1,1)),trialStart))

    # %if the recording is stopped while the animal is on the track, we modify trialEnd accordingly.
    if len(trialEnd) < len(trialStart)
        trialEnd = np.concatenate((trialEnd , len(pos_dict['Xpos'])))

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

    acc = mat73.loadmat(mat_files['LFP2'], only_include='acc')
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

def DefineMapsParams(): # TODO
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

def DefineLoadParams():
    params = {'sampleRate': 50,
            'sampleRate_rawLfp': 600,
            'pix2cm': 0.4300,
            'ShankList': [1,2,3,4],
            'ShankList_blaL': [5,6,7,8],
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