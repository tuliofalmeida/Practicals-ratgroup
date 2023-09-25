def loadparams():
    """
    Function create the dict of the
    params to load the data

    Parameters
    --------------
    None

    Returns
    --------------
    params : dict
        dict with the params
    """
    params = {'sampleRate': 50,
            'sampleRate_rawLfp': 600,
            'pix2cm': 0.4300,
            'ShankList': [1,2,3,4],
            'LfpChannel_Hpc': 2,
            'LfpChannel_Bla': 2,
            'LfpChannel_Acc': 2,
            'ThetaBand': [6,9]}
    
    return params

def load_pos(mat_files,evt_files):
    """
    Function to load position data

    Parameters
    --------------
    mat_files : dict
        dict with the .mat files path
    evt_files : dict
        dict with the .evt files path

    Returns
    --------------
    pos_dict : dict
        dict with the preprocessed data
    """
    import os
    import mat73
    import numpy as np
    import pandas as pd
    import scipy.io
    import matplotlib.pyplot as plt
    from scipy import signal,stats,sparse,interpolate
    from scipy.ndimage import median_filter,gaussian_filter1d

    params = loadparams()

    # --------------------------------------------POSITION DATA--------------------------------------------

    # loading position data and convert positions from pixel to cm
    pos = scipy.io.loadmat(mat_files['Positions'])
    pos_dict = {'sampleTimes':pos['positions'][:,0],
                'x':pos['positions'][:,1] * params['pix2cm'],
                'y':pos['positions'][:,2] * params['pix2cm'],
                'Condition': np.empty((len(pos['positions'][:,1])))*np.nan,
                'Spd': np.empty((len(pos['positions'][:,1])))*np.nan,
                'smthSpd': np.empty((len(pos['positions'][:,1])))*np.nan,
                'XDir': np.empty((len(pos['positions'][:,1])))*np.nan}

    # loading subsessions indices

    cat = pd.read_table(evt_files['cat'], header = None)

    # Filling in the Nav.Condition vector with 1 for preprun, 2 for presleep, 3 for run, 4 for postsleep, 5 for postrun

    condID = np.repeat(np.arange(1,6,1),2)
    events = {info.split(' ')[-1].split('-')[-1]+'-'+info.split(' ')[1]:[float(info.split(' ')[0])/1000,condID[idx]] for idx,info in enumerate(cat[0])}
    for idx in range(0,len(events),2):
        condidx = (pos_dict['sampleTimes'] >= list(events.values())[idx][0]) & (pos_dict['sampleTimes'] <= list(events.values())[idx+1][0])
        pos_dict['Condition'][condidx] = list(events.values())[idx][1]

    # Computing a few additional behavioral metrics (Speed, acceleration, direction of mvt?). These need to be
    # computed for each sessions seperately since recording are discontinous between sessions

    sampleRate_orig = 1/np.mean(np.diff(pos_dict['sampleTimes']))
    smthwin = np.round(sampleRate_orig * .5).astype(int)
    spd_th = 5

    # Running speed

    for icond in set(condID):
        end = np.array([pos_dict['sampleTimes'][pos_dict['Condition']==icond][-1] - pos_dict['sampleTimes'][pos_dict['Condition']==icond][-2]])
        Xdiff = np.concatenate((np.diff(pos_dict['x'][pos_dict['Condition']==icond]), np.array([np.nan])))
        Ydiff = np.concatenate((np.diff(pos_dict['y'][pos_dict['Condition']==icond]), np.array([np.nan])))
        Tdiff = np.concatenate((np.diff(pos_dict['sampleTimes'][pos_dict['Condition']==icond]),end))
        # pos_dict['Spd'][pos_dict['Condition']==icond] = median_filter(np.divide(np.sqrt(Xdiff**2 + Ydiff**2),Tdiff),smthwin)
        spd_temp = np.divide(np.sqrt(Xdiff**2 + Ydiff**2),Tdiff)
        pos_dict['Spd'][pos_dict['Condition']==icond] = spd_temp
        pos_dict['smthSpd'][pos_dict['Condition']==icond] = signal.savgol_filter(spd_temp,smthwin,1)

        # Speed along X

        Xspd =  signal.savgol_filter(np.divide(Xdiff,Tdiff),smthwin,1) # median_filter(np.divide(Xdiff,Tdiff),smthwin)
        XDirLtoR = signal.savgol_filter(Xspd > 0, smthwin,0) > 0 # median_filter(Xspd > spd_th, smthwin)
        XDirRtoL = signal.savgol_filter(Xspd < 0, smthwin,0) > 0 # median_filter(Xspd < -spd_th, smthwin)
        pos_dict['XDir'][pos_dict['Condition']==icond] = (XDirLtoR*1 - XDirRtoL*1 > 0)*1 - (XDirLtoR*1 - XDirRtoL*1 < 0)*1 # XDirLtoR*1 - XDirRtoL*1

    # Resampling behavioral data to the final resolution (Loadparams.sampleRate)

    sampleTimes_orig = pos_dict['sampleTimes']
    sampleTimes_new = np.arange(min(sampleTimes_orig),max(sampleTimes_orig),1/params['sampleRate'])
    pos_dict['x'] = interpolate.interp1d(sampleTimes_orig,pos_dict['x'],kind='linear')(sampleTimes_new)
    pos_dict['y'] = interpolate.interp1d(sampleTimes_orig,pos_dict['y'],kind='linear')(sampleTimes_new)
    pos_dict['Spd'] = interpolate.interp1d(sampleTimes_orig,pos_dict['Spd'],kind='linear')(sampleTimes_new)
    pos_dict['XDir'] = interpolate.interp1d(sampleTimes_orig,pos_dict['XDir'],kind='nearest')(sampleTimes_new)
    pos_dict['smthSpd'] = interpolate.interp1d(sampleTimes_orig,pos_dict['smthSpd'],kind='linear')(sampleTimes_new)
    pos_dict['Condition'] = interpolate.interp1d(sampleTimes_orig,pos_dict['Condition'],kind='nearest')(sampleTimes_new)
    pos_dict['sampleTimes'] = sampleTimes_new

    # loading information about types of lap (left to right, right to left).
    # Nav.laptype equals 1 or -1 for left to right and right to left trials respectively.

    pos_dict['laptype'] = np.zeros(pos_dict['x'].shape)
    pos_dict['uturn'] = np.zeros(pos_dict['x'].shape)
    LapType = mat73.loadmat(mat_files['LapType2'])

    for idx in range(len(LapType['LtoRlaps'])):
        idx = (pos_dict['sampleTimes'] >= LapType['LtoRlaps'][idx,0]) & (pos_dict['sampleTimes'] <= LapType['LtoRlaps'][idx,1])
        pos_dict['laptype'][idx] = 1

    for idx in range(len(LapType['RtoLlaps'])):
        idx = (pos_dict['sampleTimes'] >= LapType['RtoLlaps'][idx,0]) & (pos_dict['sampleTimes'] <= LapType['RtoLlaps'][idx,1])
        pos_dict['laptype'][idx] = -1

    # Nav.uturn = 1 when the rat makes a uturn before the end f the trial.
    # Nav.laptype equals zero when when the rats makes a u-turn before the end.

    for idx in range(len(LapType['Uturnlaps'])):
        idx = (pos_dict['sampleTimes'] >= LapType['Uturnlaps'][idx,0]) & (pos_dict['sampleTimes'] <= LapType['Uturnlaps'][idx,1])
        pos_dict['uturn'][idx] = 1

    # Expressing positions along X as percentage of the track length

    Xtrackstart = min(pos_dict['x'][pos_dict['laptype'] != 0])
    Xtrackend = max(pos_dict['x'][pos_dict['laptype'] != 0])
    pos_dict['Xpos'] = 100 * (pos_dict['x'] - Xtrackstart) / (Xtrackend - Xtrackstart)
    pos_dict['Xpos'][(pos_dict['Xpos'] < 0) | (pos_dict['Xpos'] > 100)] = np.nan
    pos_dict['Xpos'][np.invert(np.isin(pos_dict['Condition'],[1,3,5]))] = np.nan # new decoder

    # Defining some trials indices for whenever the animal is on the track and spend more than 0.5 second there.
    # Looking for potential start and end of trials

    trialduration_th = 0.5
    trialStart = (np.argwhere(np.sign(np.diff(np.invert(np.isnan(pos_dict['Xpos'][:-2]))*1))>0)+1).reshape(-1)
    trialEnd = (np.argwhere(np.sign(np.diff(np.invert(np.isnan(pos_dict['Xpos']))*1))<0)).reshape(-1)

    # if the animal is already on the track at the beginning, we modify trialStart accordingly

    if trialEnd[0] < trialStart[0]:
        trialStart = np.concatenate((np.array([0]),trialStart))

    # if the recording is stopped while the animal is on the track, we modify trialEnd accordingly.

    if len(trialEnd) < len(trialStart):
        trialEnd = np.concatenate((trialEnd,np.array([len(pos_dict['Xpos'])])))

    # Initializing the vector of trialIDs

    pos_dict['trialID'] = np.empty(pos_dict['Xpos'].shape)*np.nan

    # Checking that the trials are valid (i.e. longer than 1 s)
    trialnum = 0
    for k in range(len(trialStart)):
        if (pos_dict['sampleTimes'][trialEnd[k]] - pos_dict['sampleTimes'][trialStart[k]]) > trialduration_th:
            trialnum += 1
            pos_dict['trialID'][trialStart[k]:trialEnd[k]] = trialnum

    # loading left/right reward times and filling in Nav.reward with 1 for right reward and -1 for left reward

    pos_dict['reward'] = np.zeros(pos_dict['x'].shape)

    lrw = pd.read_table(evt_files['lrw'],header = None)
    lrw = lrw[0].values/1000
    rrw = pd.read_table(evt_files['rrw'],header = None)
    rrw = rrw[0].values/1000

    for idx in range(len(lrw)):
        rewidx = np.argmin(abs(pos_dict['sampleTimes'] - lrw[idx]))
        pos_dict['reward'][rewidx] = -1

    for idx in range(len(rrw)):
        rewidx = np.argmin(abs(pos_dict['sampleTimes'] - rrw[idx]))
        pos_dict['reward'][rewidx] = 1

    # loading air puffs timestamps and filling in Nav.airpuff with 1 when there is an air puff

    pos_dict['airpuff'] = np.zeros(pos_dict['x'].shape)
    puf = pd.read_table(evt_files['puf'], header = None)
    puf = puf[0].values/1000

    for idx in range(len(puf)):
        rewidx = np.argmin(abs(pos_dict['sampleTimes'] - puf[idx]))
        pos_dict['airpuff'][rewidx] = 1

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

def load_spk(mat_files,Nav):
    """
    Function to load spike data

    Parameters
    --------------
    mat_files : dict
        dict with the .mat files path
    Nav : dict
        positional data, output of load_pos()

    Returns
    --------------
    spk_dict : dict
        dict with the preprocessed data
    """     
    import os
    import mat73
    import numpy as np
    import pandas as pd
    import scipy.io
    import matplotlib.pyplot as plt
    from scipy import signal,stats,sparse,interpolate
    from scipy.ndimage import median_filter,gaussian_filter1d
   
    params = loadparams()

    # --------------------------------------------SPIKE DATA--------------------------------------------

    # loading spike times and cluster ID from the prepared .mat file
    spk = scipy.io.loadmat(mat_files['HippoSpikes'])

    # Removing spikes that are before or after behavior started
    extraspk = (spk['HippoSpikes'][:,0] < Nav['sampleTimes'][0]) | (spk['HippoSpikes'][:,0] > Nav['sampleTimes'][-1])
    spk['HippoSpikes'] = np.delete(spk['HippoSpikes'],extraspk,0)

    # Saving spike times and cluster IDs.

    spk_dict = {'spikeTimes':spk['HippoSpikes'][:,0], # - Nav['sampleTimes'][0],
                'spikeID':spk['HippoSpikes'][:,1]}

    # convert spike times into an array of spike trains, at the same resolution as the behavioral data.
    # We assume the behavioral data and the spike recording have already been aligned together

    clustList = np.unique(spk['HippoSpikes'][:,1]).astype(int)
    ncells = max(clustList)
    nTimeSamples = len(Nav['sampleTimes'])

    spk_dict['spikeTrain'] = np.zeros((nTimeSamples, ncells))
    binEdges =  np.concatenate((Nav['sampleTimes'], np.array([max(Nav['sampleTimes']) + 1/params['sampleRate']])))

    for icell in clustList:
        s = spk['HippoSpikes'][spk['HippoSpikes'][:,1] == icell][:,0]
        spk_dict['spikeTrain'][:,icell-1] = np.histogram(s,binEdges)[0]

    spk_dict['sampleTimes'] = Nav['sampleTimes']

    # Saving some cluster info into the Spk structure

    idx_type = scipy.io.loadmat(mat_files['IndexType'])
    idx_type = idx_type['IndexType']
    HippoClustidx = np.isin(idx_type[:,2],params['ShankList'])
    spk_dict['shankID'] = idx_type[HippoClustidx,2]
    spk_dict['PyrCell'] = (idx_type[HippoClustidx,5] == 1)*1
    spk_dict['IntCell'] = (idx_type[HippoClustidx,5] == 2)*1

    return spk_dict

def load_lfp(mat_files,evt_files,Nav):
    """
    Function to load lfp data

    Parameters
    --------------
    mat_files : dict
        dict with the .mat files path
    evt_files : dict
        dict with the .evt files path
    Nav : dict
        positional data, output of load_pos()

    Returns
    --------------
    lfp_dict : dict
        dict with the preprocessed data
    """     
    import os
    import mat73
    import numpy as np
    import pandas as pd
    import scipy.io
    import matplotlib.pyplot as plt
    from scipy import signal,stats,sparse,interpolate
    from scipy.ndimage import median_filter,gaussian_filter1d   
    params = loadparams()

    # --------------------------------------------LFP DATA--------------------------------------------

    # loading ripples timestamps and filling in Lfp.ripple with 1 when there
    # is a ripple and Lfp.ripplepeak with 1 for the peak of each ripple

    Lfp_dict = {'sampleTimes':Nav['sampleTimes'],
                'ripple': np.zeros(Nav['x'].shape),
                'ripplepeak':np.zeros(Nav['x'].shape)}

    rip = pd.read_table(evt_files['rip'],header = None)
    ripstart = rip[np.isin(rip[1].values, 'Ripple start 23')][0].values/1000
    ripstop = rip[np.isin(rip[1].values, 'Ripple stop 23')][0].values/1000
    rippeak = rip[np.isin(rip[1].values, 'Ripple peak 23')][0].values/1000

    for idx in range(len(ripstart)):
        rippleidx = (Nav['sampleTimes'] >= ripstart[idx]) & (Nav['sampleTimes'] <= ripstop[idx])
        Lfp_dict['ripple'][rippleidx] = 1
        rippeakidx = np.argmin(abs(Nav['sampleTimes'] - rippeak[idx]))
        Lfp_dict['ripplepeak'][rippeakidx] = 1

    # Loading Lfp oscillations

    lfp = mat73.loadmat(mat_files['LFP2'], only_include=['hippoLFPs','blaLFPs'])

    # Resampling raw LFPs at the final sampling rate

    lfpsampleTimes = lfp['hippoLFPs'][:,0]
    lfpsampRate = int(1/np.mean(np.diff(lfpsampleTimes)))
    sampleTimes_orig = lfpsampleTimes
    idxStart = np.argwhere(lfpsampleTimes > Nav['sampleTimes'][0])[0][0]
    idxEnd = np.argwhere(lfpsampleTimes > Nav['sampleTimes'][-1])[0][0]
    sampleTimes_new = np.arange(lfpsampleTimes[idxStart],lfpsampleTimes[idxEnd],1/params['sampleRate_rawLfp'])
    Lfp_dict['LfpHpc_raw'] = interpolate.interp1d(sampleTimes_orig,lfp['hippoLFPs'][:,params['LfpChannel_Hpc']],kind='linear')(sampleTimes_new)
    Lfp_dict['LfpBla_raw'] = interpolate.interp1d(sampleTimes_orig,lfp['blaLFPs'][:,params['LfpChannel_Bla']],kind='linear')(sampleTimes_new)
    Lfp_dict['sampleTimes_raw'] = sampleTimes_new

    # filter theta

    wn = np.array(params['ThetaBand'])/(lfpsampRate/2)
    b,a = signal.ellip(2,.1,40,Wn = wn, btype= 'bandpass')
    Theta = signal.filtfilt(b,a,lfp['hippoLFPs'][:,params['LfpChannel_Hpc']])

    # Computing Hilbert transform of the filtered signal to get the phasepower
    # and instantaneous frequency of Theta oscillations

    hilbertTrans = signal.hilbert(Theta)
    phs = np.angle(hilbertTrans)
    pow = abs(hilbertTrans)

    Lfp_dict['ThetaPhase'] = np.mod(phs / np.pi * 180, 360)
    Lfp_dict['ThetaPower'] = pow
    Lfp_dict['ThetaFreq'] = np.diff(np.unwrap(phs)) * lfpsampRate / (2*np.pi)

    # Resampling Theta at the final sampling rate

    sampleTimes_new = Nav['sampleTimes']
    Lfp_dict['Theta'] = interpolate.interp1d(sampleTimes_orig,Theta,kind='linear')(sampleTimes_new)
    theta_phase = interpolate.interp1d(sampleTimes_orig,np.unwrap(Lfp_dict['ThetaPhase'] / 180 * np.pi),kind='linear')(sampleTimes_new)
    Lfp_dict['ThetaPhase'] = np.mod(theta_phase / np.pi * 180, 360)
    Lfp_dict['ThetaPower'] = interpolate.interp1d(sampleTimes_orig,Lfp_dict['ThetaPower'],kind='linear')(sampleTimes_new)
    Lfp_dict['ThetaFreq'] = interpolate.interp1d(sampleTimes_orig,np.concatenate((np.array([0]),Lfp_dict['ThetaFreq'])),kind='linear')(sampleTimes_new)

    return Lfp_dict

def loaddata(path):
    """
    Function to load all data

    Parameters
    --------------
    path : str
        string of the folder with all data

    Returns
    --------------
    load_pos : dict
        dict with the preprocessed pos data
    load_spk : dict
        dict with the preprocessed spk data
    lfp_dict : dict
        dict with the preprocessed lfp data
    """   
    import os
    # get the file path
    evt_files = {filename.split('.')[-2]:os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.evt')}
    mat_files = {filename.split('.')[-2]:os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.mat')}

    Nav = load_pos(mat_files,evt_files)
    Spk = load_spk(mat_files,Nav)
    Lfp = load_lfp(mat_files,evt_files,Nav)

    return Nav, Spk, Lfp

def Compute1DMap(Xd,Z, nbins):
    import numpy as np
    from scipy import sparse
    # Compute 1D map efficiently. Xd is the binned independent varaible and Z is
    # the dependent variable.

    # Selecting valid indices
    valididx = ((np.invert(np.isnan(Xd))) & (np.invert(np.isnan(Z))))
    Xd = Xd[valididx]
    Z = Z[valididx]

    # Summing Z within indices of Xd.
    # Converting into a full matrix again for accessing elements more conveniently

    sparse_M = sparse.csr_matrix( ( Z, (np.zeros(Xd.shape),Xd-1) ),shape = (1,nbins) ).toarray()

    return sparse_M

def Compute2DMap(Xd, Yd, Z, nXbins, nYbins):
    import numpy as np
    from scipy import sparse  
    # Compute 1D map efficiently. Xd is the binned independent varaible and Z is
    # the dependent variable.

    # Selecting valid indices
    valididx = ((np.invert(np.isnan(Xd))) & (np.invert(np.isnan(Yd))))
    Xd = Xd[valididx]
    Yd = Yd[valididx]
    Z = Z[valididx]

    # Summing Z within indices of Xd.
    # Converting into a full matrix again for accessing elements more conveniently

    return sparse.csr_matrix( ( Z ,( Yd-1, Xd-1) ),shape = (nXbins,nYbins) ).toarray()

def SpatialInfo(t,o):
    import numpy as np
    # Computes the spatial information in tuning curve t, considering an
    # occupancy o. This is returned in bits per spike.

    # Mean rate of the cell
    meanRate = np.nansum( np.multiply( t, (o/np.nansum(o)) ) )

    # Spatial information in bits per seconds.
    SInfo = np.multiply( (o/np.nansum(o)),t )
    SInfo = np.nansum( np.multiply( SInfo, np.log2(t / meanRate ) ) )

    # Converting in bits per spike.
    SInfoperspike = SInfo/meanRate

    return SInfoperspike

def FieldSparsity(t, o):
    import numpy as np
    # Computes the selectivity of a tuning curve t given an occupancy o as:
    # 1 - sparsity index. The sparsity index is defined as by Jung, Wiener and
    # McNaughton (JNS, 1994). Closer to one means sparser, so more selective

    return 1 - np.power(np.nansum(np.multiply(t,o/np.nansum(o))),2)/np.nansum(np.multiply(np.power(t,2),o/np.nansum(o)))

def FieldSelectivity(t):
    import numpy as np
    # Computes the selectivity of the tuning curve t as the maximal amplitude
    # normalized by the mean.

    return (np.nanmax(t)-np.nanmin(t)) / np.nanmean(t)

def crossvalPartition(n,kfold):
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

def normpdf_python(x, mu, sigma):
    import numpy as np
    # https://stackoverflow.com/questions/19913613/matlab-to-python-stat-equation
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(x-mu)**2/2*sigma**2)

def computeEV(y,ypred):
    import numpy as np
    RSS = np.nansum(np.power((y - ypred),2))
    m = np.nanmean(y)
    TSS = np.nansum(np.power((y - m),2))

    return 1 - (RSS/TSS)

def computeLLH_normal(y, ypred, k = None):
    import numpy as np
    # Compute the log likelihood for a Gaussian model. y is the original signal;
    # ypred, the model prediction and k is the total number of model parameters.
    # k is only necessary if the Bayesian Information Criterion and Akaike
    # Information Criterion are required.

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
        BIC = k * np.log(N) - 2 * LLH;
        AIC = 2 * k - 2 * LLH

    return LLH,BIC,AIC

def likelihood_ratio(llmax,llmin,dof):
    from scipy.stats.distributions import chi2

    LR = 2*(llmax-llmin)
    p = chi2.sf(LR, dof)

    return LR,p

def FieldDirectionality(t1, t2):
    import numpy as np
    # Computes the discrepancy between to tuning curves t1 and t2.
    first = abs(np.sum(t1-t2))
    second = np.sum(t1+t2)
    DirectionalityIndex = np.divide(first,second)

    return DirectionalityIndex

def computeLLH_poisson(y, ypred, k = None):
    from scipy import stats
    import numpy as np
    # Compute the log likelihood for a Poisson model. y is the original signal;
    # ypred, the model prediction and k is the total number of model parameters.
    # k is only necessary if the Bayesian Information Criterion and Akaike
    # Information Criterion are required.

    # probability density function for a normal distribution with std = s.
    pd = stats.poisson.pmf(y, ypred)
    pd[pd == 0] = 1e-16

    LLH = np.nansum(np.log(pd))

    # if k is provided, we also compute BIC and AIC of the model.
    if k is None:
        BIC = np.nan
        AIC = np.nan
    elif k > 2:
        N = len(y)
        BIC = k * np.log(N) - 2 * LLH
        AIC = 2 * k - 2 * LLH

    return LLH, BIC, AIC

def moving_average(a, n=3):
    import numpy as np
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def MapsAnalyses(Nav, Srep, mapsparams):
    """
    Function to perform the Maps analysis

    Parameters
    --------------
    Nav : dict
        pos data in dict
    Srep : array
        Spike train data
    mapsparams : dict
        dict with the maps parameters

    Returns
    --------------
    Maps : dict
        dict with all the map analysis variables
    """
    import os
    import mat73
    import numpy as np
    import pandas as pd
    import scipy.io
    import matplotlib.pyplot as plt
    from scipy import signal,stats,sparse,interpolate
    from scipy.ndimage import median_filter,gaussian_filter1d   
    # Selecting time over which to compute place fields
    tidx = (np.isin(Nav['Condition'],mapsparams['condition'])) &\
        (np.isin(Nav['XDir'],mapsparams['dir'])) &\
        (np.isin(Nav['laptype'],mapsparams['laptype'])) &\
        (Nav['Spd'] >= mapsparams['spdthreshold']) & \
        (np.invert(np.isnan(Nav['Xpos'])))

    # Selecting cell indices for which we'll run the analysis
    cellidx = np.argwhere((mapsparams['cellidx']) & (sum(Srep[tidx,:]) > mapsparams['nspk_th'])).reshape(-1)

    # Subsetting Spk['spikeTrain'] and Nav['Xpos']
    spikeTrain = Srep[:,cellidx]
    spikeTrain = spikeTrain[tidx,:]
    Xpos = Nav['Xpos'][tidx]

    nbins = len(mapsparams['Xbinedges'])-1 # number of position bins
    ntimepts = spikeTrain.shape[0] # number of data points

    # Discretizing position vectors according to mapsparams['binedges']
    Xpos_discrete = np.digitize(Xpos,bins =  mapsparams['Xbinedges'])
    Xpos_discrete[Xpos_discrete>25]=25

    # Computing occupancy map (same for all cells)
    flat = 1/mapsparams['sampleRate'] * np.ones(len(Xpos_discrete)) *2
    occmap = Compute1DMap(Xpos_discrete,flat,nbins)

    # Removing occupancy for positions below the occupancy threshold
    occmap[occmap <= mapsparams['occ_th']] = np.nan

    # Smoothing the occupancy map with a gaussian window (mapsparams['XsmthNbins'] of sd)
    occmap = gaussian_filter1d(occmap,mapsparams['XsmthNbins'])

    ncells = spikeTrain.shape[1] # number of cells selected for place field analysis
    scmap = np.empty((ncells,nbins))*np.nan # Initializing the spike count map

    # Computing and smoothing spike count map for each cell
    for icell in range(ncells):
        scmap[icell,:] = Compute1DMap(Xpos_discrete, spikeTrain[:,icell], nbins)
        scmap[icell,np.isnan(occmap)[0]] = np.nan
        scmap[icell,:] = gaussian_filter1d(scmap[icell,:], mapsparams['XsmthNbins'])

    # Calculating the place field maps by dividing scmap and occmap
    mapX = np.divide(scmap,occmap)

    # Initilizing the vectors
    SI = np.empty((ncells,1))*np.nan
    SparsityIndex = np.empty((ncells,1))*np.nan
    SelectivityIndex = np.empty((ncells,1))*np.nan

    # Calculating all metrics for each cell
    for icell in range(ncells):
        SI[icell] = SpatialInfo(mapX[icell,:], occmap)
        SparsityIndex[icell] = FieldSparsity(mapX[icell,:], occmap)
        SelectivityIndex[icell] = FieldSelectivity(mapX[icell,:])

    # Computing shuffle controls by randomly shifting time bins of positions and
    # calculate the selectivity metrics for each shuffle control
    SI_Shf = np.empty((ncells,mapsparams['nShuffle']))*np.nan
    SparsityIndex_Shf = np.empty((ncells,mapsparams['nShuffle']))*np.nan
    SelectivityIndex_Shf = np.empty((ncells,mapsparams['nShuffle']))*np.nan

    # Initializing the random number generator for reproducibility purposes
    rng = np.random.Generator(np.random.MT19937(seed = 0))

    # Calculating the place field for each shuffle permutation
    for iperm in range(mapsparams['nShuffle']):
        tshift = int(rng.integers(ntimepts - 2 * mapsparams['sampleRate']) + 1 * mapsparams['sampleRate'])
        Xpos_discrete_shf = np.roll(Xpos_discrete, tshift)
        for icell in range(ncells):
            scmap_shf = Compute1DMap(Xpos_discrete_shf,spikeTrain[:,icell],nbins)
            scmap_shf[np.isnan(occmap)] = np.nan
            scmap_shf = gaussian_filter1d(scmap_shf, mapsparams['XsmthNbins'])
            mapX_shf = np.divide(scmap_shf,occmap)

            # saving only the spatial selectivity metrics for each permutation
            SI_Shf[icell,iperm] = SpatialInfo(mapX_shf, occmap)
            SparsityIndex_Shf[icell,iperm] = FieldSparsity(mapX_shf, occmap)
            SelectivityIndex_Shf[icell,iperm] = FieldSelectivity(mapX_shf)

    # Computing p-values from the distribution of selectivity measures obtained from the shuffle controls
    SI_pval = np.nansum(SI_Shf> SI,1) / mapsparams['nShuffle']
    SparsityIndex_pval = np.sum(SparsityIndex_Shf > SparsityIndex, 1) / mapsparams['nShuffle']
    SelectivityIndex_pval = np.sum(SelectivityIndex_Shf > SelectivityIndex, 1) / mapsparams['nShuffle']

    # Defining a partition of the data for k-fold cross-validation
    cv = crossvalPartition(ntimepts, mapsparams['kfold'])

    # Computing the spike train predicted from the place field using k-fold cross-validation
    mapX_cv = np.empty((ncells, nbins,mapsparams['kfold']))*np.nan
    Ypred = np.empty((ntimepts,ncells))*np.nan

    for i in range(mapsparams['kfold']):
        # Subsetting Xpos and spiketrain according to the train set of the current fold
        Xtraining = Xpos_discrete[cv['trainsets'][i].astype(bool)]
        Spktraining = spikeTrain[cv['trainsets'][i].astype(bool),:]

        # Computing occupancy map for the current fold
        flat = 1/mapsparams['sampleRate'] * np.ones(Xtraining.shape) *2
        occmap_cv = Compute1DMap(Xtraining, flat, nbins)
        occmap_cv[occmap_cv <= mapsparams['occ_th']] = np.nan
        occmap_cv = gaussian_filter1d(occmap_cv, mapsparams['XsmthNbins'])

        # Computing the spike count map and place field of each cell for the current fold

        for icell in range(ncells):
            scmap_cv = Compute1DMap(Xtraining,
                                    Spktraining[:,icell],
                                    nbins)
            scmap_cv[np.isnan(occmap_cv)] = np.nan
            scmap_cv = gaussian_filter1d(scmap_cv, mapsparams['XsmthNbins'])
            mapX_cv[icell,:,i] = np.divide(scmap_cv,occmap_cv)

        # Subsetting Xpos and Dir according to the test set of the current fold
        Xtest = Xpos_discrete[cv['testsets'][i].astype(bool)]-1

        # Computing the spike train predicted on the test set from the place
        # computed from the train set
        for icell in range(ncells):
            Ypred[cv['testsets'][i].astype(bool),icell] = mapX_cv[icell,Xtest,i] / mapsparams['sampleRate']

    # Computing the spike train predicted from the mean firing rate of the
    # cell using the same k-fold partition as above
    Ypred_cst = np.empty((ntimepts,ncells))*np.nan
    for i in range(mapsparams['kfold']):
        for icell in range(ncells):
            Ypred_cst[cv['testsets'][i].astype(bool),icell] = np.nanmean(spikeTrain[cv['trainsets'][i].astype(bool),
                                                                         icell])

    # Computing the percentage of explained variance and the log likelihood from
    # the predicted responses.
    EV = np.empty((ncells,1))*np.nan
    LLH = np.empty((ncells,1))*np.nan
    EV_cst = np.empty((ncells,1))*np.nan
    LLH_cst = np.empty((ncells,1))*np.nan

    for icell in range(ncells):
        # Percentage of explained variance for the place field model
        EV[icell] =  computeEV(spikeTrain[:, icell], Ypred[:, icell])
        # Log likelihood for the place field model
        LLH[icell],_,_ = computeLLH_normal(spikeTrain[:, icell], Ypred[:, icell])
        # Percentage of explained variance for the constant mean model
        EV_cst[icell] = computeEV(spikeTrain[:, icell], Ypred_cst[:, icell])
        # Log likelihood for the constant mean model
        LLH_cst[icell],_,_ = computeLLH_normal(spikeTrain[:, icell], Ypred_cst[:, icell])

    # Comparing the place field model to the constant mean model by performing a
    # likelihood ratio test
    LLH_pval = np.empty((ncells,1))*np.nan
    goodidx = LLH > LLH_cst;
    LLH_pval[np.invert(goodidx)] = 1

    # difference of degree of freedom between the two models.
    dof = np.sum(occmap.reshape(-1) > 0) - 1

    # Likelihood ratio test for cells which have a higher LLH for the place
    # field model (the other one are definitely not significant).
    _,LLH_pval[goodidx] = likelihood_ratio(LLH[goodidx], LLH_cst[goodidx], dof)

    # %Computing the Jacknife estimate of the standard error
    for i in range(mapX_cv.shape[2]):
        mapX_cv[:,:,i] = np.power(mapX_cv[:,:,i] - mapX,2)
    mapX_SE = np.sqrt( (np.divide(mapsparams['kfold'] - 1,mapsparams['kfold'])) * np.sum(mapX_cv,2) )
    
    # populate the output dict
    mapsparams['tidx'] = tidx
    ncells_orig = Srep.shape[1]
    Maps = {'mapsparams':mapsparams}
    Maps['Xbincenters'] = mapsparams['Xbinedges'][:-1] + mapsparams['Xbinsize']/2
    nbins = len(Maps['Xbincenters'])
    Maps['mapX'] = np.empty((ncells_orig,nbins))*np.nan
    Maps['mapX_cv'] = np.empty((ncells_orig,nbins,mapsparams['kfold']))*np.nan
    Maps['occmap'] = np.empty((1,nbins))*np.nan

    Maps['mapX'][cellidx,:] = mapX
    Maps['mapX_cv'][cellidx,:] = mapX_cv
    Maps['mapX_SE'] = mapX_SE
    Maps['occmap'] = occmap
    Maps['SI'] = SI
    Maps['SparsityIndex'] = SparsityIndex
    Maps['SelectivityIndex'] = SelectivityIndex
    Maps['SI_pval'] = SI_pval
    Maps['SparsityIndex_pval'] = SparsityIndex_pval 
    Maps['SelectivityIndex_pval'] = SelectivityIndex_pval
    Maps['EV'] = EV
    Maps['EV_cst'] = EV_cst
    Maps['LLH'] = LLH
    Maps['LLH_cst'] = LLH_cst
    Maps['LLH_pval'] = LLH_pval

    return Maps

def DefineMapsParams(Nav,Spk):
    """
    Function to create the Maps params

    Parameters
    --------------
    Nav : dict
        pos data in dict
    Spk : dict
        dict with the spk data

    Returns
    --------------
    mapsparams : dict
        dict with the maps parameters
    """
    # create the dict
    mapsparams = {'condition':[1,3,5], # Experimental condition over which place fields will be estimated
                'dir':[-1,1], # -1,, # Directions over which place fields will be estimated
                'laptype':[-1,0,1],# lap types over which place fields will be estimated
                'spdthreshold':2.5, # Minimum speed threshold over which place fields will be computed
                'cellidx':np.ones(len(Spk['spikeTrain'][1])).astype(bool), # Subset of cells for which place fields will be computed
                'sampleRate':np.round((1/np.mean(np.diff(Nav['sampleTimes']))),2), # Sampling rate of the data
                'Xvariablename':'Xpos', # Name of the independent variable used to map the response along X.
                'Yvariablename':'XDir', #Name of the independent variable used to map the response along Y
                'Xrange':[0,100], # Range of positions over which place fields will be estimated (in cm)
                'Yrange':[-2,2], # Range of Y over which place fields will be estimated
                'Xbinsize':4, # Size of the position bins (in cm)
                'Ybinsize':2, # Size of the Y bins
                'Xsmthbinsize':2, # Size of the gaussian window for smoothing place fields (in cm)
                'Ysmthbinsize':.5, # Size of the gaussian window for smoothing place fields (in cm)
                'occ_th':0, # Occupancy threshold above which positions are included in the place field estimate
                'nspk_th':0, # Minimal number of spikes to consider a cell,
                'nShuffle':100, # Number of shuffle controls to perform for randomization
                'kfold':10 # Number of folds to consider for cross-validation
                }

    # Size of the gaussian window for smoothing place fields (in bins)
    mapsparams['XsmthNbins'] = mapsparams['Xsmthbinsize']/mapsparams['Xbinsize']
    mapsparams['YsmthNbins'] = mapsparams['Ysmthbinsize']/mapsparams['Ybinsize']
    # Edges of position bins used to discretize positions
    mapsparams['Xbinedges'] = np.arange(mapsparams['Xrange'][0], mapsparams['Xrange'][1]+mapsparams['Xbinsize'], mapsparams['Xbinsize'])
    mapsparams['Ybinedges'] = np.arange(mapsparams['Yrange'][0], mapsparams['Yrange'][1]+mapsparams['Ybinsize'], mapsparams['Ybinsize'])
    
    return mapsparams