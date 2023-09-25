def DefineMapsParams(): # TODO
    # Define a set of parameters needed to compute place fields.
    # 
    # INPUTS:
    # - Nav: a structure with at least a field called sampleTimes containing
    #   sample times of the data from which maps will be computed.
    # - Spk: a structure with at least a field called spikeTrain containing the
    #   time sereis of responses that will be mapped.
    # 
    # OUTPUT:
    # - mapsparams: a structure whose fields contain parameters to run
    #   MapsAnalysis1D and MapsAnalysis2D.
    # 
    # Fields of mapsparams are the following:
    # 
    # condition: experimental conditions over which place fields will be 
    # estimated.
    # 
    # dir: An array representing the directions over which place fields will 
    # be estimated.
    #  
    # laptype: An array defining lap types over which place fields will be 
    # estimated.
    # 
    # spdthreshold: The minimum speed threshold (in units of velocity) over 
    # which place fields will be computed.
    # 
    # cellidx: A logical array indicating a subset of cells for which place 
    # fields will be computed.
    # 
    # sampleRate: The sampling rate of the data (in Hz).
    #  
    # scalingFactor: A scaling factor applied to the response data, typically 
    # set to 1 / samplingRate to convert spiking data to spikes per second.
    #  
    # Xvariablename: The name of the independent variable used to map the 
    # response along the X-axis.
    #  
    # Xrange: The range of X values over which place fields will be estimated.
    # 
    # Xbinsize: The size of the X-axis bins.
    #  
    # Xsmthbinsize: The size of the Gaussian window for smoothing along the X-axis.
    # 
    # XsmthNbins: The number of bins used for smoothing place fields along the X-axis.
    # 
    # Xbinedges: The edges of position bins used to discretize the X-axis.
    # 
    # Yvariablename: The name of the independent variable used to map the 
    # response along the Y-axis.
    #  
    # Yrange: The range of Y values over which place fields will be estimated.
    # 
    # Ybinsize: The size of the Y-axis bins.
    # 
    # Ysmthbinsize: The size of the Gaussian window for smoothing place fields
    # along the Y-axis.
    # 
    # YsmthNbins: The number of bins used for smoothing place fields along the Y-axis.
    #  
    # Ybinedges: The edges of position bins used to discretize the Y-axis.
    # 
    # occ_th: An occupancy threshold in seconds above which positions are 
    # included in the place field estimate.
    # 
    # nspk_th: The minimal number of spikes required to consider a cell.
    #  
    # nShuffle: The number of shuffle controls performed for randomization.
    # 
    # kfold: The number of folds considered for cross-validation.
    # 
    # USAGE:
    # mapsparams = DefineMapsParams(Nav,Spk)
    # 
    # written by J.Fournier 08/2023 for the iBio Summer school
    # Adapted by Tulio Almeida
    params = {'sampleRate': 50,
            'sampleRate_rawLfp': 600,
            'pix2cm': 0.4300,
            'ShankList': [1,2,3,4],
            'LfpChannel_Hpc': 2,
            'LfpChannel_Bla': 2,
            'LfpChannel_Acc': 2,
            'ThetaBand': [6,9]}

    return params

def Compute1DMap(Xd,Z, nbins):
    # Compute 1D map efficiently. Xd is the binned independent varaible and Z is
    # the dependent variable.

    # Written by J. Fournier in August 2023 for the iBio Summer School.
    # adapted by Tulio Almeida 

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
    # Compute 2D map efficiently. Xd is the binned independent varaible and Z is
    # the dependent variable.

    # Written by J. Fournier in August 2023 for the iBio Summer School.
    # adapted by Tulio Almeida 

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
    # [m, sd, r] = ComputeTriggeredAverage(R, S, idxwin, w)
    #
    # Computes a triggered average of vector R based on timestamps in S over a
    # window of indices idxwin, weighted by w.
    #
    # INPUTS:
    # - R: 1D array from which to compute the average.
    # - S: list of indices from which to extract value of R.
    # - idxwin: list of indices around values of S.
    # - w (optional): list of the same size as S to weight R's snippets before
    # averaging.
    #
    # OUTPUTS:
    # - m: average of R triggered on indices in S on a window defined by idxwin.
    # - sd: standard deviation of the average m.
    # - r: snippets of R triggered on S. Each line correspond to one snippet.
    #
    # USAGE:
    # [m, sd, r] = ComputeTriggeredAverage(R, S, idxwin, [w]);
    #
    # written by J.Fournier 08/2023 for the iBio Summer school
    # adapted by Tulio Almeida 

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
    # GaussianSmooth1D - Smooth a 1D vector with a Gaussian kernel.

    # output = GaussianSmooth1D(input, smthNbins)
    # GaussianSmooth1D function applies Gaussian smoothing to a 1D input array using a
    # Gaussian kernel.

    # INPUTS:
    # input:      1D array to be smoothed.
    # smthNbins:  Standard deviation of the Gaussian kernel.

    # OUTPUT:
    # output:     Smoothed 1D array of the same size as the input.

    # USAGE:
    # output = GaussianSmooth1D(input, smthNbins);

    # SEE ALSO:
    # GaussianSmooth, Compute1DMap, Compute2DMap, MapsAnalyses1D,MapsAnalyses2D

    # Written by J. Fournier in August 2023 for the iBio Summer School.
    # adapted by Tulio Almeida 
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