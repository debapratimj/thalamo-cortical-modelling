# Mik
''' helpers.py

Contains amazing functions that allow for the analisis of in vivo ephys data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allego_file_reader.allego_file_reader as afr 
import spikeinterface as si
import spikeinterface.widgets as sw
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.extractors as se 
import spikeinterface.postprocessing as spost
import spikeinterface.exporters as sexp
import probeinterface as pi
import warnings, sys
from pathlib import Path
import os
from scipy.signal import butter, filtfilt

def getStim_idx(signals):
    ''' Takes the .xdat allego file, reads channel 32 (Analog IN) and determines where the TTL pulse that triggers microstim starts.

    Parameters:
    - signals: nChannelxTimepoints numpy matrix, representing neural signals. Run afr.read_allego_xdat_all_signals to generate.
    
    Returns:
    - TTL_idx: 2D numpy array, representing all TTLs that were given during the recording.
    '''
    # Select the AI1 channel
    ain_signal = signals[32]
    if any(ain_signal >= 4):
        TTL_idx = np.where(np.diff(ain_signal) >= 2)[0]+1 # +1 gives us the start of the trigger
    else:
        ValueError('No pulses found in channel 32')
        TTL_idx = False
    return TTL_idx

def get_periStim(signals, order=False, limit=10, start_time=0.5, stop_time=2, fSample=30000):
    '''
    Extracts stimulus-triggered snippets from signals.

    Parameters:
    - signals: 2D numpy array, representing neural signals.
    - order: False or list, specifying the order of channels.
    - limit: Maximum number of stimuli to consider.
    - start_time: Start time for snippet extraction (in seconds).
    - stop_time: Stop time for snippet extraction (in seconds).
    - fSample: Sampling frequency of the signal normally 30kHz.

    Returns:
    - periStim_df: Pandas DataFrame, each row representing a stimulus with signal snippets.
    '''
    # From signals get the TTL indexes of the stimuli
    TTL_idx = getStim_idx(signals)
    if limit == False:
        limit = len(TTL_idx)   
    TTL_idx = TTL_idx[0:limit] # TODO what if you want to limit the view to the middle or the end?

    # Create a dataframe where each row refers to a stimulus and has a array of voltage values
    periStim_df = pd.DataFrame(index=range(limit), columns=['signal'])
    periStim_df['signal'] = periStim_df['signal'].apply(lambda x: {})
    
    # Go through all stimuli, snip the signal for each channel and 
    for i, TTL in enumerate(TTL_idx):
        periStim_data = periStim_df['signal'].iloc[i]   

        # Snip the signal 0.5s before and 2s after stimulus
        start = int(TTL-start_time*fSample)
        stop = int(TTL+stop_time*fSample)
        nChannels=16 # TODO hardcode
        nSamples = stop-start

        # If you want the signals order in the dictionary determine the order based on channel_idx
        if order == False:
            for i, signal in enumerate(signals[0:nChannels]):
                periStim_signal = signals[i][start:stop] # TODO only one channel # i is the channel start stop the snippet
                periStim_data[i] = periStim_signal

        elif isinstance(order, list):
            for i in order:
                periStim_signal = signals[i-1][start:stop] # -1 because the order is in channel index which starts from 1
                periStim_data[i] = periStim_signal 
                
        periStim_df['signal'][i] = periStim_data
    return periStim_df

def common_referencing(data, operator='median', nChannel=16):
    '''
    Applies common average referencing to the data.

    Parameters:
    - data: 2D numpy array, representing neural signals.
    - nChannel: Number of channels.

    Returns:
    - car_data: 2D numpy array, data after common average referencing.
    '''
    if operator == 'median':
        # Calculate the average across all channels
        average_across_priChannels = np.median(data[0:nChannel], axis=0)
    elif operator == 'mean':
        average_across_priChannels = np.mean(data[0:nChannel], axis=0)

    # Subtract the average from each channel
    car_data_priChannels = data[0:nChannel] - average_across_priChannels

    # Now add the normal channels back again
    car_data = np.vstack((car_data_priChannels, data[nChannel:]))
    return car_data

def bandpass_filter(data, lowcut, highcut, fs, nChannels=16, order=4):
    '''
    Applies a bandpass filter to the data.

    Parameters:
    - data: 2D numpy array, representing neural signals.
    - lowcut: Lower cutoff frequency for the filter.
    - highcut: Higher cutoff frequency for the filter.
    - fs: Sampling frequency of the signal.
    - nChannels: Number of channels.
    - order: Order of the bandpass filter.

    Returns:
    - filtered_data: 2D numpy array, data after bandpass filtering.
    '''
    rec_data = data[0:nChannels]

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply bandpass filter to recording channels
    filtered_recdata = filtfilt(b, a, rec_data, axis=1)

    # Concatenate the filtered recordning channels with the analog and digital ones
    filtered_data = np.vstack((filtered_recdata, data[nChannels:]))
    return filtered_data

# def createProbe(metadata):
#     '''
#     Creates a probe object based on metadata.

#     Parameters:
#     - metadata: Dictionary containing information about the probe.

#     Returns:
#     - probe: Probe object.
#     '''
#     # TODO: Shank_ids, and radius have to be extracted from the meta file in stead of hardcoded.

#     # Now that we have loaded the metadata, extract the probe information from this file
#     # Probe information needed (for Kilosort3) = chanMap, chanMap0ind, connected, kcoords, name, xcoords and ycoords
#     # What I think is needed in Spikeinterface for probeinterface based on tutorials and example NN probe: contact positions, plane axes? shapes, shape params (radius), planar contour

#     # First we get essential probe information
#     # Get nChannel for off incorrect channels, specifying shank ID 
#     nChannel = metadata['sapiens_base']['sensors_by_port']['A']['num_channels']

#     # SamplingFrequency
#     fSample = metadata['status']['samp_freq']

#     # Positions
#     xcoords = metadata['sapiens_base']['biointerface_map']['site_ctr_tcs_x']
#     ycoords = [abs(y) for y in metadata['sapiens_base']['biointerface_map']['site_ctr_tcs_y']]
#     positions = [(x,y) for x,y in zip(xcoords, ycoords)] 

#     # ProbeName 
#     probeName_raw = metadata['sapiens_base']['sensors_by_port']['A']['probe_id']
#     probeName = probeName_raw.split('__')[1]

#     # Units
#     si_units = metadata['sapiens_base']['biointerface_map']['sensor_units'][0]

#     # Now we create a probe object with ndim2 since it is a linear probe
#     probe = pi.Probe(ndim=2, si_units=si_units, name=probeName, manufacturer='NeuroNexus') # TODO where is the serialnumber in the metafile? maybe do manually?
#     probe.set_contacts(positions=positions[0:nChannel], shapes='circle', shape_params={'radius':50}, shank_ids=np.zeros(nChannel, dtype=int), contact_ids=np.arange(1,nChannel+1,1))
#     probe.set_device_channel_indices(np.arange(0, nChannel, 1))
#     return probe

def createProbe(target_file):
    '''
    Creates a probe object and returns a DataFrame based on metadata.

    Parameters:
    - metadata: Dictionary containing information about the probe.

    Returns:
    - probe: Probe object.
    - probe_df: DataFrame representing the probe information.
    - channel_order: List of contact_ids sorted by 'y' coordinate.
    '''
    # Extract metadata information
    metadata, nChannel, fSample, _, _ = extract_metadata_info(target_file)

    # Positions
    xcoords = metadata['sapiens_base']['biointerface_map']['site_ctr_tcs_x']
    ycoords = [abs(y) for y in metadata['sapiens_base']['biointerface_map']['site_ctr_tcs_y']]
    positions = [(x, y) for x, y in zip(xcoords, ycoords)]

    # ProbeName
    probeName_raw = metadata['sapiens_base']['sensors_by_port']['A']['probe_id']
    probeName = probeName_raw.split('__')[1]

    # Units
    si_units = metadata['sapiens_base']['biointerface_map']['sensor_units'][0]

    # Create probe object
    probe = pi.Probe(ndim=2, si_units=si_units, name=probeName, manufacturer='NeuroNexus')  # TODO where is the serialnumber in the metafile? maybe do manually?
    probe.set_contacts(positions=positions[0:nChannel], shapes='circle', shape_params={'radius': 50},
                       shank_ids=np.zeros(nChannel, dtype=int), contact_ids=np.arange(1, nChannel + 1, 1))
    probe.set_device_channel_indices(np.arange(0, nChannel, 1))

    # Create DataFrame from probe object
    probe_df = probe.to_dataframe()

    # Sort DataFrame by 'y' coordinate
    channel_order = probe_df.sort_values(by='y', ascending=True)['contact_ids'].astype(int).to_list()

    return probe, probe_df, channel_order

def detect_spikes(single_trace, fSample=30000, threshold_multiplier=3, window_size_ms=2, stim_window=[0.5, 0.7]):
    '''
    Detect spikes in a single trace.

    Parameters:
    - single_trace: 1D numpy array, representing the neural signal trace.
    - fSample: Sampling frequency of the signal.
    - threshold_multiplier: Multiplier for determining the threshold.
    - window_size_ms: Size of the window for marking the first occurrence of each spike (in milliseconds).

    Returns:
    - spikes_first_occurrence: 1D numpy array, marking the first occurrence of each spike within the specified window.
    '''
    # Get stim window
    stim_start_t = stim_window[0] - 0.01 
    stim_end_t = stim_window[1] + 0.01
    stim_start = int(stim_start_t * fSample)
    stim_end = int(stim_end_t * fSample)

    # Calculate mean and standard deviation of the pre-stimulus period
    pre_stim = int(stim_start)
    signal_mean = np.mean(single_trace[0:pre_stim])
    signal_std = np.std(single_trace[0:pre_stim])

    # Determine threshold values
    threshold_pos = signal_mean + (threshold_multiplier * signal_std)
    threshold_neg = signal_mean - (threshold_multiplier * signal_std)

    # Count threshold crosses
    spikes_pos = single_trace > threshold_pos
    spikes_neg = single_trace < threshold_neg
    spikes = spikes_pos | spikes_neg

    # Remove spikes that were counted during the stimulus
    spikes[stim_start:stim_end] = [False] * (stim_end - stim_start)

    # Create a new array to mark only the first occurrence of each spike within a window
    window_size = int(fSample * (window_size_ms / 1000))  # Convert window size to samples
    spikes_first_occurrence = np.zeros_like(spikes)

    for i in range(0, len(spikes), window_size):
        if np.any(spikes[i:i + window_size]):
            first_spike_index = i + np.argmax(spikes[i:i + window_size])
            spikes_first_occurrence[first_spike_index] = True

    return {'spikes':spikes_first_occurrence, 'mean':signal_mean, 'std':signal_std, 'threshold':(threshold_neg, threshold_pos)}

def get_spike_matrix(periStim_df, channel_index, threshold_multiplier=3.5):
    """
    Extracts spike information for a specified channel from stimuli in the periStim DataFrame.

    Parameters:
    - periStim_df: DataFrame containing stimulus signals.
    - channel_index: Index of the channel to extract spikes from.
    - threshold_multiplier: Multiplier for determining the threshold.

    Returns:
    - spike_array: List containing spike information for each stimulus.
    """
    # Create an empty list to store spike information for all stimuli
    spike_matrix = []

    # Iterate through all stimuli in the periStim DataFrame
    for signal in periStim_df['signal']:
        # Extract the signal for the specified channel
        stimsignal = signal[channel_index]

        # Detect spikes in the channel and append the spike information to the array
        signal_spike_dict = detect_spikes(stimsignal, threshold_multiplier=threshold_multiplier)
        signal_spikes = signal_spike_dict['spikes']
        spike_matrix.append(signal_spikes)

    return np.array(spike_matrix)

def extract_metadata_info(target_file):
    # Extract metadata
    metadata = afr.read_allego_xdat_metadata(target_file)

    # Extract relevant information from metadata
    nChannel = metadata['sapiens_base']['sensors_by_port']['A']['num_channels']
    fSample = metadata['status']['samp_freq']
    time_start, time_end = metadata['status']['t_range']

    return metadata, nChannel, fSample, time_start, time_end

