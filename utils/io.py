import re
import os
import numpy as np

def awgn(s, SNRdB, L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal
    's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power
    spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
    if s.ndim == 1:  # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
    else:  # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
    N0 = P / gamma  # Find the noise spectral density
    if np.isrealobj(s):  # check if input is real/complex object type
        n = np.sqrt(N0 / 2) * np.random.standard_normal(s.shape)  # computed noise
    else:
        n = np.sqrt(N0 / 2) * (np.random.standard_normal(s.shape) + 1j * np.random.standard_normal(s.shape))
    r = s + n  # received signal
    return r


def load_loc_of_sensors(path, num_methods=5):
    with open(path, 'r') as file:
        text = file.read()
    segments = text.strip().split("number: ")

    # Dictionary to hold the final Data_total
    data_dict = {}

    # Process each segment
    for segment in segments:
        if segment:
            # Splitting each segment into lines
            lines = segment.split('\n')
            # First line is the number, which is the key1
            key1 = str(lines[0].strip())
            # Creating a dictionary for this key1
            data_dict[key1] = {}
            # Process the remaining lines
            for line in lines[1:]:
                if line:
                    # Splitting the line into key2 and its values
                    key2, values_str = line.split(': ')
                    # Converting the string of values into a list of integers
                    values = [int(v) for v in values_str.strip('[]').split(', ')]
                    # Adding to the dictionary
                    data_dict[key1][key2] = values
    return data_dict