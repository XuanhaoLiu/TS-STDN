# This is code to manufacture contaminated EEG signals
import os
import numpy as np
import math
import numpy as np

###################################### Hyperparameters ###########################################
C = 62                      # the number of channels
T = 400                     # the time samples of EEG signals
##################################################################################################

def Get_subject(f):
    # get subject index from the file name
    if(f[1] == '_'):
        return int(f[0])
    return int(f[0])*10 + int(f[1])

def rms(x):
    return np.sqrt(x.dot(x)/x.size)

############################### Adding_noise ################################
# x is the single channels eeg signals
# n is the noise adding to x
# SNR is the signal-noise-rate
#############################################################################
def add_single_noise(x, n, SNR):
    rmsn = rms(n)
    rmsx = rms(x)
    k = (rmsx/rmsn) / math.exp(SNR / 10)
    return x + k * n

############################### Adding_noise ################################
# x is the raw eeg signals with 62 channels
# n is the noise adding to x
# SNR is the signal-noise-rate
#############################################################################
def Adding_noise(eeg, n, SNR):
    # Adding EMG noise to EEG signals
    # We will add the noise by following way according to observation:
    # We seperate EEG electrodes into 10 groups, the groups are:
    # (F7, F5, F3, F1),     (F2, F4, F6, F8)
    # (FT7, FC5, FC3, FC1), (FC2, FC4, FC6, FT8)
    # (T7, C5, C3, C1),     (C2, C4, C6, T8)
    # (TP7, CP5, CP3, CP1), (CP2, CP4, CP6, TP8)
    # (P7, P5, P3, P1),     (P2, P4, P6, P8)
    # We will randomly choose 6 groups out of 10 to add noise to simulate various conditions
    # grp_sta is the start electordes num of each group
    grp_sta = [5, 10, 14, 20, 23, 28, 32, 37, 41, 46]
    # c is the random choose of 6 groups
    a = np.array(range(10))
    c = np.zeros((10))
    np.random.shuffle(a)
    for i in range(6):
        c[a[i]] = 1
    for i in range(10):
        if(c[i] == 1):
            for j in range(4):
                eeg[grp_sta[i]+j] = add_single_noise(eeg[grp_sta[i]+j], n, SNR)
    return eeg

# readin noise data
noise_data = np.load('./data/noisydata_200Hz/' + 'EMG_all_epochs_200Hz.npy')

# print(noise_data.shape)      
# print(f'means : {np.mean(noise_data[:10], axis=1)}')
# print(f'vars : {np.var(noise_data[:10], axis=1)}')
# print(f'means : {np.mean(noise_data)}')
# print(f'vars : {np.var(noise_data)}')

for session in range(1, 4):
    print(f'session = {session} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
    Session_file_path = './data/SEED_segmented_1s1sample/'+str(session)+'/'
    files = os.listdir(Session_file_path)
    for f in files:
        subject_num = Get_subject(f)
        print(f'subject_num = {subject_num}')
        
        # read in the raw EEG signals
        train_data = np.load(Session_file_path+f+'/train_data.npy')
        train_label = np.load(Session_file_path+f+'/train_label.npy')
        test_data = np.load(Session_file_path+f+'/test_data.npy')
        test_label = np.load(Session_file_path+f+'/test_label.npy')
        
        # for each SNR intensity create a noisy EEG
        for snri in range(-8, -7):
            print(f'create session{session}, subject{subject_num}, snri {snri}')
            
            # create the save path file
            save_path = './data/NoisyEEG_200Hz/'+str(snri)+'db/'+str(session)+'/'+f +'/'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            
            a = np.array(range(noise_data.shape[0]))
            
            # a is a random permutation of range(noiselen)
            np.random.shuffle(a)
            for i in range(train_data.shape[0]):
                train_data[i] = Adding_noise(train_data[i], noise_data[a[i]], snri)
            
            # a is a random permutation of range(noiselen)
            np.random.shuffle(a)
            for i in range(test_data.shape[0]):
                test_data[i] = Adding_noise(test_data[i], noise_data[a[i]], snri)
            
            np.save(save_path + 'train_data.npy', train_data)
            np.save(save_path + 'train_label.npy', train_label)
            np.save(save_path + 'test_data.npy', test_data)
            np.save(save_path + 'test_label.npy', test_label)