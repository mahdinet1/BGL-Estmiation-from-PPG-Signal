import numpy as np
from scipy import signal
import numpy as np
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import neurokit2 as nk
import os
import pandas as pd
import scipy.io

# Assuming your signal is stored in a numpy array called 'original_signal'
# and the sampling rate of the original signal is 2157 Hz

original_sampling_rate = 2157  # Hz
target_sampling_rate = 100  # Hz
def resample_ppg_signal(ppg):
  # Calculate the resampling factor
  resample_factor = original_sampling_rate / target_sampling_rate

  # Calculate the number of samples in the resampled signal
  num_samples_resampled = int(len(ppg) / resample_factor)

  # Generate the time array for the resampled signal
  time_resampled = np.arange(num_samples_resampled) / target_sampling_rate

  # Resample the signal using scipy's resample function
  resampled_signal = signal.resample(ppg, num_samples_resampled)
  return resampled_signal


def find_windows(resampled_signal):
  ppg_signal = resampled_signal.reshape((-1,))
  # Generate a simulated PPG signal (can be replaced with actual data)
  fs = 100  # Sampling frequency
  # Peak detection parameters
  peak_height = 4
  peak_distance = int(fs * 0.8)  # Initial peak distance
  window_duration = 1



  # Iterate through different window durations
  # plt.plot(ppg_signal)
  window_size = int(window_duration * fs)
  ppg_signal = nk.ppg_clean(ppg_signal,sampling_rate=100)

  # Peak detection
  peaks, _ = find_peaks(ppg_signal, height=peak_height, distance=peak_distance)

  # Select windows with exactly one peak at the center
  windows = []
  for peak in peaks:
      window_start = max(0, peak - window_size // 2)
      window_end = min(len(ppg_signal), peak + window_size // 2)
      window = ppg_signal[window_start:window_end]
      if len(find_peaks(window, height=peak_height, distance=peak_distance)[0]) == 1 and len(window) == 100:

          windows.append(window)
  return windows
    # Evaluate peak detection performance
from sklearn.preprocessing import MinMaxScaler
def scale_to_range(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def test_model():
    import torch
    # from resnet import resnet34
    # from vgg16 import VGG16
    from cnnLstmAtt import CustomModel
    res_model = CustomModel()
    res_model.load_state_dict(torch.load("C:\\Users\\mahdi\\Documents\\models-results\\1s\\cnnLstm\\cnnLstmAtt-1s.pth",map_location=torch.device('cpu')))
    res_model.to('cpu')
    preds = []
    res_model.eval()
    with torch.no_grad():
        subjects = os.listdir("maz_data")
        for subj in subjects:
            files = os.listdir(os.path.join('maz_data',subj))
            def filter_ppgs(file_name):
                if file_name.startswith("mazandaran"):
                    return True
                else:
                    False
            ppgs = filter(filter_ppgs,files)
            for file in ppgs:
                ppg = np.load(os.path.join("maz_data",subj,file))
                bgl = float(file.split(".")[0].split("-")[-1])
                # bgl = torch.from_numpy(bgl)
                if (ppg.shape[-1] != 100):
                    continue
                ppg = torch.Tensor(ppg)
                min_vals = ppg.min(dim=1, keepdim=True)[0]
                max_vals = ppg.max(dim=1, keepdim=True)[0]
                scaled_tensor = 2 * (ppg - min_vals) / (max_vals - min_vals) - 1
                scaled_tensor = scaled_tensor.reshape((-1,1,100))
                outputs = res_model(scaled_tensor)
                mean_outputs = torch.mean(outputs,dim=0)
                preds.append((subj,bgl,mean_outputs.item()))

        np.save("maz_preds_cnnlstmatt",preds)        




def main():

    raw_datas = os.listdir("PPG_Dataset\RawData")
    raw_datas = [data for data in raw_datas if data.split(".")[-1] == 'mat']
    labels = os.listdir("PPG_Dataset\Labels")
    labels =  [data for data in labels if data.split(".")[-1] == 'csv' and data.startswith("label")]
    proccess_data = []

    for data,label_file in zip(raw_datas,labels):
        label = pd.read_csv(os.path.join("PPG_Dataset\Labels",label_file))
        bgl = label['Glucose'].values[0]
        id =  label['ID'].values[0]
        ppg = scipy.io.loadmat(os.path.join('PPG_Dataset\RawData',data))['signal']

        # print(id,label['Glucose'].values)
        resampled_ppg = resample_ppg_signal(ppg)

        # print(max(cleaned))
        fined_windows = find_windows(resampled_ppg)
        if not os.path.isdir(f"maz_data\{id}"):
            os.mkdir(f"maz_data\{id}")
        np.save(f"maz_data\{id}\\bgl-{bgl}",bgl)
        np.save(f"maz_data\{id}\mazandaran_data_proccessed-{bgl}",np.array(fined_windows))

    # print(proccess_data)
    
if __name__=="__main__":
#    main()
    test_model()
    # np.load("maz_preds.npy")
    print("hi")
