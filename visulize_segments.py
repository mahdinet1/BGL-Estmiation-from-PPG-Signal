import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

# plt.style.use('seaborn-darkgrid')  # Use a built-in style
def main():
    sample1 = np.load("maz_data\\1\\mazandaran_data_proccessed-99.npy")
    sample2 = np.load("maz_data\\2\\mazandaran_data_proccessed-111.npy")
    print(sample1.shape)
    fig = plt.figure(figsize=(10, 8),dpi=500)
    plt.plot(sample1[3],label="BGL= 99 mg/dl",linewidth=2)
    # plt.plot(sample2[0])
    # plt.legend(["99", "111"])
    plt.xlabel("Sample",fontsize=20, fontweight='bold')
    plt.ylabel("Amplitude",fontsize=20, fontweight='bold')
    plt.title("1-Second Segment of PPG Signal ()",fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', labelsize=18, )
    plt.annotate("Systolic", xy=(50, 11), xytext=(22, 8),
          arrowprops=dict(arrowstyle="->"),size=30)
    plt.annotate("Diastolic", xy=(82, -0.5), xytext=(75, 4),
          arrowprops=dict(arrowstyle="->"),size=30)
    plt.savefig("1sec-segment.png",dpi=500)
    plt.show()
    
main()