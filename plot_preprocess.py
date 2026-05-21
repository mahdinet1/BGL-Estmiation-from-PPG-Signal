import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
# import vitaldb
plt.rcParams["font.family"] = "Times New Roman"
track_names = ['SNUADC/PLETH']

# vf = vitaldb.VitalFile(50, track_names)
# np.save("sample-raw2-500",vf.to_numpy(track_names,1/500))
# np.save("sample-raw2-100",vf.to_numpy(track_names,1/100))

plt.rcParams['figure.constrained_layout.use'] = True
x_raw = np.linspace(0,32,960000)
raw = np.load("sample-raw2-500.npy")[:960000]

x_resampled = np.linspace(0,32,192000)
resampled = np.load("sample-raw2-100.npy")[:192000]
x_filtered = x_resampled
filtered = nk.ppg_clean(resampled,sampling_rate=100)

fig = plt.figure()
# fig.tight_layout()
# stage 1 raw data with 500hz
ax1 = fig.add_subplot(5,1,1)
ax2 = fig.add_subplot(5,1,2)
ax3 = fig.add_subplot(5,1,3)
ax4 = fig.add_subplot(5,1,4)
ax5 = fig.add_subplot(5,1,5)


# ax3 = fig.add_subplot(3,1,3)
ax1.title.set_text("32 min Raw Signal")
ax2.title.set_text("32 min Resampled Signal")
ax3.title.set_text("32 min Resampled Signal with Specefic Point that BGL is Measured (dt)")
ax4.title.set_text("Specefic Point that BGL is Measured (dt) and Croped Windows")
ax5.title.set_text("Filtered Croped Window")

ax1.plot(x_raw,raw)
ax2.plot(x_resampled,resampled)
ax3.plot(x_resampled,resampled)
ax3.axvline(5,color='r')
# ax3.text(21.5,-280,'dt',rotation=0)
ax4.plot(x_resampled,resampled)


ax4.axvline(22,color='r')
ax4.text(21.5,-280,'dt',rotation=0)
ax4.axvspan(14,30,color="#66bb60",alpha=0.6)

ax5.plot(x_filtered[84000:240000],filtered[84000:240000])


# stage 2 resample data
# resampled = vf.to_numpy(track_names,1/100)[1000:2000]
plt.savefig("proccessing.png",dpi=300)
plt.show()

