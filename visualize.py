import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from photometry_functions import *
from sklearn.linear_model import Lasso

data = pd.read_csv('/Users/xiaocao/Downloads/Python/Photometry_data_processing-master/example.csv')
data = data[100:]

exc_signal = data['MeanInt_470nm']
t_exc = data['Time_470nm']
ref_signal = data['MeanInt_410nm']
t_ref = data['Time_410nm']

# smooth signal
exc_signal_s = smooth_signal(exc_signal)
ref_signal_s = smooth_signal(ref_signal)
# airPLS
exc_signal_base = airPLS(exc_signal_s,50000,1,15)
ref_signal_base = airPLS(ref_signal_s,50000,1,15)
exc_subtracted = exc_signal_s - exc_signal_base
ref_subtracted = ref_signal_s - ref_signal_base
# normalize
exc_norm = (exc_subtracted - np.median(exc_subtracted))/np.std(exc_subtracted)
ref_norm = (ref_subtracted - np.median(ref_subtracted))/np.std(ref_subtracted)
# regression
lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000, positive=True, random_state=9999, selection='random')
lin.fit(ref_norm.reshape(len(ref_norm),1),exc_norm.reshape(len(ref_norm),1))
ref_reg = lin.predict(ref_norm.reshape(len(ref_norm),1)).reshape(len(ref_norm),)
dff = exc_norm - ref_reg

plt.subplot(2,1,1)
plt.plot(t_exc,exc_signal,label='Raw')
plt.plot(t_exc,exc_signal_s,label='Smooth')
plt.plot(t_exc,exc_signal_base,label='Base')
plt.title('Smooth and Baseline')

plt.subplot(2,1,2)
plt.plot(t_ref,ref_signal,label='Raw')
plt.plot(t_ref,ref_signal_s,label='Smooth')
plt.plot(t_ref,ref_signal_base,label='Base')
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence intensity (au)')
plt.legend()
plt.savefig('/Users/xiaocao/Downloads/Python/Photometry_data_processing-master/fig_1.png',dpi=300)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t_exc,exc_subtracted,label='Corrected')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t_ref,ref_subtracted,label='Corrected')
plt.legend()
plt.savefig('/Users/xiaocao/Downloads/Python/Photometry_data_processing-master/fig_2.png',dpi=300)

plt.figure()
plt.subplot(2,1,1)
plt.plot(t_exc,exc_norm,label='Normalized')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t_ref,ref_norm,label='Normalized')
plt.legend()
plt.savefig('/Users/xiaocao/Downloads/Python/Photometry_data_processing-master/fig_3.png',dpi=300)

plt.figure()
plt.subplot(2,1,1)
plt.scatter(ref_norm,exc_norm,s=4)
plt.plot(ref_norm,ref_reg,c='r')

plt.subplot(2,1,2)
plt.plot(t_ref,dff)
plt.savefig('/Users/xiaocao/Downloads/Python/Photometry_data_processing-master/fig_4.png',dpi=300)
