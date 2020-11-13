# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:24:02 2020

@author: P70049035
"""
#try to get the rat data into mne python
# first import data into letswave and make the triggers there
# then export to eeglab with letswave 7 and then into mne
#Channels
#C1 = bipolar channel in MGB
#C2 = unipolar electrode on the vertex, referenced to the electrode behind the mastoid bone
#C3 = contains the triggers for the auditory stimuli
import mne
import numpy as np
from mne.viz import plot_events, plot_compare_evokeds
from mne import combine_evoked

path = 'C:/Users/p70049035/OneDrive/University/PhD/PhD_Studienstiftung/Maas_Rats/PB_ANALYSES/2_iso/'
eeglab_name = path + 'leveltrg data_channels_rat10_iso.set'
#path to the set file


raw = mne.io.read_raw_eeglab(eeglab_name, preload = True)
#works! nice :)
#how do I get the events?
print(raw.info)
print(raw.annotations)
#%%
raw.rename_channels({'C1': 'MGB', 'C2': 'Vertex', 'C3': 'Audio'})
raw.set_channel_types({'MGB': 'ecog', 'Vertex': 'eeg', 'Audio': 'stim'})
#%%
# set the location of the MGB electrode? AP -5.5, ML +3.6, DV -6
#%%
print(len(raw.annotations))
print(set(raw.annotations.duration))
print(set(raw.annotations.description))
print(raw.annotations.onset[0])
#%% convert annotations to events
events, event_ids = mne.events_from_annotations(raw)
print(event_ids)
print(events[:5])
#%% Band-pass filter
raw.filter(1,30, fir_design = 'firwin') #0.1
#raw.plot()
print(raw.info)
#%% see line noise
fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True)
# add some arrows at 60 Hz and its harmonics:
for ax in fig.axes[:2]:
    freqs = ax.lines[-1].get_xdata()
    psds = ax.lines[-1].get_ydata()
    for freq in (50, 100, 150, 200):
        idx = np.searchsorted(freqs, freq)
        ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
                 width=0.1, head_width=3, length_includes_head=True)
#%% artifact rejection in EEG channel - amplitude criterion

reject_criteria = dict(eeg=0.3e-6)       # 0.3 µV

#%% epochs
epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=-0.07, tmax=0.63,
                    baseline =(None, 0),
                    reject=reject_criteria, preload=True)
print(epochs)
#%%
# compute evoked response and noise covariance,and plot evoked
evoked = epochs.average ()
print(evoked)
#%%
title = 'MaasRats'
evoked.plot(titles=dict(eeg=title), time_unit='s')
evoked.plot_topomap(times=[0.1], size=3., title=title, time_unit='s')
#%%
S1 = epochs["1"].average()
S2 = epochs["2"].average()
S3 = epochs["3"].average()
S4 = epochs["4"].average()
S5 = epochs["5"].average()
S6 = epochs["6"].average()
all_evokeds = [S1, S2, S3, S4, S5, S6]
print(all_evokeds)
#%%
conditions = ['1', '2', '3']
evokeds = {condition:epochs[condition].average()
           for condition in conditions}
pick = evokeds['1'].ch_names.index('MGB')
plot_compare_evokeds(evokeds, picks=pick, ylim=dict(eeg=(-0.05, 0.05)))
#%%
epochs['1'].plot_image(picks=['MGB'])
#try git