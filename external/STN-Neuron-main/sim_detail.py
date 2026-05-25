from stn_detail import CreateSth
import matplotlib.pyplot as plt
import time
import pandas as pd
import statistics
from statistics import mean
import efel
from scipy.signal import find_peaks
import math
import warnings
import pickle
import numpy as np
from neuron import h
import neuron as nrn
from neuron.units import ms, mV


def create_updated_cell(f, morphology):
    stn_cell = CreateSth(params=f, morphology=morphology)
    return stn_cell


def get_freq(v, dt):
    peaks = find_peaks(v, height=0)
    diff_p = np.diff(peaks[0])
    if detect_burst(v):
        mean_freq = 0
    elif len(diff_p) >= 1:
        mean_freq = 1000/dt/mean(diff_p)
    else:
        mean_freq = 0
    return mean_freq


def detect_burst(v):
    peaks = find_peaks(v, height=0)
    diff_p = np.diff(peaks[0])
    bursting = False
    for i in range(len(diff_p)):
        if len(diff_p) > 1:  # Should have at least 3 spikes
            if i - 1 < 0:
                i = 1
            if diff_p[i] / diff_p[i - 1] > 1.1:  # This will find the pause between two bursts
                bursting = True
            if diff_p[i - 1] / diff_p[i] > 1.1:
                bursting = True
    return bursting


def get_freq_detect_burst(v, dt):
    freq = get_freq(v, dt)
    bursting = detect_burst(v)
    if bursting:
        freq = 0
    return freq


def cal_score_HP(v, weight, v_rest, dt):
    spikes = find_peaks(v[int(112.5/dt):int(600/dt)], height=-80)
    if len(spikes[0]) == 0:
        v_HP = min(v[int(112.5/dt):int(600/dt)])
        v_ht = v_HP+(v[int(575/dt)]-v_HP)/2+1
        e_slp = v[int(575/dt)] - v_HP
    else:
        v_HP = 0
    spikes_all = find_peaks(v[24000:], height=-20)
    return v_HP


def cal_AP_width(time, voltage):
    trace1 = {}
    trace1['T'] = time
    trace1['V'] = voltage
    trace1['stim_start'] = [500]
    trace1['stim_end'] = [1000]
    traces = [trace1]
    traces_results = efel.get_feature_values(traces, ['AP2_width'])
    for trace_results in traces_results:
        # trace_result is a dictionary, with as keys the requested eFeatures
        for feature_name, feature_values in trace_results.items(): 
            if feature_name == 'AP2_width':
                spike_half_width = np.mean(feature_values)
    return spike_half_width


def check_AHP(v):
    v_AHP = min(v)
    return v_AHP


def check_peak(v):
    v_peak = max(v)
    return v_peak


def check_rest(v):
    peaks = find_peaks(v, height=0)
    if len(peaks[0]) == 0:
        v_rest = np.mean(v)
    else:
        intervals = peaks[0]
        checkpoints = []
        for i in range(len(intervals)-1):
            point = (intervals[i]+intervals[i+1])/2
            v_r = v[int(point)]
            checkpoints.append(v_r)
        v_rest = np.mean(checkpoints)
    return v_rest


def report_input_impedance(stn_cell, temp):
    h.celsius = temp
    h.finitialize()
    z = h.Impedance()
    z.loc(0.5, sec = stn_cell.soma[0])
    z.compute(0)
    z_value = z.input(0.5, sec = stn_cell.soma[0])
    return z_value


def output_sptime_params(f_index, score):
    with open('temp_full/parallelresults{}.txt'.format(f_index),'w') as f:
        f.write(str(f_index)+'\n'+str(score)+'\n')


def run_cost_simulation(f_index, morphology, plotting=False):
    # st = time.time()
    f = f_index
    h.dt = 0.025
    h.celsius = 37

    # Check input resistance
    stn_cell = create_updated_cell(f, morphology)
    z_value = report_input_impedance(stn_cell, 37)
    print(f'Input Resistence: {z_value:.2f} MΩ')

    # Check tonic spiking
    stn_cell = create_updated_cell(f, morphology)
    soma_v = h.Vector().record(stn_cell.soma[0](0.5)._ref_v)
    soma_t = h.Vector().record(h._ref_t)
    h.finitialize()
    h.continuerun(1500 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_0 = t[int(1000/h.dt):]
    v_0 = v[int(1000/h.dt):]
    soma_v.clear()
    soma_t.clear()
    freq_sp = get_freq_detect_burst(v_0, h.dt)

    # Save State
    svstate = h.SaveState()
    svstate.save()

    # Check AP shape
    shw = cal_AP_width(t_0, v_0)
    v_AHP = check_AHP(v_0)
    v_rest = check_rest(v_0)
    v_peak = check_peak(v_0)

    # Check hyperpolarization current injection
    svstate.restore()
    stim = h.IClamp(stn_cell.soma[0](0.5))
    stim.delay = 1600
    stim.dur = 500
    stim.amp = -0.1
    h.continuerun(2500 * ms)
    v_3 = soma_v.to_python()
    t_3 = soma_t.to_python()
    soma_v.clear()
    soma_t.clear()
    v_HP = cal_score_HP(v_3, 200, v_rest, h.dt)

    # Check FI curve
    svstate.restore()
    stim = h.IClamp(stn_cell.soma[0](0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.1
    h.continuerun(2500 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_1 = t[int(500/h.dt):]
    v_1 = v[int(500/h.dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi1 = get_freq_detect_burst(v_1, h.dt)

    svstate.restore()
    stim = h.IClamp(stn_cell.soma[0](0.5))
    stim.delay = 1500
    stim.dur = 1500
    stim.amp = 0.16
    h.continuerun(3000 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_2 = t[int(500/h.dt):]
    v_2 = v[int(500/h.dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi2 = get_freq_detect_burst(v_2, h.dt)

    svstate.restore()
    stim = h.IClamp(stn_cell.soma[0](0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.04
    h.continuerun(2500 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_4 = t[int(500/h.dt):]
    v_4 = v[int(500/h.dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi3 = get_freq_detect_burst(v_4, h.dt)

    svstate.restore()
    stim = h.IClamp(stn_cell.soma[0](0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.2
    h.continuerun(2500 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_5 = t[int(500/h.dt):]
    v_5 = v[int(500/h.dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi4 = get_freq_detect_burst(v_5, h.dt)

    freq_fi = [freq_sp, freq_fi3, freq_fi1, freq_fi2, freq_fi4]


    if plotting:
        print(f'half spike width: {shw:.2f} ms')
        print(f'AHP: {v_AHP:.2f} mV')

        plt.figure()
        plt.plot(t_0, v_0)
        plt.title(f'Spontaneous Spiking {freq_sp:.2f}Hz')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')

        plt.figure()
        plt.plot(t_3, v_3)
        plt.title('Hyperpolarization Current -0.1nA')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')

        plt.figure()
        plt.plot([0, 0.04, 0.1, 0.16, 0.2], freq_fi)
        plt.title('FI Curve')
        plt.xlabel('Current Injected (nA)')
        plt.ylabel('Freqency (Hz)')

        plt.show()
    
    return


if __name__ == "__main__":
    with open('MatingPool.pickle','rb') as p_file:
        MatingPool = pickle.load(p_file)
    
    index_min = MatingPool[1].index(min(MatingPool[1]))

    morphology = 'Detailed Morphology/20160119_sham5.CNG.swc'
    f_min = MatingPool[0][index_min]
    run_cost_simulation(f_min, morphology, plotting=True)
