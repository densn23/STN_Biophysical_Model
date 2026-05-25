from stn_neuron import CreateSth
import matplotlib.pyplot as plt
import time
import pandas as pd
from statistics import mean
import efel
from scipy.signal import find_peaks
import math
import pickle
import numpy as np
import neuron as nrn
from neuron import h
from neuron.units import ms, mV


def create_updated_cell(f):
    stn_cell = CreateSth(params=f)
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
        # score = (v_HP - (v_rest-20)) / 20 * weight
        score_1 = (v_HP - (-80)) / 20 * weight/2
        if score_1 < 0:
            score_1 = 0
        score_shp = 0
        v_ht = v_HP+(v[int(575/dt)]-v_HP)/2+1
        if v[int(350/dt)]>v_ht:
            score_shp = (v[int(350/dt)]-v_ht) / 3 * weight/4
        else:
            score_shp = 0
        e_slp = v[23000] - v_HP
        if e_slp >= 0:
            score_slp = cal_score(e_slp, [4, 10], weight/4, 'SLP')
        else:
            score_slp = weight/4
        score = score_1 + score_slp + score_shp
    else:
        v_HP = 0
        score = weight
    spikes_all = find_peaks(v[int(600/dt):], height=-20)
    if len(spikes_all[0]) == 0:
        score = score
    elif min(spikes_all[1]['peak_heights']) < 0:
        score = score + weight/4
    if score < 0:
        score = 0
    if score > weight:
        score = weight
    return score, v_HP


def check_hp_last_spikes(v, score, weight, dt):
    spikes_after = find_peaks(v[int(750/dt):], height=0)
    if len(spikes_after[0]) != 0:
        score = score
    else:
        score = score + weight
    return score


def cal_score(param, target, factor, category):
    if category == 'SP':
        if param < target[0]:
            score = (target[0] - param) / (target[0]/2) * factor
        elif param > target[1]:
            score = (param - target[1]) / (target[1]/2) * factor
        elif target[0] <= param <= target[1]:
            score = 0

    if category == 'SLP':
        if param < target[0]:
            score = (target[0] - param) / 4 * factor
        elif param > target[1]:
            score = (param - target[1]) / 4 * factor
        elif target[0] <= param <= target[1]:
            score = 0

    if category == 'FI1':
        if param < target[0]:
            score = (target[0] - param) / 25 * factor
        elif param > target[1]:
            score = (param - target[1]) / 25 * factor
        elif target[0] <= param <= target[1]:
            score = 0

    if category == 'FI2':
        if param < target[0]:
            score = (target[0] - param) / 40 * factor
        elif param > target[1]:
            score = (param - target[1]) / 40 * factor
        elif target[0] <= param <= target[1]:
            score = 0

    if category == 'FI3':
        if param < target[0]:
            score = (target[0] - param) / 5 * factor
        elif param > target[1]:
            score = (param - target[1]) / 5 * factor
        elif target[0] <= param <= target[1]:
            score = 0

    if category == 'SHW':
        score = (param - target) / target * factor

    if category == 'AHP':
        if param < target[0]:
            score = (target[0] - param) / 10 * factor
        elif param > target[1]:
            score = (param - target[1]) / 10 * factor
        elif target[0] <= param <= target[1]:
            score = 0

    if category == 'IR':
        if param < target[0]:
            score = (target[0] - param) / 100 * factor
        elif param > target[1]:
            score = (param - target[1]) / 100 * factor
        elif target[0] <= param <= target[1]:
            score = 0

    if category == 'Rest':
        if param <= target[0]:
            score = (target[0] - param) / 5 * factor
        elif param >= target[1]:
            score = (param - target[1]) / 5 * factor
        elif target[0] < param < target[1]:
            score = 0
        else:
            score = factor

    if category == 'AP Peak':
        if param <= target[0]:
            score = (target[0] - param) / 4 * factor
        elif param >= target[1]:
            score = (param - target[1]) / 4 * factor
        elif target[0] < param < target[1]:
            score = 0
        else:
            score = factor
    
    if score > factor:
        score = factor

    if score < 0:
        score = 0

    return score


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
    z.loc(0.5, sec = stn_cell.soma)
    z.compute(0)
    z_value = z.input(0.5, sec = stn_cell.soma)
    return z_value


def output_sptime_params(f_index, score):
    with open('temp_full/parallelresults{}.txt'.format(f_index),'w') as f:
        f.write(str(f_index)+'\n'+str(score)+'\n')


def run_cost_simulation(f_index, plotting=False):
    # st = time.time()
    f = f_index
    h.dt = 0.025
    h.celsius = 37
    
    # Check input resistance
    stn_cell = create_updated_cell(f)
    z_value = report_input_impedance(stn_cell, 37)
    score_ir = cal_score(z_value, [50, 250], 100, 'IR')
    if math.isnan(score_ir):
        score_ir = 50
    print(f'Input Resistence: {z_value:.2f} MΩ')

    # Check tonic spiking
    stn_cell = create_updated_cell(f)
    soma_v = h.Vector().record(stn_cell.soma(0.5)._ref_v)
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
    score_sp = cal_score(freq_sp, [10, 20], 100, 'SP')
    if math.isnan(score_sp):
        score_sp = 50

    # Save State
    svstate = h.SaveState()
    svstate.save()

    # Check AP shape
    # if freq_sp != 0:
    shw = cal_AP_width(t_0, v_0)
    score_shw = cal_score(shw, 1, 50, 'SHW')
    # print('SHW: {}'.format(shw))
    if math.isnan(score_shw):
        score_shw = 25
    v_AHP = check_AHP(v_0)
    score_AHP = cal_score(v_AHP, [-75, -60], 50, 'AHP')
    if math.isnan(score_AHP):
        score_AHP = 50
    v_rest = check_rest(v_0)
    score_rest = cal_score(v_rest, [-65, -55], 50, 'Rest')
    if math.isnan(score_rest):
        score_rest = 25
    v_peak = check_peak(v_0)
    score_peak = cal_score(v_peak, [10, 20], 50, 'AP Peak')
    if math.isnan(score_peak):
        score_peak = 50


    # Check hyperpolarization current injection
    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
    stim.delay = 1600
    stim.dur = 500
    stim.amp = -0.1
    h.continuerun(2500 * ms)
    v_3 = soma_v.to_python()
    t_3 = soma_t.to_python()
    soma_v.clear()
    soma_t.clear()
    score_hp, v_HP = cal_score_HP(v_3, 200, v_rest, h.dt)
    if math.isnan(score_hp):
        score_hp = 100

    # Check FI curve
    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
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
    score_fi1 = cal_score(freq_fi1, [65, 75], 100, 'FI1')
    if math.isnan(score_fi1):
        score_fi1 = 100

    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
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
    score_fi2 = cal_score(freq_fi2, [116, 126], 100, 'FI2')
    if math.isnan(score_fi2):
        score_fi2 = 100

    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
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
    score_fi3 = cal_score(freq_fi3, [26, 36], 100, 'FI3')
    if math.isnan(score_fi3):
        score_fi3 = 100

    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
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

    score_fi = score_fi1 + score_fi2 + score_fi3
    freq_fi = [freq_sp, freq_fi3, freq_fi1, freq_fi2, freq_fi4]

    # Check extreme HP condition
    if score_hp < 100:
        svstate.restore()
        stim = h.IClamp(stn_cell.soma(0.5))
        stim.delay = 1600
        stim.dur = 500
        stim.amp = -0.5
        h.continuerun(2800 * ms)
        v_check = soma_v.to_python()
        t_check = soma_t.to_python()
        soma_v.clear()
        soma_t.clear()
        score_hp = check_hp_last_spikes(v_check, score_hp, 100, h.dt)

    score_total = score_sp + score_shw  + score_AHP + score_peak + score_fi + score_hp + score_ir +score_rest

    if plotting:
        print(f'half spike width: {shw:.2f} ms')
        print(f'AHP: {v_AHP:.2f} mV')

        plt.figure()
        plt.plot(t_0, v_0)
        plt.title(f'Spontaneous Spiking {freq_sp:.2f}Hz')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')

        plt.figure()
        peaks = find_peaks(v_0, height=0)
        t_peak = peaks[0][1]
        v_single = v_0[t_peak-int(10/h.dt):t_peak+int(15/h.dt)]
        t_single = t_0[t_peak-int(10/h.dt):t_peak+int(15/h.dt)]
        plt.plot(t_single, v_single)
        plt.title('Single AP')
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

    return score_total


if __name__ == "__main__":
    with open('MatingPool.pickle','rb') as p_file:
        MatingPool = pickle.load(p_file)

    pool_mean = np.mean(MatingPool[1])
    print(f'Pool Mean = {pool_mean:.2f}'.format(pool_mean))
    
    index_min = MatingPool[1].index(min(MatingPool[1]))
    print(f'Score Min = {MatingPool[1][index_min]:.2f}')
    print(f'Index Min = {index_min}')

    f_min = MatingPool[0][index_min]
    score = run_cost_simulation(f_min, plotting=True)
    print(f'final score: {score:.2f}')
    
    
