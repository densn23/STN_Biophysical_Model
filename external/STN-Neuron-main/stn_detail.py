import sys, os
from neuron import h
import neuron as nrn
from neuron.units import ms, mV
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


path = os.getcwd()
nrn.load_mechanisms(path+'/sth')
h.load_file('stdrun.hoc')
h.load_file("stdlib.hoc")
h.load_file("import3d.hoc")


def insert_channels(sec):
    sec.insert('STh')
    sec.insert('Na')
    sec.insert('NaL')
    sec.insert('KDR')
    sec.insert('Kv31')
    sec.insert('Ih')
    sec.insert('Cacum')
    sec.insert('sKCa')
    sec.insert('CaT')
    sec.insert('HVA')
    sec.insert('extracellular')
    for i in range(1):
        sec.xraxial[i] = 1e9
        sec.xg[i] = 1e9
        sec.xc[i] = 0


def set_aCSF(i):
    if i == 3:
        # print("Setting in vitro parameters based on Beurrier et al (1999)")
        h.nai0_na_ion = 15
        h.nao0_na_ion = 150   
        h.ki0_k_ion = 140
        h.ko0_k_ion = 3.6   
        h.cai0_ca_ion = 1e-04
        h.cao0_ca_ion = 2.4    
#         h.cli0_cl_ion = 4
#         h.clo0_cl_ion = 135
    if i == 4:
        # print("Setting in vitro parameters based on Bevan & Wilson (1999)")
        h.nai0_na_ion = 15             # This is different in Miocinovic model
        h.nao0_na_ion = 128.5          # This is different in Miocinovic model
        h.ki0_k_ion = 140
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 1e-04
        h.cao0_ca_ion = 2.0
#         h.cli0_cl_ion = 4
#         h.clo0_cl_ion = 132.5
    if i == 0:
        # print("WARNING: Using NEURON defaults for in vitro parameters")
        h.nai0_na_ion = 10
        h.nao0_na_ion = 140
        h.ki0_k_ion = 54
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 5e-05
        h.cao0_ca_ion = 2
#         h.cli0_cl_ion = 0
#         h.clo0_cl_ion = 0


# Building Cell
class CreateSth():
    def __init__(self, params=None, morphology='Detailed Morphology/20160119_sham3.CNG.swc'):
        cell = h.Import3d_SWC_read()
        cell.input(morphology)
        i3d = h.Import3d_GUI(cell, 0)
        i3d.instantiate(self)
        set_aCSF(4)
        factor = 2
        for sec in self.dend:
            sec.L = sec.L * factor
            sec.nseg = 13
            sec.diam = sec.diam
            
        try:
            for sec in self.axon:
                h.delete_section(sec=sec)
        except:
            self.ad=0

        path_length = []
        for num, sec in enumerate(self.dend):
            dis = h.distance(self.soma[0](0.5), sec(1))
            path_length.append(dis)
                    
        self.max_path = max(path_length)

        self.params = params
        self.rhoa= 0.7e6
        self._initialize_constants()
        self.initseg = h.Section(name='initseg', cell=self)
        self._setup_morphology()
        self.sdi = self.soma + self.dend + [self.initseg]
        self.axon = self.node + self.MYSA + self.FLUT + self.STIN
        self.all = self.sdi + self.node + self.MYSA + self.FLUT + self.STIN
        self._setup_biophysics()
        
    
    def _initialize_constants(self):
        PI = np.pi
        fiberD=2
        paralength1=3  
        nodelength=1
        space_p1=0.002  
        space_p2=0.004
        space_i=0.004
        nodeD = 1.4
        axonD = 1.6
        paraD1 = 1.4
        paraD2 = 1.6
        self.mycm=0.1
        self.mygm = 0.001
        self.nl=30
        self.Rpn0=(self.rhoa*.01)/(PI*((((nodeD/2)+space_p1)**2)-((nodeD/2)**2)))
        self.Rpn1=(self.rhoa*.01)/(PI*((((paraD1/2)+space_p1)**2)-((paraD1/2)**2)))
        self.Rpn2=(self.rhoa*.01)/(PI*((((paraD2/2)+space_p2)**2)-((paraD2/2)**2)))
        self.Rpx=(self.rhoa*.01)/(PI*((((axonD/2)+space_i)**2)-((axonD/2)**2)))


    def _setup_morphology(self):
        self.initseg.diam = 1.8904976874853334
        self.initseg.L = 21.7413353424173
        self.node = []
        self.MYSA = []
        self.FLUT = []
        self.STIN = []
        
                    
        # Building Axon
        axonnodes=10
        paranodes1=(axonnodes-1)*2
        paranodes2=(axonnodes-1)*2
        axoninter=(axonnodes-1)*3
        
        for i in range(axonnodes):
            name_str = 'node{}'.format(i)
            self.node.append(h.Section(name=name_str, cell=self))
            self.node[i].diam = 1.4
            self.node[i].L = 1
            self.node[i].nseg = 1
        for i in range(paranodes1):
            if i < 6:
                name_str = 'MYSA{}'.format(i)
                self.MYSA.append(h.Section(name=name_str, cell=self))
                self.MYSA[i].diam = 1.4
                self.MYSA[i].L = 1.5
                self.MYSA[i].nseg = 1
            else:
                name_str = 'MYSA{}'.format(i)
                self.MYSA.append(h.Section(name=name_str, cell=self))
                self.MYSA[i].diam = 1.4
                self.MYSA[i].L = 3
                self.MYSA[i].nseg = 1
        for i in range(paranodes2):
            if i < 6:
                name_str = 'FLUT{}'.format(i)
                self.FLUT.append(h.Section(name=name_str, cell=self))
                self.FLUT[i].diam = 1.6
                self.FLUT[i].L = 5
                self.FLUT[i].nseg = 1
            else:
                name_str = 'FLUT{}'.format(i)
                self.FLUT.append(h.Section(name=name_str, cell=self))
                self.FLUT[i].diam = 1.6
                self.FLUT[i].L = 10
                self.FLUT[i].nseg = 1
        for i in range(axoninter):
            if i < 9:
                name_str = 'STIN{}'.format(i)
                self.STIN.append(h.Section(name=name_str, cell=self))
                self.STIN[i].diam = 1.6
                self.STIN[i].L = 29
                self.STIN[i].nseg = 1
            else:
                name_str = 'STIN{}'.format(i)
                self.STIN.append(h.Section(name=name_str, cell=self))
                self.STIN[i].diam = 1.6
                self.STIN[i].L = 58
                self.STIN[i].nseg = 1
        
        # Connecting axon and soma
        self.initseg.connect(self.soma[0])
        self.node[0].connect(self.initseg)
        for i in range(axonnodes-1):
            self.MYSA[2*i].connect(self.node[i])
            self.FLUT[2*i].connect(self.MYSA[2*i])
            self.STIN[3*i].connect(self.FLUT[2*i])
            self.STIN[3*i+1].connect(self.STIN[3*i])
            self.STIN[3*i+2].connect(self.STIN[3*i+1])
            self.FLUT[2*i+1].connect(self.STIN[3*i+2])
            self.MYSA[2*i+1].connect(self.FLUT[2*i+1])
            self.node[i+1].connect(self.MYSA[2*i+1])

    def _setup_biophysics(self):
        for sec in self.sdi:
            sec.Ra = self.params[18]    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
            insert_channels(sec)
        for sec in self.axon:
            sec.Ra = self.rhoa / 10000
            sec.cm = 2
        for sec in self.soma:
            for seg in sec:
                seg.NaL.gna = self.params[7]
                seg.Na.gna = self.params[8]
                seg.HVA.gcaL = self.params[0]
                seg.HVA.gcaN = self.params[1]
                seg.CaT.gcaT = self.params[2]
                seg.Ih.gk = self.params[3]
                seg.KDR.gk = self.params[4]
                seg.Kv31.gk = self.params[5] 
                seg.sKCa.gk = self.params[6] 
                seg.STh.gpas = self.params[17]
        for seg in self.initseg:
            seg.NaL.gna = self.params[7]
            seg.Na.gna = self.params[8] * self.params[19]
            seg.HVA.gcaL = self.params[0]
            seg.HVA.gcaN = self.params[1]
            seg.CaT.gcaT = self.params[2]
            seg.Ih.gk = self.params[3]
            seg.KDR.gk = self.params[4]
            seg.Kv31.gk = self.params[5]
            seg.sKCa.gk = self.params[6]
            seg.STh.gpas = self.params[17]
        proximal = self.max_path/2
        dend_i = 0
        for sec in self.dend:
            for seg in sec:
                if h.distance(self.soma[0](0.5), seg) < proximal:
                    seg.NaL.gna = self.params[15] * self.params[7]
                    seg.Na.gna = self.params[16] * self.params[8]
                    seg.HVA.gcaL = self.params[9] * self.params[0]
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = self.params[11]
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = self.params[12] * self.params[4]
                    seg.Kv31.gk = self.params[13] * self.params[5]
                    seg.sKCa.gk = self.params[14] * self.params[6]
                    seg.STh.gpas = self.params[17]
                else:
                    seg.NaL.gna = 0
                    seg.Na.gna = 0
                    seg.HVA.gcaL = 0
                    seg.HVA.gcaN = self.params[10]
                    seg.CaT.gcaT = 0
                    seg.Ih.gk = self.params[3]
                    seg.KDR.gk = 0
                    seg.Kv31.gk = 0
                    seg.sKCa.gk = self.params[14] * self.params[6] 
                    seg.STh.gpas = self.params[17]
                dend_i += 1
        
        for count, sec in enumerate(self.node):
            sec.insert('extracellular')
            for i in range(1):
                    sec.xraxial[i]=self.Rpn0
                    sec.xg[i] = 1e10
                    sec.xc[i]=0
            if count == 9:
                sec.insert('pas')
                for seg in sec:
                    seg.pas.g = 0.0001
                    seg.pas.e = -65
            else:
                sec.insert('axnode75')
                for seg in sec:
                    seg.axnode75.gnabar = 2
                    seg.axnode75.gnapbar = 0.05
                    seg.axnode75.gl = 0.005
                    seg.axnode75.ek = -85
                    seg.axnode75.ena = 55
                    seg.axnode75.el = -60
        
        for sec in self.MYSA:
            sec.insert('extracellular')
            for i in range(1):
                sec.xraxial[i]=self.Rpn1
                sec.xg[i] = self.mygm/(self.nl*2)
                sec.xc[i]= self.mycm/(self.nl*2)
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 0.0001
                seg.pas.e = -65

        for sec in self.FLUT:
            sec.insert('extracellular')
            for i in range(1):
                sec.xraxial[i]=self.Rpn2
                sec.xg[i] = self.mygm/(self.nl*2)
                sec.xc[i]= self.mycm/(self.nl*2)
            sec.insert('parak75')
            sec.insert('pas')
            for seg in sec:
                seg.parak75.gkbar = 0.02
                seg.parak75.ek = -85
                seg.pas.g = 0.0001
                seg.pas.e = -60
        
        for sec in self.STIN:
            sec.insert('extracellular')
            for i in range(1):
                sec.xraxial[i]=self.Rpx
                sec.xg[i] = self.mygm/(self.nl*2)
                sec.xc[i]= self.mycm/(self.nl*2)
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = 0.0001
                seg.pas.e = -65
            
        for sec in self.all:
            h.ion_style("na_ion",1,2,1,0,1, sec=sec)
            h.ion_style("k_ion",1,2,1,0,1, sec=sec)
            h.ion_style("ca_ion",3,2,1,1,1, sec=sec)

    def __repr__(self):
        return 'sth'


