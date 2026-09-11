# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:46:50 2026

@author: manip_F112
"""

#import requests
import urllib.request
import numpy as np
import requests
import sys

import qcodes as qc
from qcodes.logger.logger import start_all_logging
from qcodes.dataset.plotting import plot_dataset,plot_by_id
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
#from qcodes.loops import Loop
#from qcodes.plots.pyqtgraph import QtPlot
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from time import sleep
import json
import os

############### Virtual drivers #################

sys.path.append('C:\\Users\\manip_F112\\Documents\\Experiment_16T\\qcodes_drivers')

from qcodes.instrument_drivers.stanford_research.SR860 import SR860

lockin_B = SR860('Lockin', 'USB0::0xB506::0x2000::006247::INSTR')   #

from qcodes.instrument_drivers.stanford_research import SR830

from qcodes.instrument_drivers.stanford_research.SR810 import SR810

lockin_A = SR830("lockin_a", "ASRL4::INSTR",terminator="\r")

lockin_C = SR810("lockin_c", "ASRL5::INSTR",terminator="\r")

#from qcodes.instrument_drivers.yokogawa.GS200 import GS200

import DAC20_qcodes

import Myserial_qcodes

from DAC20_qcodes import Dac

from Myserial_qcodes import MySerialPort

#dac=Dac(port='COM7')

dac=Dac(port='COM3')

#chan1 is top gate     chan2 is back gate    chan3 is SET common mode  chan4 is SET gate   chan5 is BLG
islowaxis=3
ifastaxis=4

#lockin_B.sine_outdc(0.0)
Vex_lockin=2/1e4

h_planck=6.6e-34
e_charge=1.6e-19
Igain=1e7

g0=e_charge**2/h_planck

dac.StartSweep(channel=1, target=int(5.0 * 52428.8), r=100000)
dac.StartSweep(channel=2, target=int(-8.0 * 52428.8), r=100000)

#0.1s per point + 0.24s
tsl=0.1  #time sleep each point
trc=0.0
twait=10*(tsl+trc) #time sleep each loop

Vmin=0.0
Vmax=0.1
Vnstep=1376  #51
Vg=np.linspace(Vmin,Vmax,Vnstep)   # slow axis

offset_start = 2.9
Vdmin = 0.0 + offset_start
Vdmax = 0.005 + offset_start  # was good with SET gate
Vdnstep = 31
Vdc = np.linspace(Vdmin, Vdmax, Vdnstep)

minvec_vt=np.mean(Vdc)

dac.StartSweep(channel=islowaxis, target=int(Vg[0] * 52428.8), r=100000)   # 100 000 uV/s
dac.StartSweep(channel=ifastaxis, target=int(Vdc[0] * 52428.8), r=100000)
time.sleep(10.0)

###############################################################################
#
#                      INITIALIZE QCODES EXPERIMENT
#
###############################################################################

start_all_logging()

# Create a station
station = qc.Station()
# station.add_component(lockin_C)
station.add_component(lockin_B)
# station.add_component(lockin)
# station.add_component(lockin_thermometre)

station.snapshot()
station.components

# Experiment details
user='AA'
sample_name='NW6'
date=datetime.datetime.today().strftime('%Y_%m_%d')
description=sample_name
database_name = date+"_"+user+"_"+description
print(database_name)

exp_name     = 'NW6_CD1'
sample_name  = 'NW6'

###############################################################################
#                           DATA FOLDER CREATION
###############################################################################

script_dir=os.path.dirname(__file__)
data_dir=os.path.join(r'C:\\Users\\manip_F112\\Documents\\Experiment_16T\\Data\\NW6_CD1')

try :
	os.mkdir(data_dir)
except FileExistsError:
	pass

data_dir=data_dir +'\\'+description

try :
	os.mkdir(data_dir)
except FileExistsError:
	pass

###############################################################################
#                       CREATE OR INITIALIZE DATABASE
################################################################################

qc.initialise_or_create_database_at(data_dir+'\\'+database_name+'.db')
qc.config.core.db_location


exp=qc.load_or_create_experiment(experiment_name=exp_name,
									  sample_name=sample_name)

meas = qc.Measurement(exp=exp, station=station)
#meas.register_parameter(VNA.channels.S21.power)

meas.register_custom_parameter('Vdc', unit='V')
meas.register_custom_parameter('Vg', unit='V')

meas.register_custom_parameter('Chemical_potential', setpoints=['Vg'])

meas.register_custom_parameter('Time', setpoints=['Vdc', 'Vg'])

meas.register_custom_parameter('lockin_A_X', setpoints=['Vdc', 'Vg'])
meas.register_custom_parameter('lockin_A_Y', setpoints=['Vdc', 'Vg'])

meas.register_custom_parameter('lockin_B_X', setpoints=['Vdc', 'Vg'])
meas.register_custom_parameter('lockin_B_Y', setpoints=['Vdc', 'Vg'])

meas.register_custom_parameter('lockin_C_X', setpoints=['Vdc', 'Vg'])
meas.register_custom_parameter('lockin_C_Y', setpoints=['Vdc', 'Vg'])

meas.register_custom_parameter('Gxy_X', setpoints=['Vdc', 'Vg'])
meas.register_custom_parameter('Gxx_X', setpoints=['Vdc', 'Vg'])

############################## Measurement ##################################

time_start = time.time()

parameter_snap={}

print(f'database name : {database_name}')


with meas.run() as datasaver:

	id=datasaver.dataset.run_id

	qc.load_by_run_spec( captured_run_id=id).add_metadata('parameter_snap',
						 json.dumps(parameter_snap))

	time0 = time.time()
	current_time = time.time() - time0
	chemPot=np.array([])
	new_vec_vt=Vdc

	for i, valg in enumerate(Vg) :

		dac.StartSweep(channel=islowaxis, target=int(valg * 52428.8), r=100000)
		dac.StartSweep(channel=ifastaxis, target=int(new_vec_vt[0] * 52428.8), r=100000)

		#time.sleep(np.abs(Vdc[-1] - Vdc[0]) / 0.1)  # necessary??

		if i != 0 and i % 5 == 0:
			print('measurement progress : ', int(i / len(Vg) * 100), ' % and remaining time : ',
				  int(((current_time * 100 / (i / len(Vg) * 100)) - current_time) / 60), 'min')

		if i==0:
			time.sleep(5.0)

		time.sleep(twait)

		current_x=np.array([])

		for j, valdc in enumerate(new_vec_vt) :

			dac.ChangeVout(channel=ifastaxis, vb=int(valdc * 52428.8))
			#yoko2.ramp_voltage(valdc, 0.2, 0.01)

			time.sleep(tsl+trc)

			V_X2_B, V_Y2_B = lockin_B.get_values("X","Y")  #
			V_X2_A, V_Y2_A = lockin_A.snap("X","Y")  #
			V_X2_C, V_Y2_C = lockin_C.snap("X","Y")  #

			current = V_X2_A / 1e7
			current_xx = V_X2_B / 1e7

			Gxy = current / Vex_lockin / g0

			Gxx = current_xx / Vex_lockin / g0

			current_x=np.append(current_x, V_X2_C)

			current_time = time.time() - time0 # elapsed time
			datasaver.add_result(('lockin_B_X', V_X2_B),
								 ('lockin_B_Y', V_Y2_B),
								 ('lockin_A_X', V_X2_A),
								 ('lockin_A_Y', V_Y2_A),
								 ('lockin_C_X', V_X2_C),
								 ('lockin_C_Y', V_Y2_C),
								 ('Gxy_X', Gxy),
								 ('Gxx_X', Gxx),
								 ('Vdc', valdc),
								 ('Vg', valg),
								 ('Time', current_time),
								 ('Chemical_potential', minvec_vt))

		#change for adjusting measurement window around minima
		pfit=np.polyfit(new_vec_vt, current_x, 4, full=True)

		'''plt.close()
		plt.plot(new_vec_vt,current_x)
		plt.plot(new_vec_vt,np.polyval(pfit,new_vec_vt))
		plt.show(block=False)
		plt.pause(0.001)'''

		#Sminpa=np.array(np.where(np.polyval(pfit,new_vec_vt)==np.amin(np.polyval(pfit,new_vec_vt))))
		vtgar=np.arange(new_vec_vt[0],new_vec_vt[-1],1e-6)  #new
		Sminpa=np.array(np.where(np.polyval(pfit[0],vtgar)==np.amax(np.polyval(pfit[0],vtgar))))   #new  change between amin or amax depending on minima or maxima
		ri=int(np.mean(Sminpa[0,:]))
		#minvec_vt=new_vec_vt[ri]
		minvec_vt=vtgar[ri]  #new
		chemPot=np.append(chemPot, minvec_vt)

		print("middle value of the gate sweep: {:.7f} ".format(minvec_vt), "residual: {} ".format(pfit[1]))
		dvt=Vdc[1]-Vdc[0]
		f_newvt=minvec_vt-len(Vdc)/2*dvt
		l_newvt=minvec_vt+len(Vdc)/2*dvt
		new_vec_vt= np.linspace(f_newvt, l_newvt, len(Vdc) )

current_time = time.time()

print ('time measurement:', current_time-time_start, 's')

print ('WARNING : gate voltage are not at zero !!!!')
