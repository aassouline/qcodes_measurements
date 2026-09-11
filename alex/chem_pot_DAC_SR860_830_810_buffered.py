# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:46:50 2026
Merged with buffered/calibrated lock-in acquisition on 2026.

@author: manip_F112

###############################################################################
#                    WHAT WAS MERGED, AND WHY (read this first)
###############################################################################
This file is chem_pot_DAC_SR860_830_810.py (the chemical-potential tracking
sweep: DAC + SR860 + SR830 + SR810) with the buffer-capture / capture-offset
calibration machinery from the FET_gate_calibration script grafted onto it.

Design decisions made while merging (please check these against what you
actually want before running on the real setup):

1. ALL of chem_pot's instruments are kept and used exactly as before: dac
   (DAC20_qcodes), lockin_A (SR830), lockin_B (SR860), lockin_C (SR810).
   Nothing from the buffer script's own instrument set (SR865A, ITest,
   YokogawaGS200) is introduced -- the buffer/capture technique is applied
   to lockin_B (SR860), which is the instrument physically present in
   chem_pot's setup, using chem_pot's own DAC (dac.ChangeVout) as the
   fast-axis step function for calibration instead of gs200_set_fast.

2. You said the peak-maximum fit will now run on the SR860 (lockin_B),
   because you're rewiring the manip so lockin_B measures the peak instead
   of lockin_C (SR810). Concretely: the polyfit-based re-centering algorithm
   (np.polyfit -> argmax -> shift new_vec_vt) that used to run on
   current_x built from lockin_C.snap() now runs on X_binned, the
   buffer-captured and offset-corrected X trace from lockin_B. lockin_A
   (SR830, Gxy) and lockin_C (SR810) keep being read and logged exactly as
   before via .snap() each point (per your "use all of chem_pot's
   instruments") -- they're just no longer the source for the fit. If you
   actually want lockin_C dropped, or want it doing something else, that's
   a one-line change (see the second per-line loop below).

3. lockin_A and lockin_C don't have a hardware capture buffer, and they're
   read over serial (ASRL) -- querying them at every fast-axis point would
   reintroduce exactly the per-point round-trip cost that moving lockin_B
   to the buffer was meant to remove, and would slow the fast axis back
   down to serial-query speed. So they are now read ONCE per line, right
   before the fast-axis loop starts (after the twait settle, same point in
   the sequence where they used to be read per-point), and that single
   (X, Y) pair is copied into every point of that line's row in the
   dataset. The fast-axis loop itself now only does dac.ChangeVout ->
   sleep(tsl+trc) -> record timestamp, with no instrument query in it at
   all -- lockin_B's buffer captures continuously through the whole line
   and is read out + binned once per line, after the point loop. Because
   lockin_B's binned data is only known *after* the point loop,
   datasaver.add_result() is called in a second loop over the same fast
   axis, after buffer readout/binning (same pattern as in the calibration
   script).

4. ASSUMPTION TO VERIFY -- SR860 buffer API: the calibration script's buffer
   calls (lockin_B.buffer.capture_config/.capture_rate/.start_capture/
   .stop_capture/.count_capture_bytes/.bytes_per_sample/.get_capture_data/
   .set_capture_length_to_fit_samples/.capture_length_in_kb) were written
   against a qcodes SR865A instance. SR860 and SR865/865A share the same
   CAPTURECFG/CAPTURERATE/CAPTURESTART capture command set in the Stanford
   Research programming manual, and qcodes' SR860 driver is expected to
   expose the same `.buffer` submodule -- but I have not been able to
   import qcodes in this environment to confirm it against your installed
   version. There's a hard check right after instrument setup below that
   will fail loudly and immediately (before touching any gate) if
   `lockin_B.buffer` isn't there, rather than failing confusingly mid-sweep.
   Also double check the 4096 kB capture-memory limit mentioned in the
   comments below -- that number was empirically observed on the SR865A in
   the calibration script; confirm it (or the real limit) for your SR860
   before pushing SAMPLES_PER_TCLOCKIN / TCLOCKIN too far.

5. TCLOCKIN (lock-in time constant) is READ from lockin_B at runtime
   (lockin_B.time_constant()) rather than set by this script, since
   chem_pot never configures amplitude/time_constant/sensitivity/frequency
   for lockin_B itself (unlike the calibration script, which owned its
   lock-in's settings). Whatever time constant you have dialed in on the
   real SR860 is what the buffer capture rate and settle times are derived
   from -- so set it correctly on the instrument (front panel or otherwise)
   BEFORE running this script.

6. Everything else (gate scaling by 52428.8, StartSweep/ChangeVout calls,
   database/experiment setup, registered parameter names, the
   Chemical_potential bookkeeping, the closing "gate voltage not at zero"
   warning) is untouched from chem_pot.
"""

import numpy as np
import sys
import time
import datetime
import os
import ctypes
import atexit
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import qcodes as qc
from qcodes.logger.logger import start_all_logging
from qcodes.dataset.plotting import plot_dataset, plot_by_id
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter

############### Virtual drivers #################

sys.path.append('C:\\Users\\manip_F112\\Documents\\Experiment_16T\\qcodes_drivers')

from qcodes.instrument_drivers.stanford_research.SR860 import SR860

lockin_B = SR860('Lockin', 'USB0::0xB506::0x2000::006247::INSTR')   #

from qcodes.instrument_drivers.stanford_research import SR830

from qcodes.instrument_drivers.stanford_research.SR810 import SR810

lockin_A = SR830("lockin_a", "ASRL4::INSTR", terminator="\r")

lockin_C = SR810("lockin_c", "ASRL5::INSTR", terminator="\r")

if not hasattr(lockin_B, 'buffer'):
    raise RuntimeError(
        "lockin_B (SR860) has no '.buffer' submodule in this qcodes install. "
        "This script assumes SR860 exposes the same capture-buffer API as SR865A "
        "(capture_config, capture_rate, start_capture, stop_capture, "
        "count_capture_bytes, bytes_per_sample, get_capture_data, "
        "set_capture_length_to_fit_samples, capture_length_in_kb). "
        "Check your qcodes version's SR860 driver before running on the real setup."
    )

import DAC20_qcodes

import Myserial_qcodes

from DAC20_qcodes import Dac

from Myserial_qcodes import MySerialPort

#dac=Dac(port='COM7')

dac = Dac(port='COM3')

###############################################################################
#           WINDOWS TIMER RESOLUTION FIX
###############################################################################
# By default, Windows' system timer tick is ~15.6ms, and time.sleep(x) actually
# sleeps for AT LEAST x, rounded up to that tick. This requests 1ms timer
# resolution for the lifetime of this process -- carried over from the
# calibration script because the buffer <-> timestamp alignment below is only
# as good as the timing precision of the sleeps around it. No effect on
# non-Windows platforms (winmm.dll simply won't exist).
try:
	_winmm = ctypes.WinDLL('winmm')
	_winmm.timeBeginPeriod(1)
	atexit.register(_winmm.timeEndPeriod, 1)
	print('Windows timer resolution set to 1ms')
except (OSError, AttributeError):
	pass

#chan1 is top gate     chan2 is back gate    chan3 is SET common mode  chan4 is SET gate   chan5 is BLG
islowaxis = 3
ifastaxis = 4

#lockin_B.sine_outdc(0.0)
Vex_lockin = 2 / 1e4

h_planck = 6.6e-34
e_charge = 1.6e-19
Igain = 1e7

g0 = e_charge**2 / h_planck

dac.StartSweep(channel=1, target=int(5.0 * 52428.8), r=100000)
dac.StartSweep(channel=2, target=int(-8.0 * 52428.8), r=100000)

#0.1s per point + 0.24s
tsl = 0.1  #time sleep each point
trc = 0.0
twait = 10 * (tsl + trc) #time sleep each loop

Vmin = 0.0
Vmax = 0.1
Vnstep = 1376  #51
Vg = np.linspace(Vmin, Vmax, Vnstep)   # slow axis

offset_start = 2.9
Vdmin = 0.0 + offset_start
Vdmax = 0.005 + offset_start  # was good with SET gate
Vdnstep = 31
Vdc = np.linspace(Vdmin, Vdmax, Vdnstep)

minvec_vt = np.mean(Vdc)

dac.StartSweep(channel=islowaxis, target=int(Vg[0] * 52428.8), r=100000)   # 100 000 uV/s
dac.StartSweep(channel=ifastaxis, target=int(Vdc[0] * 52428.8), r=100000)
time.sleep(10.0)

###############################################################################
#                    BUFFER ACQUISITION PARAMETERS (lockin_B / SR860)
###############################################################################
# Grafted from the FET_gate_calibration script. TCLOCKIN is READ from the
# instrument rather than imposed by this script -- see point 5 in the header
# docstring. Everything below it (capture rate, settle time, buffer sizing)
# is derived from that value, exactly as in the calibration script, so it
# self-adjusts if you change the time constant on the SR860.

TCLOCKIN = lockin_B.time_constant()
print(f'Using lockin_B (SR860) time constant TCLOCKIN = {TCLOCKIN*1e3:.3f} ms '
	  f'(read from the instrument -- set it on the SR860 itself before running)')

# Capture rate is tied to TCLOCKIN, not to tsl: samples taken much faster than
# the filter's own correlation time (~TCLOCKIN) aren't independent, so packing
# in more of them buys little/no real noise reduction while still costing
# buffer memory and transfer time. SAMPLES_PER_TCLOCKIN sets the density where
# averaging is still genuinely useful; how many samples land in a given dwell
# (tsl - T_SKIP) then follows automatically.
SAMPLES_PER_TCLOCKIN = 5   # independent-ish samples per lock-in time constant

# CALIB_N_POINTS: number of calibration steps (spans Vdmin to Vdmax)
CALIB_N_POINTS = 5

# CALIB_SETTLE_TIME scales with TCLOCKIN (not a fixed absolute value) for the
# same reason tsl/T_SKIP/twait do: a fixed absolute settle_time combined with
# a capture rate that scales as 1/TCLOCKIN means the required buffer size
# (~ rate * duration) scales as 1/TCLOCKIN too -- shrink TCLOCKIN enough and
# the calibration buffer blows past the instrument's hardware memory limit
# (this happened on the SR865A: 14650 kB requested at a small TCLOCKIN --
# check the equivalent limit for your SR860). Scaling settle_time WITH
# TCLOCKIN makes rate*duration -- and therefore the required buffer size --
# independent of TCLOCKIN entirely, so this can't recur regardless of how
# TCLOCKIN is tuned later.
CALIB_SETTLE_TIME = 300 * TCLOCKIN   # s, wait after each calibration step -- must
                                       # comfortably exceed the real device's settling
                                       # time (check fitted tau values; widen the 300x
                                       # multiplier if fits look poor/inconsistent)

# Calibration's capture rate must resolve the TRANSIENT itself, not just fit
# "enough total samples" into the settle_time window -- tying rate to
# TCLOCKIN (like the main sweep's SAMPLES_PER_TCLOCKIN) keeps sampling dense
# regardless of settle_time.
CALIB_SAMPLES_PER_TCLOCKIN = 50   # denser than the main sweep's 5, since a clean
                                   # fit needs good resolution ON the transient itself

T_SKIP = 3 * TCLOCKIN   # settle time after each step before trusting lockin_B's output

###############################################################################
#                    LOCK-IN BUFFER SETUP (lockin_B / SR860)
###############################################################################

buf = lockin_B.buffer
buf.capture_config("X,Y")

target_rate = SAMPLES_PER_TCLOCKIN / TCLOCKIN
buf.capture_rate(target_rate)
print(f'Capture rate: {buf.capture_rate():.1f} Hz (requested {target_rate:.1f}, '
	  f'{SAMPLES_PER_TCLOCKIN} samples/TCLOCKIN)')
print(f'--> yields {buf.capture_rate() * (tsl - T_SKIP):.1f} samples per dwell '
	  f'(within the tsl - T_SKIP averaging window)')

_margin = 2.0
max_expected_samples = int(np.ceil(buf.capture_rate() * Vdnstep * tsl * _margin))
buf.set_capture_length_to_fit_samples(max_expected_samples)
print(f'Buffer sized for {max_expected_samples} samples ({buf.capture_length_in_kb()} kB)')


def calibrate_capture_offset(buf, step_fn, calib_voltages, settle_time, plot=True):
	"""
	Determine the fixed offset between a PC-issued command (timestamped with
	time.perf_counter()) and the corresponding sample index in the SR860
	capture buffer -- by stepping a real device via step_fn (here,
	chem_pot's fast-axis DAC channel) and fitting the resulting transient
	in X.

	step_fn        : callable(value) -> None. Whatever you pass here is what
	                  gets calibrated -- here, the chem_pot fast-axis DAC step.
	calib_voltages : values to step step_fn through, in order. Should span a
	                  large enough range for a clearly resolvable step in X.
	settle_time     : wait time after each step (s). Must be long enough for
	                  BOTH the device's own physical settling AND the
	                  lock-in filter to fully settle before the next step --
	                  if fits fail or offsets look inconsistent, increase
	                  this and check the per-step plots.
	plot            : if True, show one zoomed panel per fitted step,
	                  labeled with the gate voltage it corresponds to.

	Returns t_offset (s).
	"""
	# Append one throwaway trailing step (repeat the last value) so the real
	# last step always has a clean "next plateau" to fit against, instead of
	# bordering buf.stop_capture() directly.
	calib_voltages = list(calib_voltages) + [calib_voltages[-1]]

	# Calibration uses its own capture rate (based on settle_time), decoupled
	# from the main sweep's rate (based on tsl) -- otherwise a long tsl/low
	# main-sweep rate would starve the calibration windows of samples.
	original_capture_rate = buf.capture_rate()
	target_calib_rate = CALIB_SAMPLES_PER_TCLOCKIN / TCLOCKIN
	buf.capture_rate(target_calib_rate)
	print(f'Calibration capture rate: {buf.capture_rate():.1f} Hz '
		  f'(requested {target_calib_rate:.1f}, {CALIB_SAMPLES_PER_TCLOCKIN} samples/TCLOCKIN)')

	calib_duration = len(calib_voltages) * settle_time
	calib_samples_needed = int(np.ceil(buf.capture_rate() * calib_duration * 2.0))
	buf.set_capture_length_to_fit_samples(calib_samples_needed)

	# ---- run the calibration steps, timestamping each command ----
	buf.start_capture("CONT", "IMM")
	t_start = time.perf_counter()   # PC-side "zero" for buffer time

	step_timestamps = np.zeros(len(calib_voltages))
	for k, v in enumerate(calib_voltages):
		step_fn(v)
		step_timestamps[k] = time.perf_counter()
		time.sleep(settle_time)

	t_end = time.perf_counter()
	buf.stop_capture()

	# ---- pull the whole calibration sequence out of the buffer in one go ----
	actual_bytes = buf.count_capture_bytes()
	actual_samples = actual_bytes // (2 * buf.bytes_per_sample)   # 2 channels: X, Y
	X_buf = buf.get_capture_data(actual_samples)["X"]
	sample_times = t_start + np.arange(len(X_buf)) / buf.capture_rate()

	def step_model(t, t0, tau, A, B):
		return np.where(t < t0, A, A + B * (1 - np.exp(-(t - t0) / tau)))

	offsets, taus, fit_info = [], [], []

	for k in range(1, len(calib_voltages) - 1):   # skip k=0 and the final (dummy-bordering) step
		t_cmd = step_timestamps[k]
		t_lo = (step_timestamps[k - 1] + step_timestamps[k]) / 2
		t_hi = (step_timestamps[k] + step_timestamps[k + 1]) / 2
		mask = (sample_times >= t_lo) & (sample_times < t_hi)
		t_win, y_win = sample_times[mask], X_buf[mask]

		popt = None
		if len(t_win) >= 10:
			p0 = [t_cmd, max(TCLOCKIN, (t_hi - t_lo) / 10), y_win[0], y_win[-1] - y_win[0]]
			bounds = ([t_lo, 1e-6, -np.inf, -np.inf], [t_hi, (t_hi - t_lo), np.inf, np.inf])
			try:
				popt, _ = curve_fit(step_model, t_win, y_win, p0=p0, bounds=bounds, maxfev=10000)
				offsets.append(popt[0] - t_cmd)
				taus.append(popt[1])
			except RuntimeError:
				popt = None

		fit_info.append(dict(k=k, v=calib_voltages[k], t_cmd=t_cmd, t_win=t_win, y_win=y_win,
							  popt=popt, success=popt is not None))

	if not offsets:
		raise RuntimeError("Capture offset calibration failed: no step could be fit")

	offsets = np.array(offsets)
	taus = np.array(taus)
	tau_ok = taus < 3 * np.median(taus)
	offset_ok = np.abs(offsets - np.median(offsets)) < 2 * np.std(offsets) + 1e-9
	good = tau_ok & offset_ok
	if (~tau_ok).any():
		print(f'Excluding {(~tau_ok).sum()} step(s) with anomalously large tau: '
			  f'{np.round(taus[~tau_ok]*1e3, 2)} ms')
	# Use the LEAST NEGATIVE (max) of the good offsets, not the mean.
	# corrected_timestamps = timestamps + t_offset, and the trusted window
	# starts at corrected_timestamps + T_SKIP -- so a more negative t_offset
	# pushes that window EARLIER (risking unsettled/stale data), while a
	# less negative one pushes it LATER (safe -- just uses a bit less of the
	# averaging window). Averaging would split the difference and risk
	# under-correcting for whichever real step needed the least-negative
	# value; taking the max guards against that for every point.
	t_offset = offsets[good].max()

	print(f'Calibration step offsets (ms): {np.round(offsets*1e3, 2)}')
	print(f'Fitted tau (ms):                {np.round(taus*1e3, 2)}')
	print(f'--> using t_offset = {t_offset*1e3:.2f} ms (from {good.sum()}/{len(offsets)} steps)')

	if plot:
		n_fits = len(fit_info)
		fig, axes = plt.subplots(n_fits, 1, figsize=(7, 2.5 * n_fits), squeeze=False)
		good_iter = iter(good)
		for ax, info in zip(axes[:, 0], fit_info):
			t_rel = info['t_win'] - info['t_cmd']
			ax.plot(t_rel * 1e3, info['y_win'], '.-', ms=4, lw=0.5, color='tab:gray', label='data')
			if info['success']:
				t0, tau, A, B = info['popt']
				t_fit = np.linspace(info['t_win'][0], info['t_win'][-1], 300)
				y_fit = step_model(t_fit, *info['popt'])
				ax.plot((t_fit - info['t_cmd']) * 1e3, y_fit, '-', color='tab:red', label='fit')
				ax.axvline(0, color='tab:blue', ls='--', lw=1, label='t_cmd')
				ax.axvline((t0 - info['t_cmd']) * 1e3, color='tab:red', ls='--', lw=1, label='fitted t0')
				status = 'used' if next(good_iter) else 'outlier (excluded)'
				ax.set_title(f"step {info['k']}  (gate = {info['v']:.4f} V): "
							  f"offset = {(t0 - info['t_cmd'])*1e3:.2f} ms  [{status}]",
							  fontsize=9)
				t0_rel_ms = (t0 - info['t_cmd']) * 1e3
				zoom_half_width_ms = max(6 * tau, 5 * TCLOCKIN) * 1e3
				xlo, xhi = t0_rel_ms - 0.3 * zoom_half_width_ms, t0_rel_ms + zoom_half_width_ms
				ax.set_xlim(xlo, xhi)
				n_points_in_view = np.sum((t_rel * 1e3 >= xlo) & (t_rel * 1e3 <= xhi))
				print(f'  step {info["k"]}: {len(t_rel)} points in full fit window, '
					  f'{n_points_in_view} points in zoomed view')
			else:
				ax.set_title(f"step {info['k']}  (gate = {info['v']:.4f} V): fit failed",
							  fontsize=9, color='tab:red')
				fallback_half_width_ms = 50 * TCLOCKIN * 1e3
				ax.set_xlim(-0.3 * fallback_half_width_ms, fallback_half_width_ms)
			ax.set_xlabel('time relative to t_cmd (ms)')
			ax.set_ylabel('X (V)')
			ax.legend(fontsize=7, loc='best')
		fig.suptitle(f't_offset = {t_offset*1e3:.2f} ms', fontsize=11)
		fig.tight_layout()
		plt.show()

	buf.capture_rate(original_capture_rate)   # restore the main sweep's rate

	return t_offset


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
user = 'AA'
sample_name = 'NW6'
date = datetime.datetime.today().strftime('%Y_%m_%d')
description = sample_name
database_name = date + "_" + user + "_" + description
print(database_name)

exp_name = 'NW6_CD1'
sample_name = 'NW6'

###############################################################################
#                           DATA FOLDER CREATION
###############################################################################

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(r'C:\\Users\\manip_F112\\Documents\\Experiment_16T\\Data\\NW6_CD1')

try:
	os.mkdir(data_dir)
except FileExistsError:
	pass

data_dir = data_dir + '\\' + description

try:
	os.mkdir(data_dir)
except FileExistsError:
	pass

###############################################################################
#                       CREATE OR INITIALIZE DATABASE
################################################################################

qc.initialise_or_create_database_at(data_dir + '\\' + database_name + '.db')
qc.config.core.db_location


exp = qc.load_or_create_experiment(experiment_name=exp_name,
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

###############################################################################
#          CAPTURE-OFFSET CALIBRATION (lockin_B / SR860, via the real fast-axis gate)
###############################################################################

calib_voltages = np.linspace(Vdmin, Vdmax, CALIB_N_POINTS)

step_fn = lambda v: dac.ChangeVout(channel=ifastaxis, vb=int(v * 52428.8))

t_offset = calibrate_capture_offset(buf, step_fn, calib_voltages,
									 settle_time=CALIB_SETTLE_TIME, plot=True)

if abs(t_offset) > tsl:
	print(f'WARNING: |t_offset| ({abs(t_offset)*1e3:.2f} ms) exceeds tsl '
		  f'({tsl*1e3:.2f} ms). Increase tsl before trusting the real sweep.')

# calibrate_capture_offset resized the buffer for its own calibration windows
# and only restores the capture RATE, not the capture LENGTH -- re-size back
# to what the main sweep needs before starting the measurement.
buf.set_capture_length_to_fit_samples(max_expected_samples)
dac.ChangeVout(channel=ifastaxis, vb=int(Vdc[0] * 52428.8))

############################## Measurement ##################################

time_start = time.time()

parameter_snap = {}

print(f'database name : {database_name}')


with meas.run() as datasaver:

	id = datasaver.dataset.run_id

	qc.load_by_run_spec(captured_run_id=id).add_metadata('parameter_snap',
							 json.dumps(parameter_snap))

	time0 = time.time()
	current_time = time.time() - time0
	chemPot = np.array([])
	new_vec_vt = Vdc

	for i, valg in enumerate(Vg):

		dac.StartSweep(channel=islowaxis, target=int(valg * 52428.8), r=100000)
		dac.StartSweep(channel=ifastaxis, target=int(new_vec_vt[0] * 52428.8), r=100000)

		#time.sleep(np.abs(Vdc[-1] - Vdc[0]) / 0.1)  # necessary??

		if i != 0 and i % 5 == 0:
			print('measurement progress : ', int(i / len(Vg) * 100), ' % and remaining time : ',
				  int(((current_time * 100 / (i / len(Vg) * 100)) - current_time) / 60), 'min')

		if i == 0:
			time.sleep(5.0)

		time.sleep(twait)

		n_pts = len(new_vec_vt)

		# ---- lockin_A / lockin_C: read ONCE per line, not per fast-axis point ----
		# They're serial (ASRL) instruments with no capture buffer -- querying
		# them at every point would re-add the per-point round-trip latency that
		# putting lockin_B on the buffer was meant to remove. Their value is
		# treated as constant across the whole fast-axis line and copied into
		# every point's row below.
		V_X2_A_line, V_Y2_A_line = lockin_A.snap("X", "Y")
		V_X2_C_line, V_Y2_C_line = lockin_C.snap("X", "Y")

		# ---- lockin_B (SR860): continuous buffer capture across the whole line ----
		buf.start_capture("CONT", "IMM")
		t_start = time.perf_counter()

		perf_timestamps = np.zeros(n_pts)
		wall_timestamps = np.zeros(n_pts)

		for j, valdc in enumerate(new_vec_vt):

			dac.ChangeVout(channel=ifastaxis, vb=int(valdc * 52428.8))
			perf_timestamps[j] = time.perf_counter()   # t_cmd, for lockin_B buffer alignment
			#yoko2.ramp_voltage(valdc, 0.2, 0.01)

			time.sleep(tsl + trc)

			wall_timestamps[j] = time.time()

		t_end = time.perf_counter()
		buf.stop_capture()

		# ---- pull lockin_B's whole line out of the buffer and bin it per point ----
		actual_bytes = buf.count_capture_bytes()
		actual_samples = actual_bytes // (2 * buf.bytes_per_sample)   # 2 channels: X, Y
		data = buf.get_capture_data(actual_samples)
		X_buf, Y_buf = data["X"], data["Y"]
		sample_times = t_start + np.arange(len(X_buf)) / buf.capture_rate()

		corrected_timestamps = perf_timestamps + t_offset

		X_binned = np.full(n_pts, np.nan)
		Y_binned = np.full(n_pts, np.nan)

		for j in range(n_pts):
			t0 = corrected_timestamps[j] + T_SKIP
			t1 = corrected_timestamps[j + 1] if j + 1 < n_pts else t_end
			mask = (sample_times >= t0) & (sample_times < t1)
			if mask.any():
				X_binned[j] = X_buf[mask].mean()
				Y_binned[j] = Y_buf[mask].mean()
			else:
				idx = np.argmin(np.abs(sample_times - corrected_timestamps[j]))
				X_binned[j] = X_buf[idx]
				Y_binned[j] = Y_buf[idx]

		# ---- second pass: now that lockin_B's binned data exists, save every point ----
		for j, valdc in enumerate(new_vec_vt):

			V_X2_B, V_Y2_B = X_binned[j], Y_binned[j]
			# lockin_A / lockin_C were read once for the whole line (see above) --
			# same pair repeated on every row of this line.
			V_X2_A, V_Y2_A = V_X2_A_line, V_Y2_A_line
			V_X2_C, V_Y2_C = V_X2_C_line, V_Y2_C_line

			current = V_X2_A / 1e7
			current_xx = V_X2_B / 1e7

			Gxy = current / Vex_lockin / g0

			Gxx = current_xx / Vex_lockin / g0

			current_time = wall_timestamps[j] - time0 # elapsed time
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

		# ---- peak-maximum fit now runs on lockin_B's (SR860) binned X trace ----
		current_x = X_binned

		#change for adjusting measurement window around minima
		pfit = np.polyfit(new_vec_vt, current_x, 4, full=True)

		'''plt.close()
		plt.plot(new_vec_vt,current_x)
		plt.plot(new_vec_vt,np.polyval(pfit,new_vec_vt))
		plt.show(block=False)
		plt.pause(0.001)'''

		#Sminpa=np.array(np.where(np.polyval(pfit,new_vec_vt)==np.amin(np.polyval(pfit,new_vec_vt))))
		vtgar = np.arange(new_vec_vt[0], new_vec_vt[-1], 1e-6)  #new
		Sminpa = np.array(np.where(np.polyval(pfit[0], vtgar) == np.amax(np.polyval(pfit[0], vtgar))))   #new  change between amin or amax depending on minima or maxima
		ri = int(np.mean(Sminpa[0, :]))
		#minvec_vt=new_vec_vt[ri]
		minvec_vt = vtgar[ri]  #new
		chemPot = np.append(chemPot, minvec_vt)

		print("middle value of the gate sweep: {:.7f} ".format(minvec_vt), "residual: {} ".format(pfit[1]))
		dvt = Vdc[1] - Vdc[0]
		f_newvt = minvec_vt - len(Vdc) / 2 * dvt
		l_newvt = minvec_vt + len(Vdc) / 2 * dvt
		new_vec_vt = np.linspace(f_newvt, l_newvt, len(Vdc))

current_time = time.time()

print('time measurement:', current_time - time_start, 's')

print('WARNING : gate voltage are not at zero !!!!')
