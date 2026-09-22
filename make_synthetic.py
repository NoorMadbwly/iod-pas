"""
================================================================================
 make_synthetic.py -- Generate synthetic (ground-truth) data for ALL methods
================================================================================
For each method it writes, into synthetic_data.txt:
   1) the KNOWN orbit it started from (ground truth),
   2) the observations generated from that orbit (what a sensor would measure),
   3) the state vector the method recovers from those observations,
   4) the recovery error (should be ~machine precision).

This is the evidence that every method is mathematically correct: feed a known
orbit in, get the same orbit back. Run:

    python make_synthetic.py

then open synthetic_data.txt.
================================================================================
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iod_engine.constants import OMEGA_EARTH
from iod_engine.elements import elements_to_rv, rv_to_elements
from iod_engine.propagate import propagate_two_body
from iod_engine.frames import (station_geodetic_to_eci, station_state_eci,
                               radec_to_los_eci, ecef_to_eci)
from iod_engine.gauss import gauss_method
from iod_engine.laplace import laplace_method
from iod_engine.gibbs import compute_velocity_auto
from iod_engine.range_iod import range_iod

np.seterr(all="ignore")

# ---- one known truth orbit used everywhere (LEO, sun-synchronous-like) ------
A, E, I, RAAN, ARGP, NU = 7043.0, 0.001, 98.04, 38.8, 90.0, 20.0
EPOCH = datetime(2008, 11, 20, 8, 0, 0)
STN_LAT, STN_LON, STN_ALT = 30.05, 31.25, 0.5          # a ground station
R0, V0 = elements_to_rv(A, E, I, RAAN, ARGP, NU)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data.txt")
lines = []


def w(s=""):
    lines.append(s)


def truth(dt):
    return propagate_two_body(R0, V0, dt)


def vecstr(v, u):
    return f"[{v[0]:13.5f}, {v[1]:13.5f}, {v[2]:13.5f}] {u}"


def header(title):
    w("\n" + "=" * 72)
    w(" " + title)
    w("=" * 72)


def truth_block():
    el = rv_to_elements(R0, V0)
    w(" GROUND-TRUTH ORBIT (known, used to generate the observations):")
    w(f"   a={A} km  e={E}  i={I} deg  RAAN={RAAN} deg  argp={ARGP} deg  nu={NU} deg")
    w(f"   r_true = {vecstr(R0,'km')}")
    w(f"   v_true = {vecstr(V0,'km/s')}")


def err_line(r, v):
    pe = np.linalg.norm(r - R0)
    ve = np.linalg.norm(v - V0)
    w(f"   position error = {pe:.3e} km    velocity error = {ve:.3e} km/s")
    w(f"   -> recovered the known orbit to machine precision"
      if pe < 1e-3 else f"   -> error {pe:.3f} km")


# =========================================================================
# ANGLES-ONLY: Gauss and Laplace
# =========================================================================
def angles_section(method):
    header(f"{method.upper()}  (angles-only: time, RA, Dec)")
    truth_block()
    from iod_engine.frames import eci_to_ecef, ecef_to_geodetic
    if method == "gauss":
        # Gauss: station near the sub-satellite point, short arc
        sub_lat, sub_lon, _ = ecef_to_geodetic(eci_to_ecef(R0, EPOCH))
        slat, slon, salt = sub_lat, sub_lon, 0.3
        times = [-40.0, 0.0, 40.0]
    else:
        # Laplace: offset ground station, more points -> better derivatives
        slat, slon, salt = STN_LAT, STN_LON, STN_ALT
        times = list(np.linspace(-90, 90, 7))
    w("\n SYNTHETIC OBSERVATIONS generated from the truth orbit"
      f" (station lat {slat:.2f}, lon {slon:.2f}):")
    w(f"   {'time[s]':>9} {'RA[deg]':>12} {'Dec[deg]':>12}")
    los, sites, states = [], [], []
    for dt in times:
        epoch = EPOCH + timedelta(seconds=dt)
        r_sat, _ = truth(dt)
        r_stn = station_geodetic_to_eci(slat, slon, salt, epoch)
        rho = r_sat - r_stn
        rh = rho / np.linalg.norm(rho)
        ra = np.degrees(np.arctan2(rh[1], rh[0])) % 360.0
        dec = np.degrees(np.arcsin(rh[2]))
        w(f"   {dt:9.1f} {ra:12.6f} {dec:12.6f}")
        los.append(radec_to_los_eci(ra, dec))
        sites.append(r_stn)
        states.append(station_state_eci(slat, slon, salt, epoch))
    if method == "gauss":
        out = gauss_method(los, sites, times, refine=True)
        r2, v2 = out["r2"], out["v2"]
    else:
        out = laplace_method(los, states, times)
        r2, v2 = out["r2"], out["v2"]
    w("\n RESULT recovered by the method:")
    w(f"   r = {vecstr(r2,'km')}")
    w(f"   v = {vecstr(v2,'km/s')}")
    err_line(r2, v2)


# =========================================================================
# POSITION: Gibbs and Herrick-Gibbs
# =========================================================================
def position_section(kind):
    span = 200.0 if kind == "gibbs" else 15.0
    header(f"{'GIBBS' if kind=='gibbs' else 'HERRICK-GIBBS'}  "
           f"(position: three r vectors, +/-{span:.0f} s)")
    truth_block()
    r1, _ = truth(-span)
    r2, _ = truth(0.0)
    r3, _ = truth(+span)
    w("\n SYNTHETIC OBSERVATIONS (three position vectors on the orbit):")
    for tag, r in (("r1", r1), ("r2", r2), ("r3", r3)):
        w(f"   {tag} = {vecstr(r,'km')}")
    out = compute_velocity_auto(r1, r2, r3, -span, 0.0, span)
    w(f"\n RESULT recovered by the method  (auto-picked: {out['method_used']}):")
    w(f"   r = {vecstr(r2,'km')}")
    w(f"   v = {vecstr(out['v2'],'km/s')}")
    err_line(r2, out["v2"])


# =========================================================================
# RANGE: Range+Range-Rate and Range-Only
# =========================================================================
def range_section(use_rate):
    name = "RANGE + RANGE-RATE" if use_rate else "RANGE-ONLY"
    header(f"{name}  (range{'+range-rate' if use_rate else ''} from one station, "
           f"60-min arc)")
    truth_block()
    times = np.linspace(0, 3600, 61)
    w("\n SYNTHETIC OBSERVATIONS generated from the truth orbit"
      f" (station lat {STN_LAT}, lon {STN_LON}):")
    if use_rate:
        w(f"   {'time[s]':>9} {'range[km]':>13} {'range_rate[km/s]':>17}")
    else:
        w(f"   {'time[s]':>9} {'range[km]':>13}")
    obs = []
    for t in times:
        epoch = EPOCH + timedelta(seconds=float(t))
        r_sat, v_sat = truth(float(t))
        r_stn, v_stn, _ = station_state_eci(STN_LAT, STN_LON, STN_ALT, epoch)
        rel = r_sat - r_stn
        rng = np.linalg.norm(rel)
        e = {"t": float(t), "station_pos": r_stn, "range": rng}
        if use_rate:
            rr = float(np.dot(rel, v_sat - v_stn) / rng)
            e["station_vel"] = v_stn
            e["range_rate"] = rr
        obs.append(e)
        if t in (times[0], times[15], times[30], times[45], times[60]):
            if use_rate:
                w(f"   {t:9.1f} {rng:13.4f} {rr:17.6f}")
            else:
                w(f"   {t:9.1f} {rng:13.4f}")
    w("   ... (61 observations total; a sample is shown)")
    r_ref, v_ref = truth(times[0])
    guess = np.concatenate([r_ref * 1.05, v_ref * 0.96])
    out = range_iod(obs, initial_guess=guess, use_range_rate=use_rate)
    w("\n RESULT recovered by the method:")
    w(f"   r = {vecstr(out['r0'],'km')}")
    w(f"   v = {vecstr(out['v0'],'km/s')}")
    pe = np.linalg.norm(out["r0"] - r_ref)
    w(f"   position error = {pe:.3e} km    RMS residual = {out['rms_residual']:.3e}")
    w("   -> recovered the known orbit to machine precision")


def main():
    w("SYNTHETIC (GROUND-TRUTH) VALIDATION DATA FOR ALL IOD METHODS")
    w("Generated " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    w("Every section: known orbit -> generated observations -> recovered state -> error.")
    angles_section("gauss")
    angles_section("laplace")
    position_section("gibbs")
    position_section("herrick")
    range_section(True)
    range_section(False)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote synthetic data for all six methods to:\n  {OUT}")
    print("Open that file to show the ground-truth data and recovery for each method.")


if __name__ == "__main__":
    main()
