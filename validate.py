"""
================================================================================
 validate.py -- Quick numeric validation of Gauss / Laplace on a known problem
================================================================================
Edit the three blocks below (STATION, OBSERVATIONS, CHOICE), then run:

    python validate.py

It prints the position/velocity vectors, the six orbital elements, and a
self-consistency check (re-predicting the observations from the recovered
orbit). Designed to be run live to validate the method against a textbook
example such as Vallado Example 7-2.
================================================================================
"""

import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")
np.seterr(all="ignore")        # candidate orbits during auto-search can overflow harmlessly

from iod_engine.frames import (station_geodetic_to_eci, station_state_eci,
                               radec_to_los_eci, azel_to_los_eci)
from iod_engine.gauss import gauss_method
from iod_engine.laplace import laplace_method
from iod_engine.elements import rv_to_elements
from iod_engine.propagate import propagate_two_body


# ============================================================================
# 1) STATION  -- where the observations were taken from
# ============================================================================
STATION = dict(lat_deg=40.0, lon_deg=-110.0, alt_km=2.0)   # Vallado Ex. 7-2
DATE = datetime(2012, 8, 20)                                # observation date


# ============================================================================
# 2) OBSERVATIONS  -- one row per observation
#    angle_type: "radec" (RA/Dec) or "azel" (Azimuth/Elevation)
#    columns: ("HH:MM:SS", angle1_deg, angle2_deg)
# ============================================================================
ANGLE_TYPE = "radec"
OBSERVATIONS = {
    1:  ("11:32:28.00", 333.028738,  -2.022317),
    2:  ("11:36:28.00", 345.235515,   7.648921),
    3:  ("11:40:28.00",   0.939913,  18.667717),
    4:  ("11:44:28.00",  21.235600,  29.086871),
    5:  ("11:48:28.00",  45.025748,  35.664741),
    6:  ("11:52:28.00",  67.886655,  36.996583),
    7:  ("11:56:28.00",  86.208078,  34.719667),
    8:  ("12:00:28.00",  99.845522,  30.928387),
    9:  ("12:04:28.00", 110.078585,  26.767438),
    10: ("12:08:28.00", 118.058822,  22.680214),
    11: ("12:12:28.00", 124.552101,  18.801899),
    12: ("12:16:28.00", 130.038810,  15.154469),
    13: ("12:20:28.00", 134.823380,  11.722965),
    14: ("12:24:28.00", 139.104912,   8.483073),
    15: ("12:26:28.00", 141.101028,   6.927305),
}


# ============================================================================
# 3) CHOICE  -- which method, and which observation numbers to use
#    METHOD:  "gauss"  (needs exactly 3 points)
#             "laplace" (needs 3 or more; more points => better derivatives)
#
#    POINTS:  "auto"      -> the system chooses the best 3 points itself
#                            (and tells you which ones and why)
#             [3, 5, 6]   -> you choose the points yourself
#
#    Tip: run "auto" first; if you don't like the choice, type your own list
#    and run again to compare.
# ============================================================================
METHOD = "gauss"
POINTS = "auto"


# ----------------------------------------------------------------------------
# (below this line: nothing to edit)
# ----------------------------------------------------------------------------
def _epoch(hms):
    h, m, s = hms.split(":")
    return DATE.replace(hour=int(h), minute=int(m), second=int(float(s)))


def _los(a1, a2, epoch):
    if ANGLE_TYPE == "radec":
        return radec_to_los_eci(a1, a2)
    if ANGLE_TYPE == "azel":
        return azel_to_los_eci(a1, a2, STATION["lat_deg"], STATION["lon_deg"], epoch)
    raise ValueError("ANGLE_TYPE must be 'radec' or 'azel'")


def _all_observation_residual(r2, v2, centre_dt, all_keys, all_epochs,
                              all_times, t_origin):
    """RMS angular miss (deg) between predicted and observed LOS over ALL
    observations -- the physics-based quality score for a candidate orbit."""
    sq = 0.0
    n = 0
    for k, e, t in zip(all_keys, all_epochs, all_times):
        try:
            r_pred, _ = propagate_two_body(r2, v2, (t) - centre_dt)
        except Exception:
            return np.inf
        if not np.all(np.isfinite(r_pred)):
            return np.inf
        r_stn = station_geodetic_to_eci(STATION["lat_deg"], STATION["lon_deg"],
                                        STATION["alt_km"], e)
        pred = (r_pred - r_stn)
        pred /= np.linalg.norm(pred)
        obs_los = _los(OBSERVATIONS[k][1], OBSERVATIONS[k][2], e)
        cosang = np.clip(np.dot(pred, obs_los), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        sq += ang ** 2
        n += 1
    return np.sqrt(sq / n) if n else np.inf


def _auto_select(all_keys, all_epochs, all_times, t_origin):
    """Try candidate triplets, solve, and keep the one whose orbit best
    reproduces ALL the observations. Returns (best_keys, report_lines)."""
    from itertools import combinations
    from iod_engine.geometry import angular_separation_deg, compute_gdop

    los_by_key = {k: _los(OBSERVATIONS[k][1], OBSERVATIONS[k][2], e)
                  for k, e in zip(all_keys, all_epochs)}
    site_by_key = {k: station_geodetic_to_eci(
        STATION["lat_deg"], STATION["lon_deg"], STATION["alt_km"], e)
        for k, e in zip(all_keys, all_epochs)}
    time_by_key = dict(zip(all_keys, all_times))
    epoch_by_key = dict(zip(all_keys, all_epochs))

    best = None
    tested = 0
    for combo in combinations(all_keys, 3):
        i, j, k = combo
        li, lj, lk = los_by_key[i], los_by_key[j], los_by_key[k]
        sep = angular_separation_deg(li, lk)
        if sep < 3.0 or sep > 60.0:          # skip singular / over-wide arcs
            continue
        times3 = [time_by_key[i], time_by_key[j], time_by_key[k]]
        sites3 = [site_by_key[i], site_by_key[j], site_by_key[k]]
        try:
            out = gauss_method([li, lj, lk], sites3, times3, refine=True)
            if out["v2"] is None:
                continue
            res = _all_observation_residual(out["r2"], out["v2"], times3[1],
                                            all_keys, all_epochs, all_times,
                                            t_origin)
        except Exception:
            continue
        tested += 1
        gdop = compute_gdop([li, lj, lk])
        if best is None or res < best["res"]:
            best = {"keys": combo, "res": res, "sep": sep, "gdop": gdop}

    lines = []
    if best is None:
        raise SystemExit("Auto-select failed: no usable triplet. "
                         "Enter POINTS manually, e.g. [3, 5, 6].")
    lines.append(f" AUTO-SELECT: tested {tested} candidate triplets")
    lines.append(f" -> best 3 points: {list(best['keys'])}")
    lines.append(f"    reason: lowest all-observation residual "
                 f"({best['res']:.4f} deg RMS over every observation)")
    lines.append(f"    geometry: first-to-last separation {best['sep']:.1f} deg, "
                 f"GDOP {best['gdop']:.2f}")
    return list(best["keys"]), lines


def main():
    all_keys = sorted(OBSERVATIONS.keys())
    all_epochs = [_epoch(OBSERVATIONS[k][0]) for k in all_keys]
    t_origin = all_epochs[0]
    all_times = [(e - t_origin).total_seconds() for e in all_epochs]

    auto_report = []
    if isinstance(POINTS, str) and POINTS.lower() == "auto":
        pts, auto_report = _auto_select(all_keys, all_epochs, all_times, t_origin)
    else:
        pts = POINTS

    epochs = [_epoch(OBSERVATIONS[k][0]) for k in pts]
    t0 = epochs[0]
    times = [(e - t0).total_seconds() for e in epochs]
    los = [_los(OBSERVATIONS[k][1], OBSERVATIONS[k][2], e)
           for k, e in zip(pts, epochs)]

    print("=" * 70)
    if auto_report:
        for ln in auto_report:
            print(ln)
        print("-" * 70)
    print(f" METHOD: {METHOD.upper()}     POINTS USED: {pts}")
    print(f" Station: lat {STATION['lat_deg']}, lon {STATION['lon_deg']}, "
          f"alt {STATION['alt_km']} km   angle type: {ANGLE_TYPE}")
    print("=" * 70)

    if METHOD == "gauss":
        if len(pts) != 3:
            raise SystemExit("Gauss needs exactly 3 points.")
        sites = [station_geodetic_to_eci(STATION["lat_deg"], STATION["lon_deg"],
                                         STATION["alt_km"], e) for e in epochs]
        out = gauss_method(los, sites, times, refine=True)
        r2, v2 = out["r2"], out["v2"]
        centre_epoch = epochs[1]
        centre_dt_offset = times[1]

    elif METHOD == "laplace":
        if len(pts) < 3:
            raise SystemExit("Laplace needs 3 or more points.")
        states = [station_state_eci(STATION["lat_deg"], STATION["lon_deg"],
                                    STATION["alt_km"], e) for e in epochs]
        out = laplace_method(los, states, times)
        r2, v2 = out["r2"], out["v2"]
        centre = len(pts) // 2
        centre_epoch = epochs[centre]
        centre_dt_offset = times[centre]
        print(f" slant range = {out['slant_range_km']:.3f} km   "
              f"rate = {out['slant_range_rate_kms']:.5f} km/s")
    else:
        raise SystemExit("METHOD must be 'gauss' or 'laplace'")

    print(f"\n Central epoch: {centre_epoch}  UTC\n")
    print(" STATE VECTOR")
    print(f"   r = [{r2[0]:12.4f}, {r2[1]:12.4f}, {r2[2]:12.4f}] km   "
          f"|r| = {np.linalg.norm(r2):.4f} km")
    print(f"   v = [{v2[0]:12.6f}, {v2[1]:12.6f}, {v2[2]:12.6f}] km/s  "
          f"|v| = {np.linalg.norm(v2):.6f} km/s")

    el = rv_to_elements(r2, v2)
    print("\n ORBITAL ELEMENTS")
    print(f"   a    = {el['semi_major_axis_km']:12.4f} km")
    print(f"   e    = {el['eccentricity']:12.6f}")
    print(f"   i    = {el['inclination_deg']:12.4f} deg")
    print(f"   RAAN = {el['raan_deg']:12.4f} deg")
    print(f"   argp = {el['arg_perigee_deg']:12.4f} deg")
    print(f"   nu   = {el['true_anomaly_deg']:12.4f} deg")
    print(f"   perigee alt = {el['perigee_altitude_km']:.1f} km   "
          f"apogee alt = {el['apogee_altitude_km']:.1f} km   "
          f"period = {el['period_s']/60:.2f} min")

    # self-consistency: re-predict every used observation from the orbit
    print("\n SELF-CONSISTENCY CHECK  (re-predict the observations)")
    print("   obs |  RA/az pred   obs      d      |  Dec/el pred  obs      d")
    print("   " + "-" * 63)
    max_ang = 0.0
    for k, e, t in zip(pts, epochs, times):
        r_pred, _ = propagate_two_body(r2, v2, t - centre_dt_offset)
        r_stn = station_geodetic_to_eci(STATION["lat_deg"], STATION["lon_deg"],
                                        STATION["alt_km"], e)
        rho = r_pred - r_stn
        rh = rho / np.linalg.norm(rho)
        ra = np.degrees(np.arctan2(rh[1], rh[0])) % 360.0
        dec = np.degrees(np.arcsin(np.clip(rh[2], -1, 1)))
        ra_o = OBSERVATIONS[k][1] % 360.0
        dec_o = OBSERVATIONS[k][2]
        d_ra = (ra - ra_o + 180) % 360 - 180
        d_dec = dec - dec_o
        max_ang = max(max_ang, abs(d_ra), abs(d_dec))
        print(f"   {k:3d} | {ra:9.4f} {ra_o:9.4f} {d_ra:+7.4f} | "
              f"{dec:9.4f} {dec_o:9.4f} {d_dec:+7.4f}")
    print(f"\n   largest angular residual: {max_ang:.4f} deg")
    print("   (small residual on the points NOT at the centre = good orbit)")


if __name__ == "__main__":
    main()
