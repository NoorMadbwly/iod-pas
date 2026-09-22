"""
IOD-PAS -- Initial Orbit Determination, interactive app.

A small web front-end over the IOD engine. Pick a method, provide observations
(upload a file, paste a table, or use a bundled sample), and read the recovered
state vector, the six orbital elements, and a self-consistency check.
"""

import io
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from itertools import combinations

from iod_engine.constants import OMEGA_EARTH
from iod_engine.elements import rv_to_elements, elements_to_rv
from iod_engine.propagate import propagate_two_body
from iod_engine.frames import (station_geodetic_to_eci, station_state_eci,
                               radec_to_los_eci, azel_to_los_eci, ecef_to_eci)
from iod_engine.gauss import gauss_method
from iod_engine.laplace import laplace_method
from iod_engine.gibbs import compute_velocity_auto
from iod_engine.range_iod import range_iod
from iod_engine.geometry import angular_separation_deg, compute_gdop

np.seterr(all="ignore")
W = np.array([0.0, 0.0, OMEGA_EARTH])

st.set_page_config(page_title="IOD-PAS", page_icon="\U0001F6F0", layout="wide")


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def parse_epoch(token, base_date):
    if ":" in token:
        h, m, s = token.split(":")
        return base_date.replace(hour=int(h), minute=int(m), second=int(float(s)))
    return base_date + timedelta(seconds=float(token))


def parse_angle_table(text, base_date):
    rows = []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln or ln[0].isalpha() and ":" not in ln.split()[0]:
            continue
        parts = ln.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            epoch = parse_epoch(parts[0], base_date)
            a1, a2 = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        rows.append((epoch, a1, a2))
    return rows


def show_state(r, v):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Position r (km)**")
        st.code(f"x = {r[0]:14.4f}\ny = {r[1]:14.4f}\nz = {r[2]:14.4f}\n"
                f"|r| = {np.linalg.norm(r):.4f}")
    with c2:
        st.markdown("**Velocity v (km/s)**")
        st.code(f"vx = {v[0]:12.6f}\nvy = {v[1]:12.6f}\nvz = {v[2]:12.6f}\n"
                f"|v| = {np.linalg.norm(v):.6f}")


def show_elements(r, v):
    el = rv_to_elements(r, v)
    st.markdown("**Orbital elements**")
    st.dataframe({
        "Element": ["Semi-major axis a", "Eccentricity e", "Inclination i",
                    "RAAN", "Arg. of perigee", "True anomaly",
                    "Period", "Perigee alt.", "Apogee alt."],
        "Value": [f"{el['semi_major_axis_km']:.3f} km", f"{el['eccentricity']:.6f}",
                  f"{el['inclination_deg']:.4f} deg", f"{el['raan_deg']:.4f} deg",
                  f"{el['arg_perigee_deg']:.4f} deg", f"{el['true_anomaly_deg']:.4f} deg",
                  f"{el['period_s']/60:.2f} min", f"{el['perigee_altitude_km']:.1f} km",
                  f"{el['apogee_altitude_km']:.1f} km"],
    }, hide_index=True, use_container_width=True)
    return el


# --------------------------------------------------------------------------
# angles-only (Gauss / Laplace)
# --------------------------------------------------------------------------
VALLADO = """11:40:28.00   0.939913   18.667717
11:44:28.00  21.235600   29.086871
11:48:28.00  45.025748   35.664741
11:52:28.00  67.886655   36.996583
11:56:28.00  86.208078   34.719667"""


def _los_for(kind, a1, a2, lat, lon, epoch):
    if kind == "RA / Dec":
        return radec_to_los_eci(a1, a2)
    return azel_to_los_eci(a1, a2, lat, lon, epoch)


def _all_residual(r2, v2, centre_dt, rows, kind, lat, lon):
    sq, n = 0.0, 0
    for epoch, a1, a2 in rows:
        dt = (epoch - rows[0][0]).total_seconds()
        rp, _ = propagate_two_body(r2, v2, dt - centre_dt)
        if not np.all(np.isfinite(rp)):
            return np.inf
        st_pos = station_geodetic_to_eci(lat, lon, 0.1, epoch)
        pred = rp - st_pos
        pred /= np.linalg.norm(pred)
        obs = _los_for(kind, a1, a2, lat, lon, epoch)
        sq += np.degrees(np.arccos(np.clip(np.dot(pred, obs), -1, 1))) ** 2
        n += 1
    return np.sqrt(sq / n) if n else np.inf


def auto_triplet(rows, kind, lat, lon):
    keys = list(range(len(rows)))
    t0 = rows[0][0]
    los = {i: _los_for(kind, rows[i][1], rows[i][2], lat, lon, rows[i][0]) for i in keys}
    sites = {i: station_geodetic_to_eci(lat, lon, 0.1, rows[i][0]) for i in keys}
    times = {i: (rows[i][0] - t0).total_seconds() for i in keys}
    best = None
    for i, j, k in combinations(keys, 3):
        sep = angular_separation_deg(los[i], los[k])
        if sep < 3 or sep > 60:
            continue
        try:
            out = gauss_method([los[i], los[j], los[k]],
                               [sites[i], sites[j], sites[k]],
                               [times[i], times[j], times[k]], refine=True)
            if out["v2"] is None:
                continue
            res = _all_residual(out["r2"], out["v2"], times[j], rows, kind, lat, lon)
        except Exception:
            continue
        if best is None or res < best[1]:
            best = ((i, j, k), res, sep, compute_gdop([los[i], los[j], los[k]]))
    return best


def run_angles(method):
    st.subheader(f"{method} \u2014 angles-only")
    c1, c2, c3, c4 = st.columns(4)
    lat = c1.number_input("Station latitude (deg)", value=40.0, format="%.4f")
    lon = c2.number_input("Station longitude (deg)", value=-110.0, format="%.4f")
    alt = c3.number_input("Station altitude (km)", value=2.0, format="%.3f")
    date = c4.date_input("Observation date (UTC)", value=datetime(2012, 8, 20))
    kind = st.radio("Angle type", ["RA / Dec", "Az / El"], horizontal=True)

    st.caption("Paste rows: `time  angle1  angle2` (time as HH:MM:SS or seconds).")
    if st.button("Load Vallado Example 7-2 sample"):
        st.session_state["ang_text"] = VALLADO
    text = st.text_area("Observations", key="ang_text",
                        value=st.session_state.get("ang_text", VALLADO), height=170)

    base = datetime(date.year, date.month, date.day)
    rows = parse_angle_table(text, base)
    st.write(f"Parsed **{len(rows)}** observations.")

    if method == "Gauss":
        mode = st.radio("Point selection", ["Auto (best 3)", "Manual"], horizontal=True)
        chosen = None
        if mode == "Manual":
            idx = st.multiselect("Pick exactly 3 (by row number, 1-based)",
                                 list(range(1, len(rows) + 1)))
            chosen = [i - 1 for i in idx] if len(idx) == 3 else None
    if not st.button("Run", type="primary"):
        return
    if len(rows) < 3:
        st.error("Need at least 3 observations.")
        return

    if method == "Gauss":
        if mode == "Auto (best 3)":
            best = auto_triplet(rows, kind, lat, lon)
            if not best:
                st.error("Auto-select failed; try manual points.")
                return
            sel = list(best[0])
            st.success(f"Auto-selected rows {[s+1 for s in sel]} \u2014 "
                       f"residual {best[1]:.4f} deg RMS, separation {best[2]:.1f} deg, "
                       f"GDOP {best[3]:.2f}")
        else:
            if not chosen:
                st.error("Select exactly 3 rows.")
                return
            sel = chosen
        pts = [rows[i] for i in sel]
        t0 = pts[0][0]
        times = [(p[0] - t0).total_seconds() for p in pts]
        los = [_los_for(kind, p[1], p[2], lat, lon, p[0]) for p in pts]
        sites = [station_geodetic_to_eci(lat, lon, alt, p[0]) for p in pts]
        out = gauss_method(los, sites, times, refine=True)
        r2, v2 = out["r2"], out["v2"]
        centre_dt = times[1]
    else:
        t0 = rows[0][0]
        times = [(r[0] - t0).total_seconds() for r in rows]
        los = [_los_for(kind, r[1], r[2], lat, lon, r[0]) for r in rows]
        states = [station_state_eci(lat, lon, alt, r[0]) for r in rows]
        out = laplace_method(los, states, times)
        r2, v2 = out["r2"], out["v2"]
        centre_dt = times[len(rows) // 2]

    if v2 is None:
        st.error("The method did not return a velocity for these data.")
        return
    st.divider()
    show_state(r2, v2)
    show_elements(r2, v2)


# --------------------------------------------------------------------------
# ASN / ITNP readers (real tracking files)
# --------------------------------------------------------------------------
def read_asn_text(text):
    date = None
    recs = []
    for ln in text.splitlines():
        if ln.startswith("NKA"):
            for tok in ln.split():
                if tok.startswith("DAT="):
                    date = tok[4:]
        p = ln.split()
        if len(p) == 7:
            try:
                recs.append([float(x) for x in p])
            except ValueError:
                pass
    d, m, y = (int(x) for x in date.split("."))
    base = datetime(y, m, d)
    out = []
    for r in recs:
        epoch = base + timedelta(seconds=r[0])
        div = 1000.0 if abs(r[1]) > 1e5 else 1.0
        r_ecef = np.array(r[1:4]) / div
        v_ecef = np.array(r[4:7]) / div
        r_eci = ecef_to_eci(r_ecef, epoch)
        v_eci = ecef_to_eci(v_ecef, epoch) + np.cross(W, r_eci)
        out.append({"epoch": epoch, "r": r_eci, "v": v_eci})
    return out


def read_itnp_text(text):
    date = None
    rows = []
    for ln in text.splitlines():
        if ln.startswith("KNC"):
            for tok in ln.split():
                if tok.startswith("DAT="):
                    date = tok[4:]
        p = ln.replace("\r", "").split()
        if len(p) == 3:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    d, m, y = (int(x) for x in date.split("."))
    base = datetime(y, m, d)
    div = 1000.0 if rows and abs(rows[0][1]) > 1e5 else 1.0
    return [{"epoch": base + timedelta(seconds=r[0]),
             "range_km": r[1] / div, "range_rate_kms": r[2] / div} for r in rows]


def _read_upload_or_sample(kind):
    up = st.file_uploader(f"Upload a {kind} file", type=None)
    if up is not None:
        return up.getvalue().decode("utf-8", errors="ignore")
    import os
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")
    try:
        opts = sorted(f for f in os.listdir(sdir)
                      if (kind.lower() in f.lower()
                          or (kind == "ASN" and "asn" in f.lower())
                          or (kind == "ITNP" and "iz" in f.lower())))
    except FileNotFoundError:
        opts = []
    if opts:
        pick = st.selectbox(f"...or use a bundled {kind} sample", ["(none)"] + opts)
        if pick != "(none)":
            with open(os.path.join(sdir, pick)) as fh:
                return fh.read()
    return None


# --------------------------------------------------------------------------
# position (Gibbs / Herrick-Gibbs)
# --------------------------------------------------------------------------
def run_position(method):
    st.subheader(f"{method} \u2014 position vectors")
    st.caption("Provide an ASN state-vector file; three positions are taken from "
               "the real orbit to recover the velocity.")
    text = _read_upload_or_sample("ASN")
    span = st.slider("Spacing between the 3 points (s)",
                     5, 400, 200 if method == "Gibbs" else 15)
    if not st.button("Run", type="primary"):
        return
    if not text:
        st.error("Upload an ASN file or choose a sample.")
        return
    recs = read_asn_text(text)
    if not recs:
        st.error("No ASN state vectors found in the file.")
        return
    ref = recs[len(recs) // 2]
    r0, v0 = ref["r"], ref["v"]
    r1, _ = propagate_two_body(r0, v0, -span)
    r2, _ = propagate_two_body(r0, v0, 0.0)
    r3, _ = propagate_two_body(r0, v0, +span)
    out = compute_velocity_auto(r1, r2, r3, -span, 0.0, span)
    st.success(f"Method auto-picked: **{out['method_used']}**  "
               f"(epoch {ref['epoch']} UTC)")
    st.divider()
    show_state(r2, out["v2"])
    show_elements(r2, out["v2"])
    v_err = np.linalg.norm(out["v2"] - v0) * 1000
    st.metric("Velocity error vs the file's own velocity", f"{v_err:.4f} m/s")


# --------------------------------------------------------------------------
# range (Range+Range-Rate / Range-Only)
# --------------------------------------------------------------------------
def run_range(use_rate):
    name = "Range + Range-Rate" if use_rate else "Range-Only"
    st.subheader(f"{name} \u2014 from a ground station")
    text = _read_upload_or_sample("ITNP")
    c1, c2, c3 = st.columns(3)
    x = c1.number_input("Station ECEF X (km)", value=4706.1195, format="%.4f")
    y = c2.number_input("Station ECEF Y (km)", value=2895.9984, format="%.4f")
    z = c3.number_input("Station ECEF Z (km)", value=3175.3805, format="%.4f")
    st.caption("Default station is Cairo (KNC=001). An a-priori is estimated by "
               "trilateration from the data.")
    if not st.button("Run", type="primary"):
        return
    if not text:
        st.error("Upload an ITNP file or choose a sample.")
        return
    itnp = read_itnp_text(text)
    if not itnp:
        st.error("No ITNP range records found in the file.")
        return
    station = np.array([x, y, z])
    t0 = itnp[0]["epoch"]
    obs = []
    for o in itnp:
        r_stn = ecef_to_eci(station, o["epoch"])
        v_stn = np.cross(W, r_stn)
        obs.append({"t": (o["epoch"] - t0).total_seconds(),
                    "station_pos": r_stn, "station_vel": v_stn,
                    "range": o["range_km"], "range_rate": o["range_rate_kms"]})
    with st.spinner("Solving..."):
        out = range_iod(obs, use_range_rate=use_rate, max_iter=60)
    st.success(f"Converged={out['converged']}  "
               f"RMS residual={out['rms_residual']:.3e} km  "
               f"({len(obs)} observations, pass {t0} UTC)")
    st.divider()
    show_state(out["r0"], out["v0"])
    show_elements(out["r0"], out["v0"])
    span_min = (itnp[-1]["epoch"] - itnp[0]["epoch"]).total_seconds() / 60.0
    if span_min < 20:
        st.warning("This pass is short (%.0f min) from a single station. Such data "
                   "is under-observable: the fit to the ranges can be excellent while "
                   "the orbit is not uniquely determined. A longer arc or extra "
                   "stations are needed for a unique orbit." % span_min)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
st.title("IOD-PAS \u2014 Initial Orbit Determination")
st.caption("Angles-only (Gauss, Laplace) \u00b7 Position (Gibbs, Herrick-Gibbs) "
           "\u00b7 Range (Range+Range-Rate, Range-Only). Reference frame: GCRF.")

method = st.sidebar.radio(
    "Method",
    ["Gauss", "Laplace", "Gibbs", "Herrick-Gibbs",
     "Range + Range-Rate", "Range-Only"])
st.sidebar.markdown("---")
st.sidebar.caption("Gauss/Laplace take an angle table. Gibbs/Herrick-Gibbs take "
                   "an ASN file. Range methods take an ITNP file.")

if method in ("Gauss", "Laplace"):
    run_angles(method)
elif method in ("Gibbs", "Herrick-Gibbs"):
    run_position(method)
elif method == "Range + Range-Rate":
    run_range(True)
else:
    run_range(False)
