# IOD-PAS — Initial Orbit Determination

A compact engine and web app that determine an orbit from tracking observations.
All six classical initial-orbit-determination (IOD) methods are implemented behind
one interface and exposed through a small Streamlit app.

| Class | Methods | Input |
|-------|---------|-------|
| Angles-only | Gauss, Laplace | time + two angles (RA/Dec or Az/El) from a station |
| Position | Gibbs, Herrick-Gibbs | three position vectors |
| Range | Range + Range-Rate, Range-Only | range (and range-rate) from a station |

Every method returns the same result: an epoch **state vector** (r, v), the full
set of **six orbital elements**, and a **self-consistency** check. Reference frame
is GCRF/ECI.

## Live app

Pick a method in the sidebar, provide observations (upload a file, paste a table,
or use a bundled sample), and read the recovered state and elements.

- **Gauss / Laplace** — paste an angle table (`time  angle1  angle2`) and set the
  station. Gauss can auto-select the best three points or let you choose them.
- **Gibbs / Herrick-Gibbs** — upload an ASN state-vector file (or use a sample).
- **Range / Range-Only** — upload an ITNP range file (or use a sample) and set the
  station coordinates (Cairo is the default).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. "Create app" → select this repository, branch `main`, main file `app.py`.
4. Deploy. The app builds from `requirements.txt` automatically.

## Command-line validation

Beyond the app, three scripts print numeric validation:

```bash
python validate.py          # Gauss / Laplace on an angle table (Vallado Example 7-2 built in)
python validate_range.py    # Gibbs, Herrick-Gibbs, Range, Range-Only on the sample data
python make_synthetic.py    # writes synthetic_data.txt: known orbit -> observations -> recovery, per method
```

`make_synthetic.py` is the mathematical proof: each method is fed observations
generated from a known orbit and recovers that orbit (position/range methods to
machine precision; angles-only to sub-kilometre, as expected).

## Repository layout

```
iod-pas/
├── app.py               # Streamlit web app
├── iod_engine/          # the engine (all six methods + frames, time, elements, propagation)
├── validate.py          # angles-only numeric validation
├── validate_range.py    # position/range numeric validation
├── make_synthetic.py    # ground-truth data generator (all methods)
├── data/
│   ├── ground_stations.xlsx
│   └── samples/         # example ASN and ITNP tracking files
├── requirements.txt
└── README.md
```

## Notes on accuracy

- Position (Gibbs/Herrick-Gibbs) and range methods recover a known orbit to
  machine precision when the observation arc is adequate.
- Angles-only methods (Gauss/Laplace) are inherently less precise and depend on
  geometry and arc length.
- A single short pass from one station is **under-observable**: the fit to the
  measurements can be excellent while the orbit is not uniquely determined. A
  longer arc or additional stations resolves this.
