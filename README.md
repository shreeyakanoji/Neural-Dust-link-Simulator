# Neural Dust Link-Budget Simulator

An interactive engineering simulator for ultrasonic power delivery and
backscatter communication in implantable "neural dust"-style sensors,
following the design approach introduced in Seo et al. 2013 (UC Berkeley) 
with reference to Charles Lieber's syringe-injectable mesh electronics work
as a related minimally-invasive neural interface approach.

**[Live demo →](#)** (Will upload soon hehe)

## What it does

Given a target implant depth, operating frequency, and TX power budget,
the simulator computes:

- A full two-way link budget: TX transducer → layered tissue → sub-mm mote
  receive coupling → backscatter modulation → tissue → RX, in dB
- The operating frequency that maximizes received SNR at a given depth,
  under a **physical mote-size constraint** (capped at 1mm — the actual
  constraint that makes "dust" dust, and what makes the frequency/depth
  tradeoff real rather than trivial)
- A 2D spatial SNR map through a synthetic head cross-section (skin →
  skull → brain), showing near-field beam collimation and far-field
  spreading
- A 3D interactive volumetric view of the same field, built by revolving
  the validated 2D axial model around the depth axis (physically justified
  for a circular, rotationally-symmetric transducer)

## Why it's structured this way

This isn't a formula-chain toy model. The interesting engineering question
is: *given a target depth and a power budget, what frequency and mote size
maximize received SNR?* That's a real tradeoff  higher frequency improves
transducer coupling efficiency but tissue attenuation gets worse, and mote
size and frequency aren't independent (aperture wants to scale with
wavelength, but is capped by the sub-mm form factor that defines "dust").
The optimizer surfaces that tradeoff instead of just plotting a curve.

## Accuracy — what's validated vs. heuristic

This project treats its own accuracy honestly rather than presenting every
number with equal confidence. Full breakdown with quantified error metrics
lives in `validation.py` and is also shown live in the app under
"Model accuracy." Short version:

- **Validated (exact by construction):** near field length formula,
  reflection coefficients, power-law attenuation form
- **Checked against reference values:** soft-tissue attenuation vs. the
  ~1 dB/cm/MHz rule of thumb, skull insertion loss vs. commonly cited
  10-20 dB range, optimal frequency vs. the published neural dust
  operating point (order-of-magnitude match)
- **Heuristic, not independently validated:** piezo receive-coupling
  efficiency (simplified penalty function, not a derived KLM/Mason
  circuit model  the single biggest accuracy gap here), lateral beam
  falloff (Gaussian approximation, no true diffraction pattern),
  backscatter modulation loss (fixed placeholder)
- **Known limitation:** the spatial/3D views are ray-based, not a
  wave-equation solve no interference, refraction, or multi-path.
  A k-Wave (pseudospectral time-domain) simulation is the natural
  research-grade next step.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
tissue.py         layered tissue attenuation/impedance model
link_budget.py     two-way TX -> tissue -> mote -> RX link budget
sweep.py            frequency optimizer with physical mote-size constraint
spatial_map.py     2D spatial SNR field over a synthetic head cross-section
view_3d.py          3D volumetric field (revolved from the 2D model)
validation.py       quantified accuracy checks against reference values
test_sanity.py      unit tests
app.py               Streamlit app tying it all together
```

## Roadmap

- Replace the heuristic piezo coupling penalty with a real KLM/Mason
  equivalent-circuit model (ABCD matrices for cascaded layers)
- Multi-layer tissue stack refinement per target site (transcranial vs.
  peripheral) with sourced literature values per layer
- k-Wave-based wave-equation simulation for real diffraction/multi-path
  effects, once the link-budget model above is fully circuit-validated

## Background

Built as part of ongoing exploration into non-invasive brain-computer
interfaces, alongside EEG/ECG-based intent and emotion decoding work
(see [Emotion-and-Intent-Recognition-software](https://github.com/shreeyakanoji/Intent-Recognizer)).
