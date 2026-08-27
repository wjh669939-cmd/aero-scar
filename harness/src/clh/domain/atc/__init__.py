"""ATC / aerodrome-weather domain notes for the research agent."""

ATC_RESEARCH_BRIEF = """
Research problem: airport weather prediction and ATC-informed decision support.
Baseline: AeroWF-style compact model (dummy persistence here; swap pipeline_root to a real AeroWF tree).
Multi-source data: AWOS/METAR-like observations, operational context, external evidence.

Closed-loop axes (one per trial, file-level lock):
- data: discover/augment labelled evidence through the leakage filter only
- representation: runway-relative wind, METAR encodings, multi-scale features
- model: estimator family, capacity, fusion, probability heads
- physics: meteorological mechanisms, extreme-event losses, ATC domain constraints

Certification (search never reads these labels):
- temporal: future hours of seen airports
- spatial: unseen airport
- event: held-out storm processes
"""
