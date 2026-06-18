"""
parameter_key.py
================
Single source of truth for tRIBS/pytRIBS calibration parameter metadata.

All plot scripts, summary tables, and reporting tools should import from
here rather than defining their own labels.  CSV column names and tRIBS
attribute names are NOT changed — those remain as-is in the scripts and
output files.

Fields per entry
----------------
code_name    : key used in BASELINE dict, LHS_PARAMS, and CSV output columns
display_name : full human-readable name for tables, figure titles, captions
symbol       : concise physics symbol for plot axis labels and legends
units        : physical units string ('' = dimensionless)
category     : broad grouping for figures and reports
short_tag    : compact label used in run ID strings

Usage
-----
    from parameter_key import PARAM_KEY, METRIC_KEY

    label = PARAM_KEY["kinemvelcoef"]["display_name"]   # → "Hillslope velocity coefficient"
    sym   = PARAM_KEY["kinemvelcoef"]["symbol"]          # → "cv"
    tag   = PARAM_KEY["kinemvelcoef"]["short_tag"]       # → "cv"
"""

# ---------------------------------------------------------------------------
# CALIBRATION PARAMETERS
# ---------------------------------------------------------------------------
PARAM_KEY = {

    # -- Soil / infiltration ------------------------------------------------
    "Ks_mult": {
        "code_name":    "Ks_mult",
        "display_name": "Saturated hydraulic conductivity multiplier",
        "symbol":       "Ks",
        "units":        "multiplier",
        "category":     "Soil / infiltration",
        "short_tag":    "Ks",
    },
    "f_RS_abs": {
        "code_name":    "f_RS_abs",
        "display_name": "Hydraulic conductivity decay rate (RS class)",
        "symbol":       "f",
        "units":        "mm\u207b\u00b9",          # mm⁻¹
        "category":     "Soil / infiltration",
        "short_tag":    "fRS",
    },
    "thetaS_mult": {
        "code_name":    "thetaS_mult",
        "display_name": "Saturated soil moisture multiplier",
        "symbol":       "\u03b8s",                 # θs
        "units":        "multiplier",
        "category":     "Soil / infiltration",
        "short_tag":    "thS",
    },
    "psiB_mult": {
        "code_name":    "psiB_mult",
        "display_name": "Air entry pressure multiplier",
        "symbol":       "\u03c8B",                 # ψB
        "units":        "multiplier",
        "category":     "Soil / infiltration",
        "short_tag":    "psiB",
    },
    "As_value": {
        "code_name":    "As_value",
        "display_name": "Saturated anisotropy ratio",
        "symbol":       "As",
        "units":        "",
        "category":     "Soil / infiltration",
        "short_tag":    "As",
    },
    "Au_value": {
        "code_name":    "Au_value",
        "display_name": "Unsaturated anisotropy ratio",
        "symbol":       "Au",
        "units":        "",
        "category":     "Soil / infiltration",
        "short_tag":    "Au",
    },
    "AsAu_value": {
        "code_name":    "AsAu_value",
        "display_name": "Anisotropy ratio (As = Au)",
        "symbol":       "As=Au",
        "units":        "",
        "category":     "Soil / infiltration",
        "short_tag":    "AsAu",
    },

    # -- Routing — hillslope ------------------------------------------------
    "kinemvelcoef": {
        "code_name":    "kinemvelcoef",
        "display_name": "Hillslope velocity coefficient",
        "symbol":       "cv",
        "units":        "",
        "category":     "Routing \u2014 hillslope",
        "short_tag":    "cv",
    },
    "flowexp": {
        "code_name":    "flowexp",
        "display_name": "Hillslope velocity exponent",
        "symbol":       "r",
        "units":        "",
        "category":     "Routing \u2014 hillslope",
        "short_tag":    "r",
    },

    # -- Routing — channel --------------------------------------------------
    "channelroughness": {
        "code_name":    "channelroughness",
        "display_name": "Channel Manning's roughness",
        "symbol":       "n",
        "units":        "",
        "category":     "Routing \u2014 channel",
        "short_tag":    "n",
    },
    "channelwidthcoeff": {
        "code_name":    "channelwidthcoeff",
        "display_name": "Channel width coefficient",
        "symbol":       "\u03b1B",                 # αB
        "units":        "",
        "category":     "Routing \u2014 channel",
        "short_tag":    "cw",
    },

    # -- Channel percolation (inactive when OPTPERCOLATION = 0) ------------
    "optpercolation": {
        "code_name":    "optpercolation",
        "display_name": "Channel percolation option",
        "symbol":       "",
        "units":        "flag",
        "category":     "Channel percolation",
        "short_tag":    "optperc",
    },
    "channelconductivity_mmhr": {
        "code_name":    "channelconductivity_mmhr",
        "display_name": "Channel bed hydraulic conductivity",
        "symbol":       "Kch",
        "units":        "mm hr\u207b\u00b9",        # mm hr⁻¹
        "category":     "Channel percolation",
        "short_tag":    "Kch",
    },
    "channelporosity": {
        "code_name":    "channelporosity",
        "display_name": "Channel bed porosity",
        "symbol":       "\u03c6ch",                # φch
        "units":        "",
        "category":     "Channel percolation",
        "short_tag":    "porCh",
    },
}


# ---------------------------------------------------------------------------
# HYDROGRAPH METRICS
# ---------------------------------------------------------------------------
METRIC_KEY = {

    # -- Pre-peak -----------------------------------------------------------
    "first_arrival_error_min": {
        "code_name":    "first_arrival_error_min",
        "display_name": "First arrival error",
        "symbol":       "",
        "units":        "min",
        "phase":        "pre-peak",
        "ideal":        0.0,
        "direction":    -1,   # zero error is best
    },
    "rising_limb_steepness_ratio": {
        "code_name":    "rising_limb_steepness_ratio",
        "display_name": "Rising limb steepness ratio",
        "symbol":       "",
        "units":        "",
        "phase":        "pre-peak",
        "ideal":        1.0,
        "direction":    -1,
    },
    "time_to_peak_from_exc_min": {
        "code_name":    "time_to_peak_from_exc_min",
        "display_name": "Time to peak from threshold",
        "symbol":       "",
        "units":        "min",
        "phase":        "pre-peak",
        "ideal":        0.0,
        "direction":    -1,
    },

    # -- Peak ---------------------------------------------------------------
    "peak_error_pct": {
        "code_name":    "peak_error_pct",
        "display_name": "Peak discharge error",
        "symbol":       "",
        "units":        "%",
        "phase":        "peak",
        "ideal":        0.0,
        "direction":    -1,
    },
    "peak_timing_error_hr": {
        "code_name":    "peak_timing_error_hr",
        "display_name": "Peak timing error",
        "symbol":       "",
        "units":        "hr",
        "phase":        "peak",
        "ideal":        0.0,
        "direction":    -1,
    },

    # -- Volume -------------------------------------------------------------
    "pbias_pct": {
        "code_name":    "pbias_pct",
        "display_name": "Volume bias (PBIAS)",
        "symbol":       "\u03b2",                  # β
        "units":        "%",
        "phase":        "volume",
        "ideal":        0.0,
        "direction":    -1,
    },
    "duration_above_thresh_error_min": {
        "code_name":    "duration_above_thresh_error_min",
        "display_name": "Duration above threshold error",
        "symbol":       "",
        "units":        "min",
        "phase":        "volume",
        "ideal":        0.0,
        "direction":    -1,
    },

    # -- Recession ----------------------------------------------------------
    "recession_rate_ratio": {
        "code_name":    "recession_rate_ratio",
        "display_name": "Recession rate ratio",
        "symbol":       "",
        "units":        "",
        "phase":        "recession",
        "ideal":        1.0,
        "direction":    -1,
    },

    # -- Summary ------------------------------------------------------------
    "kge": {
        "code_name":    "kge",
        "display_name": "Kling-Gupta Efficiency",
        "symbol":       "KGE",
        "units":        "",
        "phase":        "summary",
        "ideal":        1.0,
        "direction":    +1,   # higher is better
    },
    "kge_r": {
        "code_name":    "kge_r",
        "display_name": "KGE correlation component",
        "symbol":       "r",
        "units":        "",
        "phase":        "summary",
        "ideal":        1.0,
        "direction":    +1,
    },
    "kge_alpha": {
        "code_name":    "kge_alpha",
        "display_name": "KGE variability ratio",
        "symbol":       "\u03b1",                  # α
        "units":        "",
        "phase":        "summary",
        "ideal":        1.0,
        "direction":    +1,
    },
    "kge_beta": {
        "code_name":    "kge_beta",
        "display_name": "KGE bias ratio",
        "symbol":       "\u03b2",                  # β
        "units":        "",
        "phase":        "summary",
        "ideal":        1.0,
        "direction":    +1,
    },
    "nse": {
        "code_name":    "nse",
        "display_name": "Nash-Sutcliffe Efficiency",
        "symbol":       "NSE",
        "units":        "",
        "phase":        "summary",
        "ideal":        1.0,
        "direction":    +1,
    },
}
