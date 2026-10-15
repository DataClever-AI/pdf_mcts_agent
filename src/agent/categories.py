"""
Manual category definitions — domain context for prompt injection.

Each category provides:
  label:          Display name shown in the UI dropdown
  guard_context:  Topics that are ALWAYS on-topic for this domain
  domain_hint:    Terminology hint for query expansion prompt
  expert_role:    Expert persona for answer generation prompt
  pass_examples:  Guard PASS examples specific to this domain

Synonym/terminology equivalences are NOT hardcoded here.
The answer VLM is instructed to use SECTION TITLES from the retrieved evidence
as the authoritative vocabulary of the specific document being queried.
This approach is document-adaptive and unbiased — each manual defines its own
terminology through its own section structure.
"""
from __future__ import annotations

from typing import TypedDict


class CategoryConfig(TypedDict):
    """Configuration for a manual category."""

    label: str
    guard_context: str
    domain_hint: str
    expert_role: str
    pass_examples: list[str]


CATEGORIES: dict[str, CategoryConfig] = {
    "medical": {
        "label": "🏥 Medical / Clinical Equipment",
        "guard_context": (
            "medical devices, patient monitoring, clinical procedures, "
            "hemodynamic parameters, physiological alarms, drug delivery, "
            "diagnostic equipment, patient data, clinical decisions, bedside "
            "workflows, regulatory compliance (FDA/CE), biocompatibility"
        ),
        "domain_hint": (
            "medical device manual. Use clinical terminology: hemodynamic, "
            "physiological, calibration, patient safety, waveform, transducer, "
            "cardiac output, alarm thresholds, in vitro, in vivo"
        ),
        "expert_role": "medical device expert and clinical technician",
        "pass_examples": [
            "What clinical decisions can this monitoring index support?",
            "How does patient data get exported or downloaded securely?",
            "What are the alarm thresholds for this physiological parameter?",
            "How do you perform in vitro calibration before use?",
        ],
    },
    "automotive": {
        "label": "🚗 Automotive / Vehicle",
        "guard_context": (
            "vehicle repair, engine, transmission, brakes, electrical systems, "
            "OBD diagnostic codes, torque specifications, maintenance intervals, "
            "body panels, fuel system, cooling system, suspension, drivetrain"
        ),
        "domain_hint": (
            "vehicle service manual. Use automotive terminology: OBD, torque "
            "(Nm/ft-lb), SAE, ECU, fuel injection, timing belt, coolant, TSB"
        ),
        "expert_role": "automotive technician and service advisor",
        "pass_examples": [
            "What is the torque specification for the cylinder head bolts?",
            "How do you clear OBD fault codes after repair?",
            "When should transmission fluid be changed?",
        ],
    },
    "industrial": {
        "label": "⚙️ Industrial / Machinery",
        "guard_context": (
            "industrial machinery, manufacturing equipment, PLC programming, "
            "SCADA systems, safety relays, rated load, maintenance schedules, "
            "pneumatic/hydraulic systems, conveyor, CNC, robot arms"
        ),
        "domain_hint": (
            "industrial equipment manual. Use industrial terminology: PLC, "
            "SCADA, interlock, rated capacity, preventive maintenance, "
            "safety relay, pneumatic, hydraulic, torque, IP rating"
        ),
        "expert_role": "industrial equipment technician and maintenance engineer",
        "pass_examples": [
            "How do you reset the PLC after a safety interlock trip?",
            "What is the rated load capacity for this conveyor section?",
            "How should hydraulic fluid levels be checked?",
        ],
    },
    "technology": {
        "label": "💻 Technology / Electronics",
        "guard_context": (
            "software, hardware, networking, firmware, drivers, API, "
            "connectivity, configuration, troubleshooting, specifications, "
            "compatibility, installation, cybersecurity, data backup"
        ),
        "domain_hint": (
            "technology product manual. Use tech terminology: firmware, driver, "
            "API, bandwidth, latency, protocol (TCP/IP, USB, HDMI), encryption, "
            "configuration file, port, interface"
        ),
        "expert_role": "technical support engineer and system administrator",
        "pass_examples": [
            "How do you update the firmware on this device?",
            "What network protocols does this system support?",
            "How do you configure the API authentication settings?",
        ],
    },
    "aviation": {
        "label": "✈️ Aviation / Aerospace",
        "guard_context": (
            "aircraft systems, avionics, navigation, autopilot, engine "
            "maintenance, airworthiness directives, MEL, preflight checks, "
            "weight and balance, emergency procedures, ATC communication"
        ),
        "domain_hint": (
            "aviation manual. Use aviation terminology: METAR, NOTAMs, "
            "airworthiness directive (AD), squawk code, VFR/IFR, "
            "manifold pressure, TBO, annual inspection, squawk"
        ),
        "expert_role": "aviation maintenance technician and flight operations specialist",
        "pass_examples": [
            "What are the preflight inspection requirements for this system?",
            "How do you perform the engine run-up procedure?",
            "What is the time between overhaul (TBO) for this component?",
        ],
    },
    "scientific": {
        "label": "🔬 Scientific / Laboratory",
        "guard_context": (
            "laboratory instruments, reagents, calibration standards, "
            "analytical methods, sample preparation, quality control, "
            "spectral analysis, centrifuge, microscopy, PCR, titration"
        ),
        "domain_hint": (
            "laboratory instrument manual. Use scientific terminology: "
            "reagent, calibration curve, standard deviation, LOD/LOQ, "
            "spectral range, absorbance, analyte, QC sample, SOP"
        ),
        "expert_role": "laboratory scientist and instrument specialist",
        "pass_examples": [
            "How do you perform a multi-point calibration for this instrument?",
            "What reagents are required for this analysis method?",
            "How do you interpret out-of-range QC results?",
        ],
    },
    "construction": {
        "label": "🏗️ Construction / Safety Equipment",
        "guard_context": (
            "construction equipment, safety gear, load ratings, inspection "
            "schedules, PPE requirements, fall protection, scaffolding, "
            "crane operations, harness, OSHA compliance"
        ),
        "domain_hint": (
            "construction/safety equipment manual. Use terminology: "
            "SWL (safe working load), fall arrest, anchorage point, "
            "OSHA, harness, lanyard, inspection interval, load rating"
        ),
        "expert_role": "safety equipment specialist and site manager",
        "pass_examples": [
            "What is the maximum safe working load for this equipment?",
            "How often should this safety harness be inspected?",
            "What are the OSHA requirements for this type of equipment?",
        ],
    },
    "appliances": {
        "label": "🏠 Consumer / Appliances",
        "guard_context": (
            "home appliances, consumer electronics, installation, settings, "
            "energy consumption, maintenance, error codes, warranty, "
            "cleaning procedures, parts replacement"
        ),
        "domain_hint": (
            "consumer product manual. Use plain language with product-specific "
            "terminology: model number, error code, cycle settings, "
            "capacity, wattage, replacement part number"
        ),
        "expert_role": "product support specialist",
        "pass_examples": [
            "What does error code E3 mean on this appliance?",
            "How do you clean the filter on this device?",
            "What is the maximum load capacity for this cycle?",
        ],
    },
    "general": {
        "label": "📄 Other / General Technical",
        "guard_context": (
            "technical documentation, specifications, procedures, "
            "installation, configuration, maintenance, troubleshooting"
        ),
        "domain_hint": (
            "technical manual. Use precise terminology from the document domain"
        ),
        "expert_role": "technical documentation expert",
        "pass_examples": [
            "How do you perform the installation procedure?",
            "What are the technical specifications for this component?",
            "What should be checked during routine maintenance?",
        ],
    },
}

# Ordered list for UI dropdown
CATEGORY_OPTIONS: list[tuple[str, str]] = [
    ("medical", CATEGORIES["medical"]["label"]),
    ("automotive", CATEGORIES["automotive"]["label"]),
    ("industrial", CATEGORIES["industrial"]["label"]),
    ("technology", CATEGORIES["technology"]["label"]),
    ("aviation", CATEGORIES["aviation"]["label"]),
    ("scientific", CATEGORIES["scientific"]["label"]),
    ("construction", CATEGORIES["construction"]["label"]),
    ("appliances", CATEGORIES["appliances"]["label"]),
    ("general", CATEGORIES["general"]["label"]),
]


def get_category(key: str) -> CategoryConfig:
    """Return category config for a key, defaulting to 'general'."""
    return CATEGORIES.get(key, CATEGORIES["general"])
