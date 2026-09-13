"""Core package for the interactive 2D Gaussian optics simulator."""

from .model import FrequencyChannel, ObjectLink, OpticalElement, OpticalSystem, default_band6_system
from .physics import BeamSegment, BeamTrace, TraceResult, trace_all_beams, trace_system

__all__ = [
    "BeamSegment",
    "BeamTrace",
    "FrequencyChannel",
    "ObjectLink",
    "OpticalElement",
    "OpticalSystem",
    "TraceResult",
    "default_band6_system",
    "trace_system",
    "trace_all_beams",
]
