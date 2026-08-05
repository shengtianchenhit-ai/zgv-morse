"""Stable stage registry for the first-paper numerical workflows."""

from types import MappingProxyType

from . import convergence, critical_points, green, isotropic, scaling, sensitivity, silicon
from .common import OUTPUT_FILES


STAGES = MappingProxyType(
    {
        "isotropic": isotropic.run,
        "sensitivity": sensitivity.run,
        "critical_points": critical_points.run,
        "scaling": scaling.run,
        "green": green.run,
        "convergence": convergence.run,
        "silicon": silicon.run,
    }
)
OUTPUTS = OUTPUT_FILES


__all__ = ["OUTPUTS", "STAGES"]
