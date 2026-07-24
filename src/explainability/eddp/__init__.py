"""Explanation-Derived Driving Primitives (EDDP).

The package discovers temporal driving concepts from label-free explanation
signatures.  Frozen M2 primitive labels are deliberately opened only by the
reconciliation stage.
"""

from .schema import EDDP_SCHEMA_VERSION, AnchorRecord, ExplanationAtom

__all__ = ("EDDP_SCHEMA_VERSION", "AnchorRecord", "ExplanationAtom")
