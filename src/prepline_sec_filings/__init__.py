"""Subset of Unstructured's SEC pipeline utilities vendored locally.

Source: https://github.com/Unstructured-IO/pipeline-sec-filings
Commit: babc6430e36c76c21a9c963ceda4867c9b5d28a9
Licensed under the Apache License, Version 2.0.
"""

from .sections import (
    SECSection,
    SECTIONS_10K,
    SECTIONS_10Q,
    SECTIONS_S1,
    section_string_to_enum,
    validate_section_names,
    ALL_SECTIONS,
)
from .sec_document import SECDocument, REPORT_TYPES, VALID_FILING_TYPES, is_section_elem

__all__ = [
    "SECSection",
    "SECTIONS_10K",
    "SECTIONS_10Q",
    "SECTIONS_S1",
    "section_string_to_enum",
    "validate_section_names",
    "ALL_SECTIONS",
    "SECDocument",
    "REPORT_TYPES",
    "VALID_FILING_TYPES",
    "is_section_elem",
]


