"""High-level SEC filing preprocessor using Unstructured's SEC pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import re

from bs4 import BeautifulSoup

from src.prepline_sec_filings.sec_document import SECDocument, is_section_elem, Table
from src.prepline_sec_filings.sections import (
    ALL_SECTIONS,
    SECSection,
    SECTIONS_10K,
    SECTIONS_10Q,
    SECTIONS_S1,
)
from unstructured.cleaners.core import clean
from unstructured.documents.elements import Element, NarrativeText, Title, ListItem
from unstructured.nlp.partition import is_possible_title


@dataclass
class NarrativeSection:
    section: str
    title: str
    text: str
    doc_id: str


@dataclass
class TableSection:
    section: Optional[str]
    section_title: Optional[str]
    text: str
    html: Optional[str]
    doc_id: str


@dataclass
class ItemSection:
    item: str
    text: str
    doc_id: str


class SECFilingPreprocessor:
    """Parse SEC filings into narrative sections and table payloads."""

    def __init__(self, include_all_sections: bool = True):
        self.include_all_sections = include_all_sections

    ITEM_PATTERN = re.compile(r"item\s+(1a|1b|7a|7|8|9|15)", re.IGNORECASE)

    def parse_file(
        self, *, file_path: str, doc_id: str
    ) -> Tuple[List[NarrativeSection], List[TableSection], List[ItemSection]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
            raw_text = fp.read()

        sec_doc = SECDocument.from_string(raw_text)
        if not sec_doc.filing_type:
            raise ValueError(f"Unable to determine filing type for {file_path}")

        cleaned_doc = sec_doc.doc_after_cleaners(
            skip_headers_and_footers=True, skip_table_text=False, inplace=False
        )
        narrative_doc = sec_doc.doc_after_cleaners(
            skip_headers_and_footers=True, skip_table_text=True, inplace=False
        )

        narrative_sections = self._extract_narratives(narrative_doc, doc_id=doc_id)
        tables = self._extract_tables(cleaned_doc, raw_text=raw_text, doc_id=doc_id)
        item_sections = self._extract_item_sections(
            raw_text=raw_text, filing_type=sec_doc.filing_type, doc_id=doc_id
        )

        return narrative_sections, tables, item_sections

    # ------------------------------------------------------------------
    def _extract_narratives(self, doc: SECDocument, *, doc_id: str) -> List[NarrativeSection]:
        filing_type = doc.filing_type
        section_enums = self._sections_for_filing(filing_type)

        narratives: List[NarrativeSection] = []
        for section_enum in section_enums:
            elements = doc.get_section_narrative(section_enum)
            if not elements:
                continue

            text_blocks = [
                el.text.strip()
                for el in elements
                if isinstance(el, (NarrativeText, ListItem)) and el.text.strip()
            ]
            if not text_blocks:
                continue

            joined = "\n\n".join(text_blocks)
            narratives.append(
                NarrativeSection(
                    section=section_enum.name,
                    title=self._normalize_title(section_enum),
                    text=joined,
                    doc_id=doc_id,
                )
            )
        return narratives

    def _extract_tables(
        self, doc: SECDocument, *, raw_text: str, doc_id: str
    ) -> List[TableSection]:
        tables: List[TableSection] = []
        if Table is None:
            tables.extend(self._fallback_tables(raw_text=raw_text, doc_id=doc_id))
            return tables
        current_section: Optional[str] = None
        current_section_title: Optional[str] = None

        element_iter = doc.elements if hasattr(doc, "elements") else []
        for element in element_iter:
            if isinstance(element, Title):
                current_section, current_section_title = self._match_section(element, doc.filing_type)
            elif isinstance(element, Table):
                html = None
                metadata = getattr(element, "metadata", None)
                if metadata is not None:
                    metadata_dict = metadata.to_dict()
                    html = metadata_dict.get("text_as_html") or metadata_dict.get("text_as_html_")
                tables.append(
                    TableSection(
                        section=current_section,
                        section_title=current_section_title,
                        text=element.text.strip(),
                        html=html,
                        doc_id=doc_id,
                    )
                )
        return tables

    def _fallback_tables(self, *, raw_text: str, doc_id: str) -> List[TableSection]:
        """Extract tables by traversing the raw HTML when Unstructured Table elements are unavailable."""
        tables: List[TableSection] = []
        try:
            soup = BeautifulSoup(raw_text, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(raw_text, "html.parser")
            except Exception:
                return tables

        for table in soup.find_all("table"):
            table_text = table.get_text(separator=" ", strip=True)
            if not table_text:
                continue
            section_title = None
            section_name = None
            heading = table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
            if heading:
                section_title = heading.get_text(separator=" ", strip=True)
                match = self.ITEM_PATTERN.search(section_title)
                if match:
                    section_name = match.group(0).strip().upper()

            tables.append(
                TableSection(
                    section=section_name,
                    section_title=section_title,
                    text=table_text,
                    html=str(table),
                    doc_id=doc_id,
                )
            )
        return tables

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_title(section: SECSection) -> str:
        if isinstance(section.value, str):
            return section.value
        return section.name.replace("_", " ").title()

    def _sections_for_filing(self, filing_type: str) -> Iterable[SECSection]:
        if not self.include_all_sections:
            return []
        if filing_type.startswith("10-K"):
            return SECTIONS_10K
        if filing_type.startswith("10-Q"):
            return SECTIONS_10Q
        if filing_type.startswith("S-1"):
            return SECTIONS_S1
        return []

    def _match_section(
        self, title_element: Title, filing_type: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        title_text = clean(title_element.text or "", extra_whitespace=True).strip()
        if not title_text:
            return None, None

        matched_section = None
        if filing_type:
            for section in self._sections_for_filing(filing_type):
                if self._title_matches_section(section, title_element, filing_type):
                    matched_section = section.name
                    break

        return matched_section, title_text

    @staticmethod
    @staticmethod
    def _title_matches_section(
        section: SECSection, title_element: Title, filing_type: Optional[str]
    ) -> bool:
        try:
            return is_section_elem(section, title_element, filing_type)
        except Exception:
            return False

    def _extract_item_sections(
        self, *, raw_text: str, filing_type: Optional[str], doc_id: str
    ) -> List[ItemSection]:
        """Extract canonical 10-K item sections using regex boundaries."""
        if filing_type not in ("10-K", "10-K/A"):
            return []

        matches = list(self.ITEM_PATTERN.finditer(raw_text))
        if not matches:
            return []

        item_sections: List[ItemSection] = []
        for idx, match in enumerate(matches):
            item_label = match.group(0)
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            block = raw_text[start:end]
            cleaned_block = self._clean_html_block(block)
            if not cleaned_block:
                continue
            item_sections.append(
                ItemSection(
                    item=item_label.upper(),
                    text=cleaned_block,
                    doc_id=doc_id,
                )
            )
        return item_sections

    @staticmethod
    def _clean_html_block(block: str) -> str:
        try:
            soup = BeautifulSoup(block, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(block, "html.parser")
            except Exception:
                return clean(block, extra_whitespace=True)
        for script in soup(["script", "style"]):
            script.decompose()
        return clean(soup.get_text(separator=" "), extra_whitespace=True)

