"""SEC document helpers vendored from Unstructured-IO/pipeline-sec-filings."""

from functools import partial
import re
from typing import List, Optional, Iterable, Iterator, Any, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN
from collections import defaultdict

from unstructured.cleaners.core import clean
from unstructured.documents.elements import Text, ListItem, NarrativeText, Title, Element
try:
    from unstructured.documents.elements import Table  # type: ignore
except ImportError:  # Older unstructured versions may not expose Table
    Table = None  # type: ignore
from unstructured.documents.html import HTMLDocument
from unstructured.nlp.partition import is_possible_title

from .sections import SECSection

VALID_FILING_TYPES = [
    "10-K",
    "10-Q",
    "S-1",
    "10-K/A",
    "10-Q/A",
    "S-1/A",
]
REPORT_TYPES = ["10-K", "10-Q", "10-K/A", "10-Q/A"]
S1_TYPES = ["S-1", "S-1/A"]

ITEM_TITLE_RE = re.compile(r"(?i)item \d{1,3}(?:[a-z]|\([a-z]\))?(?:\.)?(?::)?")

clean_sec_text = partial(clean, extra_whitespace=True, dashes=True, trailing_punctuation=True)


def _raise_for_invalid_filing_type(filing_type: Optional[str]):
    if not filing_type:
        raise ValueError("Filing type is empty.")
    elif filing_type not in VALID_FILING_TYPES:
        raise ValueError(f"Filing type was {filing_type}. Expected: {VALID_FILING_TYPES}")


class SECDocument(HTMLDocument):
    filing_type = None

    def _filter_table_of_contents(self, elements: List[Text]) -> List[Text]:
        if self.filing_type in REPORT_TYPES:
            start, end = None, None
            for i, element in enumerate(elements):
                if bool(re.match(r"(?i)part i\b", clean_sec_text(element.text))):
                    if start is None:
                        start = i
                    else:
                        end = i - 1
                        filtered_elements = elements[start:end]
                        return filtered_elements
        elif self.filing_type in S1_TYPES:
            title_indices = defaultdict(list)
            for i, element in enumerate(elements):
                clean_title_text = clean_sec_text(element.text).lower()
                title_indices[clean_title_text].append(i)
            duplicate_title_indices = {k: v for k, v in title_indices.items() if len(v) > 1}
            for title, indices in duplicate_title_indices.items():
                if "prospectus" in title and len(indices) == 2:
                    start = indices[0]
                    end = indices[1] - 1
                    filtered_elements = elements[start:end]
                    return filtered_elements
        return []

    def get_table_of_contents(self) -> HTMLDocument:
        out_cls = self.__class__
        _raise_for_invalid_filing_type(self.filing_type)
        title_locs = to_sklearn_format(self.elements)
        if len(title_locs) == 0:
            return out_cls.from_elements([])
        res = DBSCAN(eps=6.0).fit_predict(title_locs)
        for i in range(res.max() + 1):
            idxs = cluster_num_to_indices(i, title_locs, res)
            cluster_elements: List[Text] = [self.elements[i] for i in idxs]
            if any(
                [
                    is_risk_title(el.text, self.filing_type)
                    for el in cluster_elements
                    if isinstance(el, Title)
                ]
            ) and any([is_toc_title(el.text) for el in cluster_elements if isinstance(el, Title)]):
                return out_cls.from_elements(self._filter_table_of_contents(cluster_elements))
        return out_cls.from_elements(self._filter_table_of_contents(self.elements))

    def get_section_narrative_no_toc(self, section: SECSection) -> List[NarrativeText]:
        _raise_for_invalid_filing_type(self.filing_type)
        section_elements: List[NarrativeText] = list()
        in_section = False
        for element in self.elements:
            is_title = is_possible_title(element.text)
            if in_section:
                if is_title and is_item_title(element.text, self.filing_type):
                    if section_elements:
                        return section_elements
                    else:
                        in_section = False
                elif isinstance(element, NarrativeText) or isinstance(element, ListItem):
                    section_elements.append(element)

            if is_title and is_section_elem(section, element, self.filing_type):
                in_section = True

        return section_elements

    def _get_toc_sections(self, section: SECSection, toc: HTMLDocument) -> Tuple[Text, Text]:
        section_toc = first(
            el for el in toc.elements if is_section_elem(section, el, self.filing_type)
        )
        if section_toc is None:
            return (None, None)

        after_section_toc = toc.after_element(section_toc)
        next_section_toc = first(
            el
            for el in after_section_toc.elements
            if not is_section_elem(section, el, self.filing_type)
        )
        if next_section_toc is None:
            return (section_toc, None)
        return (section_toc, next_section_toc)

    def get_section_narrative(self, section: SECSection) -> List[NarrativeText]:
        _raise_for_invalid_filing_type(self.filing_type)
        toc = self.get_table_of_contents()
        if not toc.pages:
            return self.get_section_narrative_no_toc(section)

        section_toc, next_section_toc = self._get_toc_sections(section, toc)
        if section_toc is None:
            return []

        doc_after_section_toc = self.after_element(
            next_section_toc if next_section_toc else section_toc
        )
        section_start_element = get_element_by_title(
            reversed(doc_after_section_toc.elements), section_toc.text, self.filing_type
        )
        if section_start_element is None:
            return []
        doc_after_section_heading = self.after_element(section_start_element)

        if self._is_last_section_in_report(section, toc) or next_section_toc is None:
            return get_narrative_texts(doc_after_section_heading, up_to_next_title=True)

        section_end_element = get_element_by_title(
            doc_after_section_heading.elements, next_section_toc.text, self.filing_type
        )

        if section_end_element is None:
            return get_narrative_texts(doc_after_section_heading, up_to_next_title=True)

        return get_narrative_texts(doc_after_section_heading.before_element(section_end_element))

    def get_risk_narrative(self) -> List[NarrativeText]:
        return self.get_section_narrative(SECSection.RISK_FACTORS)

    def doc_after_cleaners(
        self, skip_headers_and_footers=False, skip_table_text=False, inplace=False
    ) -> HTMLDocument:
        new_doc = super().doc_after_cleaners(skip_headers_and_footers, skip_table_text, inplace)
        if not inplace:
            new_doc.filing_type = self.filing_type
        return new_doc

    def _read_xml(self, content):
        super()._read_xml(content)
        type_tag = self.document_tree.find(".//type")
        if type_tag is not None:
            self.filing_type = type_tag.text.strip()
        return self.document_tree

    def _is_last_section_in_report(self, section: SECSection, toc: HTMLDocument) -> bool:
        if self.filing_type in ["10-K", "10-K/A"]:
            if section == SECSection.FORM_SUMMARY:
                return True
            if section == SECSection.EXHIBITS:
                form_summary_section = first(
                    el
                    for el in toc.elements
                    if is_section_elem(SECSection.FORM_SUMMARY, el, self.filing_type)
                )
                if form_summary_section is None:
                    return True
        if self.filing_type in ["10-Q", "10-Q/A"]:
            if section == SECSection.EXHIBITS:
                return True
        return False


def get_narrative_texts(doc: HTMLDocument, up_to_next_title: Optional[bool] = False) -> List[Text]:
    if up_to_next_title:
        narrative_texts = []
        for el in doc.elements:
            if isinstance(el, NarrativeText) or isinstance(el, ListItem):
                narrative_texts.append(el)
            else:
                break
        return narrative_texts
    else:
        return [
            el for el in doc.elements if isinstance(el, NarrativeText) or isinstance(el, ListItem)
        ]


def is_section_elem(section: SECSection, elem: Text, filing_type: Optional[str]) -> bool:
    _raise_for_invalid_filing_type(filing_type)
    if section is SECSection.RISK_FACTORS:
        return is_risk_title(elem.text, filing_type=filing_type)
    else:

        def _is_matching_section_pattern(text):
            return bool(re.search(section.pattern, clean_sec_text(text, lowercase=True)))

        if filing_type in REPORT_TYPES:
            return _is_matching_section_pattern(remove_item_from_section_text(elem.text))
        else:
            return _is_matching_section_pattern(elem.text)


def is_item_title(title: str, filing_type: Optional[str]) -> bool:
    if filing_type in REPORT_TYPES:
        return is_10k_item_title(title)
    elif filing_type in S1_TYPES:
        return is_s1_section_title(title)
    return False


def is_risk_title(title: str, filing_type: Optional[str]) -> bool:
    if filing_type in REPORT_TYPES:
        return ("1a" in title.lower() or "risk factors" in title.lower()) and not (
            "summary" in title.lower()
        )
    elif filing_type in S1_TYPES:
        return title.strip().lower() == "risk factors"
    return False


def is_toc_title(title: str) -> bool:
    clean_title = clean_sec_text(title, lowercase=True)
    return (clean_title == "table of contents") or (clean_title == "index")


def is_10k_item_title(title: str) -> bool:
    return ITEM_TITLE_RE.match(clean_sec_text(title, lowercase=True)) is not None


def is_s1_section_title(title: str) -> bool:
    return title.strip().isupper()


def to_sklearn_format(elements: List[Element]) -> npt.NDArray[np.float32]:
    is_title: npt.NDArray[np.bool_] = np.array(
        [is_possible_title(el.text) for el in elements][: len(elements)], dtype=bool
    )
    title_locs = np.arange(len(is_title)).astype(np.float32)[is_title].reshape(-1, 1)
    return title_locs


def cluster_num_to_indices(
    num: int, elem_idxs: npt.NDArray[np.float32], res: npt.NDArray[np.int_]
) -> List[int]:
    idxs = elem_idxs[res == num].astype(int).flatten().tolist()
    return idxs


def first(it: Iterable) -> Any:
    try:
        out = next(iter(it))
    except StopIteration:
        out = None
    return out


def match_s1_toc_title_to_section(text: str, title: str) -> bool:
    return text == title


def match_10k_toc_title_to_section(text: str, title: str) -> bool:
    if re.match(ITEM_TITLE_RE, title):
        return text.startswith(title)
    else:
        text = remove_item_from_section_text(text)
        return text.startswith(title)


def remove_item_from_section_text(text: str) -> str:
    return re.sub(ITEM_TITLE_RE, "", text).strip()


def get_element_by_title(
    elements: Iterator[Element],
    title: str,
    filing_type: Optional[str],
) -> Optional[Element]:
    _raise_for_invalid_filing_type(filing_type)
    if filing_type in REPORT_TYPES:
        match = match_10k_toc_title_to_section
    elif filing_type in S1_TYPES:
        match = match_s1_toc_title_to_section
    return first(
        el
        for el in elements
        if isinstance(el, Text)
        and el.text is not None
        and match(clean_sec_text(el.text), clean_sec_text(title))
    )

