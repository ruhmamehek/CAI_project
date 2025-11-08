"""Detect company mentions and years in queries."""

import re
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class MetadataDetector:
    """Detect company tickers, years, and filing types from text queries."""
    
    def __init__(self):
        self.company_names = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "netflix": "NFLX",
            "jpmorgan": "JPM",
            "jpm": "JPM",
            "bank of america": "BAC",
            "goldman sachs": "GS",
            "visa": "V",
            "mastercard": "MA",
            "coca cola": "KO",
            "pepsi": "PEP",
            "walmart": "WMT",
            "disney": "DIS",
            "intel": "INTC",
            "amd": "AMD",
            "ibm": "IBM",
            "oracle": "ORCL",
            "salesforce": "CRM",
            "adobe": "ADBE",
            "paypal": "PYPL",
            "uber": "UBER",
            "airbnb": "ABNB"
        }
        
        self.ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')
        self.year_pattern = re.compile(r'\b(19\d{2}|20\d{2})\b')
        self.filing_type_pattern = re.compile(r'\b(10-K|10-Q|8-K|10-KT)\b', re.IGNORECASE)
    
    def detect_from_query(
        self, 
        query: str, 
        available_tickers: List[str] = None,
        available_years: List[int] = None
    ) -> Optional[Dict]:
        """
        Detect metadata (company, year, filing type) from query text.
        
        Args:
            query: User query text
            available_tickers: List of tickers available in the index (optional)
            available_years: List of years available in the index (optional)
            
        Returns:
            Dict with detected metadata or None if nothing detected
        """
        detected = {}
        
        query_lower = query.lower()
        query_upper = query.upper()
        
        detected_ticker = None
        
        for company_name, ticker in self.company_names.items():
            if company_name in query_lower:
                detected_ticker = ticker
                break
        
        if not detected_ticker:
            ticker_matches = self.ticker_pattern.findall(query_upper)
            for ticker in ticker_matches:
                if len(ticker) >= 2 and len(ticker) <= 5:
                    if available_tickers is None or ticker in available_tickers:
                        detected_ticker = ticker
                        break
        
        if detected_ticker:
            detected["ticker"] = detected_ticker
        
        year_matches = self.year_pattern.findall(query)
        if year_matches:
            year = int(year_matches[0])
            if available_years is None or year in available_years:
                detected["year"] = year
        
        filing_type_matches = self.filing_type_pattern.findall(query)
        if filing_type_matches:
            detected["filing_type"] = filing_type_matches[0].upper()
        
        if detected:
            logger.info(f"Detected metadata from query: {detected}")
            return detected
        
        return None
    
    def add_company_mapping(self, company_name: str, ticker: str):
        """Add a custom company name to ticker mapping."""
        self.company_names[company_name.lower()] = ticker.upper()

