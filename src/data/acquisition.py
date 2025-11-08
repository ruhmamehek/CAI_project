"""Data acquisition for SEC filings and FOMC texts."""

import os
from pathlib import Path
from typing import List, Optional
from sec_edgar_downloader import Downloader
import logging

logger = logging.getLogger(__name__)


class SECFilingDownloader:
    """Download SEC filings (10-K, 10-Q) for specified companies."""
    
    def __init__(self, output_dir: str, email: str):
        """
        Args:
            output_dir: Directory to save filings
            email: Email for SEC EDGAR 
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = Downloader(output_dir, email)
        
    def download_filings(
        self,
        tickers: List[str],
        filing_types: List[str] = ["10-K", "10-Q"],
        years: Optional[List[int]] = None
    ):
        """
        Download filings for given tickers.
        
        Args:
            tickers: List of company ticker symbols
            filing_types: Types of filings to download
            years: Optional list of years to filter
        """
        for ticker in tickers:
            logger.info(f"Downloading filings for {ticker}")
            for filing_type in filing_types:
                try:
                    if years:
                        for year in years:
                            self.downloader.get(
                                filing_type, 
                                ticker, 
                                after=f"{year}-01-01", 
                                before=f"{year}-12-31"
                            )
                    else:
                        self.downloader.get(filing_type, ticker)
                    logger.info(f"Downloaded {filing_type} for {ticker}")
                except Exception as e:
                    logger.error(f"Error downloading {filing_type} for {ticker}: {e}")


class FOMCDownloader:
    """Download FOMC statements and minutes."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_fomc_texts(self, years: List[int]):
        """
        # TODO: Implement FOMC text download
        """
        logger.info(f"Downloading FOMC texts for years {years}")
        logger.warning("FOMC download not yet implemented - manual download may be required")
