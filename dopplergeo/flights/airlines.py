"""Airline names from ADS-B callsigns.

An airline callsign is the operator's 3-letter ICAO designator followed by the
flight number (``EZY32NC`` = easyJet). The ADS-B data carries no operator name,
so common carriers are listed here; unknown designators and registration-style
callsigns (``GABCD``) give ``None``.
"""

from __future__ import annotations

import re

AIRLINES: dict[str, str] = {
    # UK and Ireland
    "BAW": "British Airways", "SHT": "British Airways (Shuttle)", "CFE": "BA CityFlyer", "EFW": "BA Euroflyer",
    "VIR": "Virgin Atlantic", "EZY": "easyJet", "EJU": "easyJet Europe", "EZS": "easyJet Switzerland",
    "TOM": "TUI Airways", "EXS": "Jet2", "LOG": "Loganair", "EIN": "Aer Lingus", "EUK": "Aer Lingus UK",
    "EAI": "Emerald Airlines", "RYR": "Ryanair", "RUK": "Ryanair UK", "RYS": "Buzz", "MAY": "Malta Air",
    "LDM": "Lauda Europe", "WUK": "Wizz Air UK", "BEE": "Flybe", "TAW": "Titan Airways", "BCY": "CityJet",
    "NJE": "NetJets Europe", "NJU": "NetJets",
    # Europe
    "WZZ": "Wizz Air", "WMT": "Wizz Air Malta", "KLM": "KLM", "AFR": "Air France", "DLH": "Lufthansa",
    "CLH": "Lufthansa CityLine", "EWG": "Eurowings", "SWR": "Swiss", "AUA": "Austrian", "BEL": "Brussels Airlines",
    "SAS": "SAS", "FIN": "Finnair", "ICE": "Icelandair", "NOZ": "Norwegian", "NSZ": "Norwegian",
    "NAX": "Norwegian", "IBE": "Iberia", "IBS": "Iberia Express", "VLG": "Vueling", "AEA": "Air Europa",
    "TAP": "TAP Air Portugal", "ITY": "ITA Airways", "LOT": "LOT Polish Airlines", "TRA": "Transavia",
    "TVF": "Transavia France", "CND": "Corendon", "THY": "Turkish Airlines", "PGT": "Pegasus",
    "SXS": "SunExpress", "AEE": "Aegean", "CTN": "Croatia Airlines", "TUI": "TUIfly", "TFL": "TUI fly Netherlands",
    "JAF": "TUI fly Belgium", "BTI": "airBaltic", "ELY": "El Al", "CYP": "Cyprus Airways",
    # Middle East, Africa, Asia, Americas
    "UAE": "Emirates", "QTR": "Qatar Airways", "ETD": "Etihad", "SVA": "Saudia", "GFA": "Gulf Air",
    "OMA": "Oman Air", "RJA": "Royal Jordanian", "MSR": "EgyptAir", "RAM": "Royal Air Maroc",
    "ETH": "Ethiopian Airlines", "PIA": "Pakistan International", "AIC": "Air India", "SIA": "Singapore Airlines",
    "CPA": "Cathay Pacific", "CCA": "Air China", "CES": "China Eastern", "CSN": "China Southern",
    "HVN": "Vietnam Airlines", "MAS": "Malaysia Airlines", "THA": "Thai Airways", "JAL": "Japan Airlines",
    "ANA": "All Nippon Airways", "KAL": "Korean Air", "AAL": "American Airlines", "UAL": "United Airlines",
    "DAL": "Delta Air Lines", "ACA": "Air Canada", "TSC": "Air Transat", "WJA": "WestJet",
    # Cargo
    "FDX": "FedEx", "UPS": "UPS", "DHK": "DHL Air UK", "BCS": "DHL (EAT Leipzig)", "CLX": "Cargolux",
    "GTI": "Atlas Air", "ABR": "ASL Airlines Ireland",
}

_CALLSIGN = re.compile(r"^([A-Z]{3})\d")


def airline_name(callsign: str | None) -> str | None:
    """The operator for an airline-style callsign (3 letters then a digit), else None."""
    if not callsign:
        return None
    m = _CALLSIGN.match(callsign.strip().upper())
    return AIRLINES.get(m.group(1)) if m else None
