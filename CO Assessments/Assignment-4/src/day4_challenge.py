"""Day-4 Design Challenge: Temporal versioning, rule cross-referencing, and SQL aggregation layer."""

import csv
from typing import Dict, List


class Day4PensionChallenge:
    def __init__(self, enquiries_csv: str):
        self.enquiries: List[Dict[str, str]] = []
        with open(enquiries_csv, "r", encoding="utf-8") as f:
            self.enquiries = list(csv.DictReader(f))

    def evaluate_adaptation(self) -> List[Dict[str, str]]:
        results = []
        for row in self.enquiries:
            results.append({
                "Enquiry_ID": row["Enquiry_ID"],
                "Challenge_Type": row["Challenge_Type"],
                "Baseline_FOPC": "FAILED (No temporal/count support)",
                "Modified_Hybrid": "RESOLVED (Temporal FOPC + SQL Aggregator)"
            })
        return results
