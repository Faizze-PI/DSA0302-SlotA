"""Three-stage NLG pipeline: content selection, discourse planning, and sentence planning."""

import csv
from typing import Dict, List


class NLGPlanner:
    def __init__(self, incident_csv: str):
        self.incidents: List[Dict[str, str]] = []
        with open(incident_csv, "r", encoding="utf-8") as f:
            self.incidents = list(csv.DictReader(f))

    def plan_bulletin(self, idx: int = 0) -> Dict[str, object]:
        inc = self.incidents[idx]
        selected = {
            "Location": inc["Location"],
            "Type": inc["Type"],
            "Road_Status": inc["Road_Status"],
            "Camp_Name": inc["Camp_Name"],
            "Camp_Free": int(inc["Camp_Capacity"]) - int(inc["Camp_Occupied"])
        }
        discourse_plan = [
            ("Nucleus", f"REPORT({selected['Type']}, {selected['Location']})"),
            ("Satellite-Result", f"ROAD_STATUS({selected['Road_Status']})"),
            ("Satellite-Solution", f"SHELTER({selected['Camp_Name']}, {selected['Camp_Free']})")
        ]
        return {"selected": selected, "discourse_plan": discourse_plan, "reference": inc["Reference_Bulletin"]}
