"""Finite-state and 5-slot frame-based dialogue managers for disaster shelter enquiries."""

from typing import Dict, List, Tuple


class FrameDialogueManager:
    def __init__(self):
        self.slots: Dict[str, str] = {
            "Location": "",
            "Incident_Type": "",
            "Need_Type": "",
            "Caller_Count": "",
            "Urgency": ""
        }

    def reset(self):
        for k in self.slots:
            self.slots[k] = ""

    def trace_dialogue(self, turns: List[Tuple[str, Dict[str, str]]]) -> List[Dict[str, str]]:
        self.reset()
        history = []
        for utt, updates in turns:
            for k, v in updates.items():
                if k in self.slots:
                    self.slots[k] = v
            history.append(dict(self.slots))
        return history
