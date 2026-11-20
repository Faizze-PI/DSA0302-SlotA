"""Grammar-based surface realiser, referring expression generation, and BLEU evaluation."""

from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class SurfaceRealiser:
    def realise_bulletin(self, plan: Dict[str, object], audience: str = "citizen") -> str:
        sel = plan["selected"]
        loc = sel["Location"]
        inc_type = str(sel["Type"]).capitalize()
        camp = sel["Camp_Name"]
        free = sel["Camp_Free"]
        if audience == "field_officer":
            return f"[PRIORITY] {inc_type} at {loc}. Road: {sel['Road_Status']}. Shelter {camp} capacity remaining: {free} beds."
        return f"{inc_type} is reported in {loc}. The main road is {str(sel['Road_Status']).lower()}, and the nearest relief camp at {camp} is open with {free} spaces."

    def compute_bleu(self, hypothesis: str, reference: str) -> float:
        smooth = SmoothingFunction().method1
        return float(sentence_bleu([reference.lower().split()], hypothesis.lower().split(), smoothing_function=smooth))
