"""Centralized safety rules, prompt/response filtering, and report sanitization/formatting."""

from __future__ import annotations
import re
from typing import Any

class SafetyService:
    """Centralized safety rules, prompt/response filtering, and report sanitization/formatting."""

    # Safety-critical disclaimers that MUST be included in every report.
    DISCLAIMERS = (
        "NOT FOR CLINICAL USE. This is a research/educational portfolio prototype only.\n"
        "Requires qualified expert clinical review.\n"
        "No treatment or medication recommendations are provided.\n"
        "Do not use for emergency triage. In case of emergency, contact local emergency services immediately.\n"
        "Uncertainty warning: Model predictions are probabilistic and subject to significant uncertainty.\n"
        "Model status: Synthetic/demo model. Not clinically validated or certified.\n"
        "Data limitations: The model runs on synthetic or demo data. Clinical interpretation requires standard diagnostic workflow."
    )

    DANGEROUS_CLINICAL_PATTERNS = [
        # Match phrases like "You have [disease]", "The patient has [disease]", "Diagnosis: [disease]"
        r"\b(?:you have|the patient has|diagnosis\s*is\s*|final diagnosis\s*is\s*|clinical diagnosis\s*is\s*|diagnosed with)\s*(?:alzheimer(?:'s|s)?|dementia|mci|parkinson|ftd|lbd|vd|stroke|tumou?r|tumor)\b",
        # Match phrases indicating a definitive diagnosis confirmation
        r"\b(?:confirm|guarantee|prove|certify|definitive|100\s*%)\s*(?:alzheimer(?:'s|s)?|dementia|mci|parkinson|ftd|lbd|vd|stroke|tumou?r|tumor)\b",
    ]

    UNSAFE_MEDICATION_TREATMENT_PATTERNS = [
        r"\b(?:prescribe|prescription|dosage|dose adjustment|take donepezil|take medication|should take|treatment plan|recommend donepezil|recommend memantine|recommend medication)\b",
        r"\b(?:donepezil|memantine|galantamine|rivastigmine|levodopa|carbidopa)\b",
    ]

    EMERGENCY_TRIAGE_PATTERNS = [
        r"\b(?:emergency|triage|chest pain|shortness of breath|sudden weakness|stroke symptoms|act fast|call 911|go to er|urgent care)\b"
    ]

    @classmethod
    def sanitize_report_text(cls, report_text: str) -> str:
        """Sanitize report text to prevent dangerous medical language, direct diagnoses, or medication recommendations."""
        text = report_text or ""
        
        disease_map = {
            "alzheimer": "AD",
            "dementia": "Dementia",
            "mci": "MCI",
            "parkinson": "Parkinson's",
            "ftd": "FTD",
            "lbd": "LBD",
            "vd": "VD",
            "stroke": "Stroke",
            "tumor": "Tumor",
            "tumour": "Tumour",
        }

        # Safer wording template
        def replace_diagnosis(match: re.Match) -> str:
            matched_text = match.group(0).lower()
            found_disease = "X"
            for key, val in disease_map.items():
                if key in matched_text:
                    found_disease = val
                    break
            return (
                f"The demo model assigned higher probability to class {found_disease}. "
                "This is not a diagnosis. Clinical interpretation requires a qualified professional"
            )

        # Replace direct/dangerous phrasing with the safer wording template
        for pattern in cls.DANGEROUS_CLINICAL_PATTERNS:
            text = re.sub(pattern, replace_diagnosis, text, flags=re.IGNORECASE)

        # Also, check if there are any medication/treatment recommendations
        for pattern in cls.UNSAFE_MEDICATION_TREATMENT_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                text = re.sub(
                    pattern,
                    "[BLOCKED: Medication recommendations are not permitted]",
                    text,
                    flags=re.IGNORECASE
                )

        return text

    @classmethod
    def format_report(cls, report_text: str) -> str:
        """Ensure report text is sanitized and includes safety disclaimers."""
        sanitized = cls.sanitize_report_text(report_text)
        
        # Append the standardized disclaimers at the end
        formatted = (
            f"{sanitized}\n\n"
            f"========================================================================\n"
            f"⚠️ SAFETY DISCLAIMERS AND LIMITATIONS:\n"
            f"{cls.DISCLAIMERS}\n"
            f"========================================================================"
        )
        return formatted

    @classmethod
    def evaluate_query_safety(cls, query: str) -> tuple[bool, list[str]]:
        """Evaluate a clinical/user query for safety. Refuse unsafe/emergency/medication requests.
        
        Returns:
            (is_blocked, list_of_flags)
        """
        normalized = (query or "").strip().lower()
        flags = []

        # Check for medication or treatment advice
        for pattern in cls.UNSAFE_MEDICATION_TREATMENT_PATTERNS:
            if re.search(pattern, normalized):
                flags.append("MEDICATION_ADVICE_REQUEST")
                break

        # Check for emergency/triage requests
        for pattern in cls.EMERGENCY_TRIAGE_PATTERNS:
            if re.search(pattern, normalized):
                flags.append("EMERGENCY_TRIAGE_REQUEST")
                break

        # Check for definitive diagnosis requests or bypass attempts
        if any(re.search(pat, normalized) for pat in [
            r"\b(?:confirm|guarantee|prove|certify|definitive|100\s*%)\b",
            r"\b(?:ignore|bypass|remove)\b.*\b(?:disclaimer|safety|policy|guardrail|rule)\b"
        ]):
            flags.append("DEFINITIVE_DIAGNOSIS_OR_BYPASS_REQUEST")

        is_blocked = len(flags) > 0
        return is_blocked, flags
