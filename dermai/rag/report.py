from __future__ import annotations

from typing import Any

from dermai.labels import HAM10000_LABEL_DESCRIPTIONS


def _format_probability_row(item: dict) -> str:
    class_name = item.get("class_name", "unknown")
    description = item.get("description") or HAM10000_LABEL_DESCRIPTIONS.get(class_name, class_name)
    probability = float(item.get("probability", 0.0))
    return f"- {description} ({class_name}): {probability:.1%}"


def generate_triage_report(
    probabilities: list[dict],
    symptoms: str,
    guidance_chunks: list[Any],
) -> str:
    """Generate cautious educational triage text without diagnostic claims."""

    symptom_text = symptoms.strip() or "No symptom details were provided."
    probability_lines = "\n".join(_format_probability_row(item) for item in probabilities[:5])

    guidance_lines = []
    for chunk in guidance_chunks[:3]:
        source = str(getattr(chunk, "source", "unknown"))
        text = str(getattr(chunk, "text", ""))
        if isinstance(chunk, dict):
            source = str(chunk.get("source", source))
            text = str(chunk.get("text", text))
        if text:
            guidance_lines.append(f"- From {source}: {text}")

    guidance_text = "\n".join(guidance_lines) or "- No local guidance chunks were retrieved."

    return f"""## Educational Triage Report

This report is informational only. It does not diagnose, rule out disease, or replace a clinician's assessment.

### User Context

{symptom_text}

### Model Output

The image model produced the following visual pattern ranking:

{probability_lines}

These probabilities are not clinical certainty. Image quality, lighting, lesion location, skin tone, missing history, and dataset bias can substantially change model behavior.

### Retrieved Safety Guidance

{guidance_text}

### Cautious Next Steps

- Arrange a non-urgent review with a qualified clinician for persistent, changing, painful, bleeding, rapidly growing, or concerning skin lesions.
- Seek urgent care if there is rapid spreading redness, fever, severe pain, pus, blackening skin, or other signs of acute infection or tissue injury.
- Consider prompt dermatology review when a pigmented lesion is new, changing, asymmetric, has irregular borders, has multiple colors, is larger than usual, itches, bleeds, or looks notably different from other spots.
- Do not start, stop, or change treatment based only on this software output.
"""
