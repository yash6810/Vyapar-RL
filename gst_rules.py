"""
gst_rules.py — GST slab lookup and validation logic.
Ported from Vyapar's knowledge_base.
"""

# GST slab rates
GST_SLABS = [0, 5, 12, 18, 28]

# Keyword → slab mapping for Task 1 grading
CATEGORY_SLAB_MAP = {
    # 0% items
    "vegetable": 0, "fruit": 0, "milk": 0, "egg": 0, "bread": 0,
    "book": 0, "newspaper": 0, "education": 0, "healthcare": 0,
    "hospital": 0, "medicine": 0, "agricultural": 0, "fresh food": 0,

    # 5% items
    "tea": 5, "coffee": 5, "spice": 5, "packaged food": 5,
    "restaurant": 5, "transport": 5, "rail": 5, "bus": 5,
    "fertilizer": 5, "medicine": 5, "pharma": 5,

    # 12% items
    "processed food": 12, "frozen": 12, "computer": 12, "laptop": 12,
    "printer": 12, "mobile phone": 12, "hotel budget": 12,

    # 18% items — most common for SMEs
    "it service": 18, "software": 18, "saas": 18, "cloud": 18,
    "aws": 18, "azure": 18, "gcp": 18, "google cloud": 18,
    "telecom": 18, "internet": 18, "broadband": 18,
    "financial service": 18, "insurance": 18, "banking": 18,
    "consulting": 18, "professional service": 18,
    "zomato": 18, "swiggy": 18, "food delivery": 18,
    "uber": 18, "ola": 18, "cab": 18, "taxi": 18,
    "hotel": 18, "restaurant ac": 18, "ac restaurant": 18,
    "stationery": 18, "office supply": 18, "furniture": 18,
    "electronic": 18, "tv": 18, "washing machine": 18, "fridge": 18,

    # 28% items
    "luxury car": 28, "tobacco": 28, "cigarette": 28, "bidi": 28,
    "aerated drink": 28, "cola": 28, "pepsi": 28, "casino": 28,
    "lottery": 28, "cement": 28, "paint": 28, "dye": 28,
    "motorcycle luxury": 28, "luxury hotel": 28,
}


def get_expected_slab(description: str) -> int:
    """Return the expected GST slab for a transaction description."""
    desc_lower = description.lower()
    for keyword, slab in CATEGORY_SLAB_MAP.items():
        if keyword in desc_lower:
            return slab
    return 18  # Default: most SME B2B services are 18%


def validate_slab(slab: int) -> bool:
    """Check if a slab value is valid."""
    return slab in GST_SLABS


def compute_gst(taxable_value: float, slab: int,
                is_intrastate: bool = True) -> dict:
    """Compute GST components for a transaction."""
    total_gst = taxable_value * slab / 100
    if is_intrastate:
        return {
            "cgst": round(total_gst / 2, 2),
            "sgst": round(total_gst / 2, 2),
            "igst": 0.0,
            "total": round(total_gst, 2)
        }
    else:
        return {
            "cgst": 0.0,
            "sgst": 0.0,
            "igst": round(total_gst, 2),
            "total": round(total_gst, 2)
        }
