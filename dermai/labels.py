HAM10000_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

HAM10000_LABEL_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

LABEL_TO_INDEX = {label: index for index, label in enumerate(HAM10000_LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
