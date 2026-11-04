"""Hand-checked end groups for substrates the rules cannot resolve.

Each entry lists the anchor atom indices, in the order of its SMILES, of the ends to keep.
Ends not listed stay in the side chains.
"""

MANUAL_ENDS = [
    {
        # CSV #8: experts convert the two C-aryl nitriles; the N-aryl arms are side chains.
        "smiles": "N#Cc1ccc(-c2cc3c(cc(-c4ccc(C#N)cc4)n3-c3ccc(C#N)cc3)n2-c2ccc(C#N)cc2)cc1",
        "anchors": (1, 16),
        "source": "expert pore side, spreadsheet row 9",
    },
]
