"""Hand-checked end groups for substrates the rules cannot resolve.

Atom indices are positions in the entry's SMILES. Each entry may have:
- "anchors": the ends to keep. Ends the rules found but that are not listed stay in the
  side chains. Leave it out to keep every end the rules found.
- "extra_anchors": atoms the main chain must also reach, where no end group marks the place
  (the backbone goes on there in the pore-side drawing). They are not end groups: they are
  not counted in n_ends and are drawn as main chain.
"""

MANUAL_ENDS = [
    {
        # CSV #8: experts convert the two C-aryl nitriles; the N-aryl arms are side chains.
        "smiles": "N#Cc1ccc(-c2cc3c(cc(-c4ccc(C#N)cc4)n3-c3ccc(C#N)cc3)n2-c2ccc(C#N)cc2)cc1",
        "anchors": (1, 16),
        "source": "expert pore side, spreadsheet row 9",
    },
    {
        # CSV #36: the phenyl on the thianthrene is the stub of the second TPE-biphenyl arm,
        # so it is backbone, not a side chain.
        "smiles": "Brc1ccc(C(c2ccc(Br)cc2)C(c2ccc(Br)cc2)c2ccc(-c3ccc(-c4cc5c(cc4Br)Sc4cc(Br)c(-c6ccccc6)cc4S5)cc3)cc2)cc1",
        "extra_anchors": (45,),
        "source": "pore-side workbook, spreadsheet row 39 (substrate and pore side)",
    },
    {
        # CSV #38: the ethyl end is half of the ethylenedianiline bridge, so the backbone goes
        # on through it. The Br positions link; the phenol OH stay pendant.
        "smiles": "CCc1ccc(NCc2cc(O)c(CNc3ccc(CCc4ccc(CNc5ccc(N(c6ccc(NCc7ccc(Br)cc7O)cc6)c6ccc(NCc7ccc(Br)cc7O)cc6)cc5)c(O)c4)cc3)cc2O)cc1",
        "anchors": (42, 58),
        "extra_anchors": (0,),
        "source": "pore-side workbook, spreadsheet row 41 (pore side and molecule with a node)",
    },
    {
        # CSV #39: no reactive group. The backbone runs through one aryl at each end of the
        # anthraquinodimethane (the methoxyphenyl and one far phenyl); the other phenyl at
        # each end is a side chain.
        "smiles": "COc1ccc(C(c2ccccc2)=c2c3ccccc3c(=C(c3ccccc3)c3ccccc3)c3ccccc23)cc1",
        "extra_anchors": (0, 25),
        "source": "pore-side workbook, spreadsheet row 42 (substrate and pore side)",
    },
    {
        # CSV #40: same core as #39. The backbone runs from the dichlorotriazinyl-phenyl through
        # one far phenyl; the other phenyl at each end is a side chain.
        "smiles": "Clc1nc(Cl)nc(-c2ccc(C(c3ccccc3)=c3c4ccccc4c(=C(c4ccccc4)c4ccccc4)c4ccccc34)cc2)n1",
        "extra_anchors": (30,),
        "source": "pore-side workbook, spreadsheet row 43 (substrate and pore side)",
    },
]
