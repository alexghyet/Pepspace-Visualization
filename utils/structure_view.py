"""Inspect an uploaded PDB alpha-carbon trace separately from embedding space."""

import math


def read_pdb_trace(text):
    chains = {}
    seen = set()
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break  # A trace of the first model, not a blend of an ensemble.
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA" or line[16:17] not in (" ", "A"):
            continue
        chain, residue = line[21:22].strip() or "_", line[22:27].strip()
        if (chain, residue) in seen:
            continue
        try:
            point = [float(line[start:start + 8]) for start in (30, 38, 46)]
        except ValueError as error:
            raise ValueError("The PDB contains an invalid atom coordinate.") from error
        if not all(math.isfinite(value) for value in point):
            raise ValueError("The PDB contains a nonfinite atom coordinate.")
        seen.add((chain, residue))
        chains.setdefault(chain, []).append((point, f"{line[17:20].strip()} {residue}"))
    if not chains:
        raise ValueError("No alpha-carbon ATOM records were found in the first PDB model.")
    return chains


def show_structure_inspector():
    import plotly.graph_objects as go
    import streamlit as st

    with st.expander("Inspect a molecular structure (PDB)"):
        st.caption("Upload a peptide PDB to inspect its alpha-carbon trace. Molecular coordinates are in ångströms; the embedding globe has different axes. This view does not predict structures or assign the file to a peptide automatically.")
        uploaded = st.file_uploader("Peptide structure (.pdb)", type=["pdb"])
        if uploaded is None:
            return
        try:
            chains = read_pdb_trace(uploaded.getvalue().decode("utf-8"))
        except (ValueError, UnicodeError) as error:
            st.error(str(error))
            return
        figure = go.Figure()
        for chain, records in chains.items():
            # Markers avoid implying bonds across missing residues.
            xyz = list(zip(*(point for point, _ in records)))
            figure.add_trace(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode="markers",
                                         text=[label for _, label in records], name=f"Chain {chain}", marker={"size": 4}))
        figure.update_layout(title=uploaded.name, scene={"aspectmode": "data", "xaxis_title": "X (Å)",
                                                      "yaxis_title": "Y (Å)", "zaxis_title": "Z (Å)"})
        st.plotly_chart(figure, key="molecular_structure", width="stretch")
