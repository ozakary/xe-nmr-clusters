# XeNMRClusters

---
📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [Academic Portfolio](https://ozakary.github.io/)
---

A Python tool for extracting molecular clusters around Xenon atoms from periodic molecular dynamics trajectories. Designed for preparing input structures for quantum chemistry NMR calculations of Xe magnetic shielding tensors in porous liquids.

## Overview

XeNMRClusters reads periodic MD trajectories and extracts compact, non-periodic molecular clusters centered around Xenon atoms. The tool:
- Identifies molecules using ASE's connectivity analysis
- Extracts complete molecules within a user-defined cutoff radius
- Unwraps periodic boundaries to create correct clusters
- Preserves local atomic environments

## Features

- **Automatic molecule identification** using ASE NeighborList with adaptive cutoff thresholds
- **Complete molecule extraction** - ensures no partial molecules in output
- **PBC-aware distance calculations** - correctly handles wrapped coordinates
- **Connectivity-based unwrapping** - preserves molecular structure (especially important for large cage molecules)
- **Parallel processing** - multi-worker support for fast trajectory processing
- **Multiple solvent types** - supports DCT, HAP, TBA, and other common solvents

## Requirements

```bash
pip install numpy scipy ase tqdm
```

## Usage

Basic usage:

```bash
python extract_xenon_clusters.py \
    -i trajectory.xyz \
    -o clusters.xyz \
    -s DCT \
    -r 9.0 \
    -w 8
```

### Arguments

- `-i, --input`: Input XYZ trajectory file (required)
- `-o, --output`: Output XYZ file for extracted clusters (required)
- `-s, --solvent`: Solvent type: DCT, HAP, or TBA (required)
- `-r, --radius`: Cutoff radius in Angstroms (default: 6.0)
- `-w, --workers`: Number of parallel workers (default: 4)

### Input Format

The tool expects XYZ files with extended format including:
- Periodic lattice information in the comment line
- Atom IDs in the 5th column (after x, y, z coordinates)

Example:
```
710
Timestep=1030 Lattice="21.19 0.0 0.0 0.0 21.19 0.0 0.0 0.0 21.19" Properties=species:S:1:pos:R:3:atom_id:I:1
Xe 10.523 9.876 10.123 1
H 11.234 10.567 9.876 2
...
```

### Output Format

Non-periodic XYZ clusters with:
- Variable number of atoms per snapshot (depends on cutoff and molecular positions)
- Complete molecules only (2 Xe + 1 CC3 cage + N solvent molecules)
- Unwrapped coordinates forming compact clusters
- No lattice information (non-periodic)

## System Requirements

Designed for systems containing:
- 2 Xenon atoms (typically one inside CC3 cage, one outside)
- 1 CC3 porous organic cage (H₈₄C₇₂N₁₂, 168 atoms)
- Multiple solvent molecules (DCT: H₆C₇Cl₂, HAP: H₈C₈O₂, TBA: H₇C₈O₂F₃)

## Algorithm Details

### Molecule Identification
1. Uses ASE's `natural_cutoffs()` based on covalent radii
2. Iteratively adjusts cutoff multiplier (0.75-1.8) to find optimal molecular partitioning
3. Identifies molecules via BFS on connectivity graph
4. Validates compositions against expected molecular formulas
5. Handles edge cases like merged molecules at high temperatures

### Cluster Extraction
1. Finds all atoms within cutoff radius of any Xenon (using minimum image convention)
2. Identifies complete molecules containing those atoms
3. Always includes full CC3 cage (all 168 atoms)
4. Includes complete solvent molecules only

### Unwrapping Strategy
1. Calculates center of both Xenon atoms
2. Unwraps each molecule using BFS along chemical bonds (preserves structure)
3. Positions each molecule at minimum image distance from Xenon center
4. Results in compact, spherical clusters suitable for QM calculations

---
For further details, please refer to the respective folders or contact the author via the provided email.
