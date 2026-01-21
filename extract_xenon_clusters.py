#!/usr/bin/env python3
"""
Extract molecular clusters around Xenon atoms from MD trajectory XYZ files.

This script processes MD trajectories of porous organic cages (CC3) with solvents
and extracts clusters around Xenon atoms, considering periodic boundary conditions.

Usage:
    python extract_xenon_clusters.py --input trajectory.xyz --output clusters.xyz \
                                      --solvent DCT --radius 6.0 --workers 4

Author: Generated for Ouail's research
"""

import numpy as np
import argparse
from typing import List, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing import Pool
import sys


@dataclass
class Snapshot:
    """Container for a single MD snapshot."""
    timestep: int
    lattice: np.ndarray
    species: List[str]
    positions: np.ndarray
    atom_ids: np.ndarray


# Molecular composition definitions
MOLECULE_COMPOSITIONS = {
    'CC3': {'H': 84, 'C': 72, 'N': 12},  # Total: 168 atoms
    'DCT': {'H': 6, 'C': 7, 'Cl': 2},    # Total: 15 atoms
    'HAP': {'H': 8, 'C': 8, 'O': 2},     # Total: 18 atoms
    'TBA': {'H': 7, 'C': 8, 'O': 2, 'F': 3}  # Total: 20 atoms
}


def parse_xyz_snapshot(lines: List[str], start_idx: int) -> Tuple[Snapshot, int]:
    """
    Parse a single snapshot from XYZ file.
    
    Args:
        lines: All lines from the XYZ file
        start_idx: Starting line index for this snapshot
        
    Returns:
        Tuple of (Snapshot object, next snapshot start index)
    """
    natoms = int(lines[start_idx].strip())
    
    # Parse header line
    header = lines[start_idx + 1]
    timestep = int(header.split('Timestep=')[1].split()[0])
    
    # Parse lattice
    lattice_str = header.split('Lattice="')[1].split('"')[0]
    lattice_values = list(map(float, lattice_str.split()))
    lattice = np.array(lattice_values).reshape(3, 3)
    
    # Parse atoms
    species = []
    positions = []
    atom_ids = []
    
    for i in range(natoms):
        parts = lines[start_idx + 2 + i].split()
        species.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        atom_ids.append(int(parts[4]))
    
    snapshot = Snapshot(
        timestep=timestep,
        lattice=lattice,
        species=species,
        positions=np.array(positions),
        atom_ids=np.array(atom_ids)
    )
    
    return snapshot, start_idx + 2 + natoms


def read_xyz_trajectory(filename: str) -> List[Snapshot]:
    """Read all snapshots from XYZ trajectory file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    snapshots = []
    idx = 0
    
    with tqdm(desc="Reading trajectory", unit=" snapshots") as pbar:
        while idx < len(lines):
            if lines[idx].strip():
                snapshot, idx = parse_xyz_snapshot(lines, idx)
                snapshots.append(snapshot)
                pbar.update(1)
            else:
                idx += 1
    
    return snapshots


def minimum_image_distance(r1: np.ndarray, r2: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Calculate minimum image distance considering PBC.
    
    Args:
        r1: Position array (N, 3) or (3,)
        r2: Position array (M, 3) or (3,)
        box: Box dimensions (3,) - diagonal elements of lattice
        
    Returns:
        Distance array or scalar
    """
    delta = r1 - r2
    delta = delta - box * np.round(delta / box)
    return np.sqrt(np.sum(delta**2, axis=-1))


def find_atoms_within_radius(xenon_pos: np.ndarray, all_positions: np.ndarray, 
                             box: np.ndarray, radius: float) -> np.ndarray:
    """
    Find all atoms within radius of xenon atom using PBC.
    
    Args:
        xenon_pos: Xenon position (3,)
        all_positions: All atomic positions (N, 3)
        box: Box dimensions (3,)
        radius: Cutoff radius
        
    Returns:
        Indices of atoms within radius
    """
    distances = minimum_image_distance(all_positions, xenon_pos, box)
    return np.where(distances <= radius)[0]


def build_molecule_map_from_first_snapshot(snapshot: Snapshot, solvent_type: str) -> dict:
    """
    Build atom_id -> molecule_id mapping using ASE's analysis tools.
    
    Uses ASE's natural_cutoffs and NeighborList for proper molecular identification.
    
    Args:
        snapshot: First snapshot from trajectory
        solvent_type: Type of solvent
        
    Returns:
        Dictionary mapping atom_id -> molecule_id
    """
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, NeighborList
    
    print("  Building ASE Atoms object...")
    
    # Create ASE Atoms object
    atoms = Atoms(
        symbols=snapshot.species,
        positions=snapshot.positions,
        cell=snapshot.lattice,
        pbc=True
    )
    
    print("  Analyzing connectivity with ASE NeighborList...")
    
    # Use natural cutoffs (based on covalent radii) with a multiplier
    # Try wider range to handle different temperatures
    best_result = None
    best_score = -1000
    
    # Try coarse steps first, then fine steps in promising range
    coarse_range = list(np.arange(0.75, 1.5, 0.1))
    fine_range = list(np.arange(0.75, 1.2, 0.02))
    
    for cutoff_multiplier in sorted(set(coarse_range + fine_range)):
        cutoffs = [c * cutoff_multiplier for c in natural_cutoffs(atoms)]
        
        # Build neighbor list
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        # Build connectivity graph using BFS
        n_atoms = len(atoms)
        atom_id_to_mol_id = {}
        visited = set()
        molecule_id = 0
        molecules = []
        
        for start_idx in range(n_atoms):
            if start_idx in visited:
                continue
            
            # BFS to find connected component
            molecule_atoms = []
            queue = [start_idx]
            visited.add(start_idx)
            
            while queue:
                curr_idx = queue.pop(0)
                molecule_atoms.append(curr_idx)
                
                # Get neighbors
                indices, _ = nl.get_neighbors(curr_idx)
                for neighbor_idx in indices:
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)
            
            # Get composition
            comp = {}
            for idx in molecule_atoms:
                spec = snapshot.species[idx]
                comp[spec] = comp.get(spec, 0) + 1
            
            # Map to molecule ID
            for idx in molecule_atoms:
                atom_id_to_mol_id[snapshot.atom_ids[idx]] = molecule_id
            
            molecules.append((molecule_id, len(molecule_atoms), comp))
            molecule_id += 1
        
        # Check composition
        cage_comp = MOLECULE_COMPOSITIONS['CC3']
        solvent_comp = MOLECULE_COMPOSITIONS[solvent_type]
        
        valid_count = sum(1 for _, _, comp in molecules 
                         if comp == {'Xe': 1} or comp == cage_comp or comp == solvent_comp)
        total_count = len(molecules)
        
        # Score: prioritize 39 molecules with high validity
        # Special handling: if we have 38 molecules with one being 30 atoms (2 DCT merged), that's acceptable
        if total_count == 39:
            score = valid_count * 100
        elif total_count == 38 and valid_count == 37:
            # Check if we have one 30-atom molecule (2 DCT merged)
            has_merged_dct = any(size == 30 for _, size, _ in molecules)
            if has_merged_dct:
                score = 90  # Good, but not perfect
            else:
                score = valid_count - abs(39 - total_count) * 10
        else:
            score = valid_count - abs(39 - total_count) * 10
        
        if cutoff_multiplier <= 1.4 or total_count >= 35:
            print(f"    Cutoff multiplier {cutoff_multiplier:.2f}: {total_count} molecules ({valid_count} valid)")
        
        if score > best_score:
            best_score = score
            best_result = (atom_id_to_mol_id, molecules, cutoff_multiplier)
        
        # Perfect result
        if total_count == 39 and valid_count == 39:
            print(f"  ✓ Found 39 molecules (all valid)!")
            mol_sizes = {}
            for atom_id, mol_id in atom_id_to_mol_id.items():
                mol_sizes[mol_id] = mol_sizes.get(mol_id, 0) + 1
            print(f"  DEBUG: Mapped {len(atom_id_to_mol_id)} atom_ids to {len(mol_sizes)} molecules")
            print(f"  DEBUG: Molecule sizes from map: {sorted(mol_sizes.values())}")
            return atom_id_to_mol_id
        
        # Good enough: 38 molecules with one merged DCT pair
        if total_count == 38 and valid_count >= 35:
            # Check if this is 2 DCT merged (30 atoms) or cage fragment
            has_30_atom = any(size == 30 for _, size, _ in molecules)
            if has_30_atom:
                print(f"  ✓ Found 38 molecules (2 DCT merged into one)")
                for mol_id, size, comp in molecules:
                    if size == 30:
                        print(f"    WARNING: Molecule {mol_id} (size {size}): {comp} - likely 2 DCT touching")
                mol_sizes = {}
                for atom_id, mol_id in atom_id_to_mol_id.items():
                    mol_sizes[mol_id] = mol_sizes.get(mol_id, 0) + 1
                print(f"  DEBUG: Mapped {len(atom_id_to_mol_id)} atom_ids to {len(mol_sizes)} molecules")
                print(f"  DEBUG: Molecule sizes from map: {sorted(mol_sizes.values())}")
                return atom_id_to_mol_id
        if total_count == 39 and valid_count >= 37:
            print(f"  ✓ Found 39 molecules ({39-valid_count} warnings)")
            for mol_id, size, comp in molecules:
                if not (comp == {'Xe': 1} or comp == cage_comp or comp == solvent_comp):
                    print(f"    WARNING: Molecule {mol_id} (size {size}): {comp}")
            mol_sizes = {}
            for atom_id, mol_id in atom_id_to_mol_id.items():
                mol_sizes[mol_id] = mol_sizes.get(mol_id, 0) + 1
            print(f"  DEBUG: Mapped {len(atom_id_to_mol_id)} atom_ids to {len(mol_sizes)} molecules")
            print(f"  DEBUG: Molecule sizes from map: {sorted(mol_sizes.values())}")
            return atom_id_to_mol_id
        
        # Stop if fragmenting too much
        if total_count > 60:
            break
    
    # Use best result
    if best_result is None:
        raise ValueError("Could not identify molecules - all cutoffs failed")
    
    atom_id_to_mol_id, molecules, best_mult = best_result
    print(f"  ⚠ Using best result: {len(molecules)} molecules at cutoff {best_mult:.2f}")
    for mol_id, size, comp in molecules[:5]:
        is_valid = comp == {'Xe': 1} or comp == cage_comp or comp == solvent_comp
        if not is_valid:
            print(f"    Molecule {mol_id} (size {size}): {comp}")
    
    mol_sizes = {}
    for atom_id, mol_id in atom_id_to_mol_id.items():
        mol_sizes[mol_id] = mol_sizes.get(mol_id, 0) + 1
    print(f"  DEBUG: Mapped {len(atom_id_to_mol_id)} atom_ids to {len(mol_sizes)} molecules")
    print(f"  DEBUG: Molecule sizes: {sorted(mol_sizes.values())}")
    
    return atom_id_to_mol_id


def build_molecule_map(atom_ids: np.ndarray, atom_id_to_mol_id: dict) -> dict:
    """
    Build mapping from atom index to molecule ID using pre-computed atom_id mapping.
    
    Args:
        atom_ids: Array of atom IDs for this snapshot
        atom_id_to_mol_id: Pre-computed mapping from atom_id -> molecule_id
        
    Returns:
        Dictionary mapping atom_idx -> molecule_id
    """
    molecule_map = {}
    for i, atom_id in enumerate(atom_ids):
        if atom_id in atom_id_to_mol_id:
            molecule_map[i] = atom_id_to_mol_id[atom_id]
        else:
            raise ValueError(f"Atom ID {atom_id} not found in molecule map")
    return molecule_map


def get_molecule_indices(atom_idx: int, molecule_map: dict) -> Set[int]:
    """
    Get all atoms in the same molecule as atom_idx.
    
    Args:
        atom_idx: Index of the atom
        molecule_map: Dictionary mapping atom_idx -> molecule_id
        
    Returns:
        Set of indices belonging to the same molecule
    """
    mol_id = molecule_map[atom_idx]
    return {idx for idx, mid in molecule_map.items() if mid == mol_id}


def unwrap_cluster(positions: np.ndarray, species: List[str], 
                   cluster_indices: List[int], original_molecule_map: dict,
                   xenon_cluster_indices: List[int], box: np.ndarray) -> np.ndarray:
    """
    Unwrap cluster using ASE to create a compact blob around Xenon atoms.
    
    Uses connectivity-aware unwrapping for large molecules like CC3 cage.
    
    Args:
        positions: Atomic positions in cluster (N, 3)
        species: List of species for each atom in cluster
        cluster_indices: Original indices of cluster atoms
        original_molecule_map: Mapping of original indices to molecule IDs
        xenon_cluster_indices: Indices of Xenon atoms within cluster
        box: Box dimensions (3,)
        
    Returns:
        Unwrapped positions forming a compact cluster
    """
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, NeighborList
    
    # Create ASE Atoms object for the cluster
    cell = np.diag(box)
    atoms = Atoms(
        symbols=species,
        positions=positions,
        cell=cell,
        pbc=True
    )
    
    # Calculate Xenon center as reference point
    xe_positions = atoms.positions[xenon_cluster_indices]
    xe_center = np.mean(xe_positions, axis=0)
    
    # Group atoms by molecule
    cluster_molecules = {}
    for i, orig_idx in enumerate(cluster_indices):
        mol_id = original_molecule_map[orig_idx]
        if mol_id not in cluster_molecules:
            cluster_molecules[mol_id] = []
        cluster_molecules[mol_id].append(i)
    
    # Build neighbor list for connectivity
    cutoffs = natural_cutoffs(atoms, mult=1.1)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    # For each molecule, make it whole using BFS along bonds
    unwrapped_positions = atoms.positions.copy()
    
    for mol_id, atom_indices in cluster_molecules.items():
        if len(atom_indices) == 1:
            # Single atom (Xenon) - just apply minimum image to Xenon center
            idx = atom_indices[0]
            delta = atoms.positions[idx] - xe_center
            delta = delta - box * np.round(delta / box)
            unwrapped_positions[idx] = xe_center + delta
        else:
            # Multi-atom molecule - unwrap using BFS along connectivity
            mol_indices_set = set(atom_indices)
            
            # Start from first atom (arbitrary choice)
            start_idx = atom_indices[0]
            
            # Place start atom at minimum image from Xenon
            delta = atoms.positions[start_idx] - xe_center
            delta = delta - box * np.round(delta / box)
            unwrapped_positions[start_idx] = xe_center + delta
            
            # BFS to unwrap all connected atoms
            visited = {start_idx}
            queue = [start_idx]
            
            while queue:
                curr_idx = queue.pop(0)
                
                # Get bonded neighbors
                indices, offsets = nl.get_neighbors(curr_idx)
                
                for neighbor_idx, offset in zip(indices, offsets):
                    # Only process atoms in this molecule
                    if neighbor_idx not in mol_indices_set:
                        continue
                    
                    if neighbor_idx in visited:
                        continue
                    
                    # Unwrap neighbor relative to current atom
                    # Account for PBC offset from neighbor list
                    delta = atoms.positions[neighbor_idx] - atoms.positions[curr_idx]
                    delta = delta - box * np.round(delta / box)
                    unwrapped_positions[neighbor_idx] = unwrapped_positions[curr_idx] + delta
                    
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
            
            # Now shift entire molecule so its COM is at minimum image from Xenon
            mol_positions = unwrapped_positions[atom_indices]
            mol_com = np.mean(mol_positions, axis=0)
            
            delta_com = mol_com - xe_center
            shift = box * np.round(delta_com / box)
            
            # Apply shift
            for idx in atom_indices:
                unwrapped_positions[idx] -= shift
    
    # Recenter everything so Xenon center is at a nice position
    final_xe_center = np.mean(unwrapped_positions[xenon_cluster_indices], axis=0)
    unwrapped_positions = unwrapped_positions - final_xe_center + box/2  # Center in box
    
    return unwrapped_positions


def validate_cluster_composition(species: List[str], solvent_type: str) -> Tuple[bool, str]:
    """
    Validate that the cluster has correct molecular composition.
    
    Args:
        species: List of atomic species in cluster
        solvent_type: Type of solvent
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Count species
    species_count = {}
    for s in species:
        species_count[s] = species_count.get(s, 0) + 1
    
    # Check Xenon
    if species_count.get('Xe', 0) != 2:
        return False, f"Expected 2 Xe atoms, found {species_count.get('Xe', 0)}"
    
    # Check CC3 cage
    cage_comp = MOLECULE_COMPOSITIONS['CC3']
    for elem, count in cage_comp.items():
        if species_count.get(elem, 0) < count:
            return False, f"CC3 cage incomplete: expected {count} {elem}, found {species_count.get(elem, 0)}"
    
    # Calculate number of solvent molecules
    solvent_comp = MOLECULE_COMPOSITIONS[solvent_type]
    solvent_atoms_per_mol = sum(solvent_comp.values())
    
    # Total atoms in cluster minus Xe and cage
    total_atoms = len(species)
    cage_total = sum(cage_comp.values())
    remaining_atoms = total_atoms - 2 - cage_total
    
    if remaining_atoms % solvent_atoms_per_mol != 0:
        return False, f"Partial solvent molecules: {remaining_atoms} atoms, expected multiple of {solvent_atoms_per_mol}"
    
    num_solvents = remaining_atoms // solvent_atoms_per_mol
    
    # Validate solvent composition
    # Subtract cage atoms from counts
    for elem, count in cage_comp.items():
        species_count[elem] -= count
    
    # Subtract Xe
    species_count['Xe'] -= 2
    
    # Check solvents
    for elem, count_per_mol in solvent_comp.items():
        expected_count = num_solvents * count_per_mol
        actual_count = species_count.get(elem, 0)
        if actual_count != expected_count:
            return False, f"Solvent composition incorrect: expected {expected_count} {elem}, found {actual_count}"
    
    return True, f"Valid cluster: 2 Xe + 1 CC3 + {num_solvents} {solvent_type} molecules ({total_atoms} atoms)"


def extract_cluster_from_snapshot(snapshot: Snapshot, radius: float, 
                                  solvent_type: str, atom_id_to_mol_id: dict) -> Tuple[Snapshot, str]:
    """
    Extract cluster around Xenon atoms from a single snapshot.
    
    Args:
        snapshot: Input snapshot
        radius: Cutoff radius in Angstroms
        solvent_type: Type of solvent
        atom_id_to_mol_id: Pre-computed mapping from atom_id -> molecule_id
        
    Returns:
        Tuple of (cluster snapshot, validation message)
    """
    box = np.diag(snapshot.lattice)
    
    # Find Xenon atoms
    xenon_indices = [i for i, s in enumerate(snapshot.species) if s == 'Xe']
    
    if len(xenon_indices) != 2:
        raise ValueError(f"Expected 2 Xenon atoms, found {len(xenon_indices)}")
    
    # Start with Xenon atoms
    cluster_atom_indices = set(xenon_indices)
    
    # ALWAYS include complete CC3 cage
    # The cage is the molecule with composition H84C72N12
    # Find which molecule_id corresponds to the cage by checking the molecule map
    # We know from the first snapshot which molecule is the cage
    cage_mol_id = None
    mol_compositions = {}
    
    # Count atoms per molecule to identify cage
    for atom_id, mol_id in atom_id_to_mol_id.items():
        if mol_id not in mol_compositions:
            mol_compositions[mol_id] = {}
        # Find this atom in first snapshot to get its species
        # We need to determine species from the first snapshot mapping
        # Actually, we can just check all atoms in current snapshot
    
    # Simpler approach: find cage by checking if a molecule has 168 atoms in the map
    mol_sizes_in_map = {}
    for atom_id, mol_id in atom_id_to_mol_id.items():
        mol_sizes_in_map[mol_id] = mol_sizes_in_map.get(mol_id, 0) + 1
    
    # Cage has 168 atoms, Xe has 1, DCT has 15
    for mol_id, size in mol_sizes_in_map.items():
        if size == 168:
            cage_mol_id = mol_id
            break
    
    if cage_mol_id is None:
        raise ValueError("Could not identify CC3 cage molecule in atom_id_to_mol_id map")
    
    # Include all atoms of the cage molecule from current snapshot
    for i, atom_id in enumerate(snapshot.atom_ids):
        if atom_id in atom_id_to_mol_id and atom_id_to_mol_id[atom_id] == cage_mol_id:
            cluster_atom_indices.add(i)
    
    # Find solvent molecules within radius of ANY Xenon
    # Use atom IDs to identify which molecules to include
    solvent_molecules_to_include = set()
    
    for xe_idx in xenon_indices:
        # Find all atoms within radius (including across PBC)
        nearby_indices = find_atoms_within_radius(
            snapshot.positions[xe_idx], 
            snapshot.positions, 
            box, 
            radius
        )
        
        # For each nearby atom, find its molecule via atom_id
        for atom_idx in nearby_indices:
            # Skip if it's Xenon or cage atom (already included)
            if snapshot.species[atom_idx] in ['Xe', 'H', 'C', 'N']:
                continue
            
            # Get the molecule ID for this solvent atom using its atom_id
            atom_id = snapshot.atom_ids[atom_idx]
            if atom_id in atom_id_to_mol_id:
                mol_id = atom_id_to_mol_id[atom_id]
                solvent_molecules_to_include.add(mol_id)
    
    # Add ALL atoms from selected solvent molecules (by atom_id)
    # We need to find all atoms in this snapshot that belong to these molecules
    atoms_found_for_molecules = {mol_id: 0 for mol_id in solvent_molecules_to_include}
    
    for i, atom_id in enumerate(snapshot.atom_ids):
        if atom_id in atom_id_to_mol_id:
            mol_id = atom_id_to_mol_id[atom_id]
            if mol_id in solvent_molecules_to_include:
                cluster_atom_indices.add(i)
                atoms_found_for_molecules[mol_id] += 1
    
    # Convert to sorted list
    cluster_indices = sorted(list(cluster_atom_indices))
    
    # Extract cluster data
    cluster_species = [snapshot.species[i] for i in cluster_indices]
    cluster_positions = snapshot.positions[cluster_indices]
    cluster_ids = snapshot.atom_ids[cluster_indices]
    
    # Find Xenon indices in cluster
    xe_indices_in_cluster = [i for i, idx in enumerate(cluster_indices) if idx in xenon_indices]
    
    # Build molecule map for unwrapping - maps ORIGINAL indices to molecule_id
    original_molecule_map = {}
    for orig_idx in cluster_indices:
        atom_id = snapshot.atom_ids[orig_idx]
        if atom_id in atom_id_to_mol_id:
            original_molecule_map[orig_idx] = atom_id_to_mol_id[atom_id]
    
    # Unwrap PBC
    unwrapped_positions = unwrap_cluster(
        cluster_positions, 
        cluster_species,
        cluster_indices,
        original_molecule_map,
        xe_indices_in_cluster, 
        box
    )
    
    # Create cluster snapshot
    cluster_snapshot = Snapshot(
        timestep=snapshot.timestep,
        lattice=snapshot.lattice,
        species=cluster_species,
        positions=unwrapped_positions,
        atom_ids=cluster_ids
    )
    
    # Validate composition
    is_valid, message = validate_cluster_composition(cluster_species, solvent_type)
    
    if not is_valid:
        raise ValueError(f"Cluster validation failed: {message}")
    
    return cluster_snapshot, message


def process_snapshot_wrapper(args):
    """Wrapper for parallel processing."""
    snapshot, radius, solvent_type, snap_idx, atom_id_to_mol_id = args
    try:
        cluster, message = extract_cluster_from_snapshot(snapshot, radius, solvent_type, atom_id_to_mol_id)
        return snap_idx, cluster, message, None
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n"
        if snap_idx < 3:  # Detailed traceback for first few
            error_msg += traceback.format_exc()
        return snap_idx, None, None, error_msg


def write_xyz_snapshot(f, snapshot: Snapshot):
    """Write a single snapshot to XYZ file without lattice (non-periodic cluster)."""
    natoms = len(snapshot.species)
    
    f.write(f"{natoms}\n")
    f.write(f'Timestep={snapshot.timestep}\n')
    
    for i in range(natoms):
        f.write(f"{snapshot.species[i]} {snapshot.positions[i, 0]:.6f} "
                f"{snapshot.positions[i, 1]:.6f} {snapshot.positions[i, 2]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract molecular clusters around Xenon atoms from MD trajectories',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, help='Input XYZ trajectory file')
    parser.add_argument('--output', '-o', required=True, help='Output XYZ trajectory file')
    parser.add_argument('--solvent', '-s', required=True, choices=['DCT', 'HAP', 'TBA'],
                       help='Solvent type')
    parser.add_argument('--radius', '-r', type=float, default=6.0,
                       help='Cutoff radius in Angstroms')
    parser.add_argument('--workers', '-w', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--reference', '-ref', default=None,
                       help='Reference XYZ file with whole molecules (optional)')
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"Xenon Cluster Extraction")
    print(f"{'='*70}")
    print(f"Input file    : {args.input}")
    print(f"Output file   : {args.output}")
    print(f"Solvent type  : {args.solvent}")
    print(f"Cutoff radius : {args.radius} Å")
    print(f"Workers       : {args.workers}")
    if args.reference:
        print(f"Reference file: {args.reference}")
    print(f"{'='*70}\n")
    
    # Read trajectory
    print("Reading input trajectory...")
    snapshots = read_xyz_trajectory(args.input)
    print(f"Read {len(snapshots)} snapshots\n")
    
    # Build atom_id -> molecule_id mapping from first snapshot or reference
    atom_id_to_mol_id = None
    if args.reference:
        print("Building molecule map from reference structure...")
        ref_snapshots = read_xyz_trajectory(args.reference)
        if len(ref_snapshots) == 0:
            print("ERROR: Reference file is empty!")
            sys.exit(1)
        atom_id_to_mol_id = build_molecule_map_from_first_snapshot(ref_snapshots[0], args.solvent)
    else:
        print("Building molecule map from first snapshot...")
        atom_id_to_mol_id = build_molecule_map_from_first_snapshot(snapshots[0], args.solvent)
    
    print()
    
    # Process snapshots
    print("Extracting clusters...")
    
    if args.workers > 1:
        # Parallel processing
        with Pool(processes=args.workers) as pool:
            tasks = [(snap, args.radius, args.solvent, i, atom_id_to_mol_id) 
                    for i, snap in enumerate(snapshots)]
            results = list(tqdm(
                pool.imap(process_snapshot_wrapper, tasks),
                total=len(snapshots),
                desc="Processing snapshots"
            ))
    else:
        # Serial processing
        results = []
        for i, snap in enumerate(tqdm(snapshots, desc="Processing snapshots")):
            results.append(process_snapshot_wrapper((snap, args.radius, args.solvent, i, atom_id_to_mol_id)))
    
    # Check for errors
    errors = [(idx, err) for idx, _, _, err in results if err is not None]
    if errors:
        print(f"\n{'!'*70}")
        print(f"ERRORS OCCURRED:")
        for idx, err in errors[:10]:  # Show first 10 errors
            print(f"  Snapshot {idx}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print(f"{'!'*70}\n")
        sys.exit(1)
    
    # Extract successful clusters
    clusters = [cluster for _, cluster, _, _ in results if cluster is not None]
    
    # Print validation summary
    print(f"\n{'='*70}")
    print("Validation Summary:")
    print(f"{'='*70}")
    messages = set(msg for _, _, msg, _ in results if msg is not None)
    for msg in messages:
        print(f"  {msg}")
    print(f"{'='*70}\n")
    
    # Write output
    print(f"Writing output trajectory to {args.output}...")
    with open(args.output, 'w') as f:
        for cluster in tqdm(clusters, desc="Writing snapshots"):
            write_xyz_snapshot(f, cluster)
    
    print(f"\n{'='*70}")
    print(f"Successfully processed {len(clusters)} snapshots")
    print(f"Output written to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
