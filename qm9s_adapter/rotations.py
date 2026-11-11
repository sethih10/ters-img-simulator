"""
Computes the orientation of every QM9S molecule once and writes it as a rotation table.

The simulator (--rotations) and derived datasets (e.g. QH9) use these stored values, because PCA
orientation is not reproducible for molecules without a well-defined plane (see README.md).

Table fields (one row per molecule, sorted by qm9s_index):
    qm9s_index, qm9_index, smiles
    euler_angles      (N, 3)    (phi, theta, psi) in degrees
    rotation_matrix   (N, 3, 3) image frame = (coords - c) @ rotation_matrix
    centroid          (N, 3)    centroid of all atoms, Å (c for --center all)
    heavy_centroid    (N, 3)    centroid of the non-hydrogen atoms, Å (c for --center heavy)
    pca_atoms         str       atoms used for the PCA normal

Usage:
    python qm9s_adapter/rotations.py qm9s_csv.zip qm9s_rotations.npz [--pca_atoms {all,heavy}]
"""
import argparse
import functools
import multiprocessing
import os
import zipfile
from pathlib import PurePosixPath

import numpy as np

from ters_img_simulator.core.load_molecule import rotation_matrix
from ters_img_simulator.qm9s_adapter.reader import BOHR_TO_ANGSTROM, parse_csv
from ters_img_simulator.utils.utils import molecule_normal, normal_to_zyz_ters


def pca_angles(coords, atomic_numbers, pca_atoms="all"):
    """
    Euler angles (degrees) of the simulator's PCA auto-rotation for coordinates in Å.
    The PCA is done in bohr, as in the .fchk pipeline.
    """
    return normal_to_zyz_ters(molecule_normal(np.asarray(coords) / BOHR_TO_ANGSTROM, atomic_numbers, pca_atoms))


def molecule_center(coords, atomic_numbers, center="none"):
    """
    Point moved to the grid centre before rotation, in Å: the origin ("none"), the centroid of all
    atoms ("all") or of the non-hydrogen atoms ("heavy"; all atoms if there are none).
    """
    if center not in ("none", "all", "heavy"):
        raise ValueError(f"center must be 'none', 'all' or 'heavy', got {center!r}")
    coords = np.asarray(coords)
    if center == "none":
        return np.zeros(3)
    heavy = coords[np.asarray(atomic_numbers) > 1]
    return (heavy if center == "heavy" and len(heavy) else coords).mean(axis=0)


def load_table(path):
    """
    Rotation table -> {qm9s_index: euler_angles}.
    """
    with np.load(path) as table:
        return dict(zip(table["qm9s_index"].tolist(), table["euler_angles"]))


_zip = None  # zip archive, opened once per worker process


def _init_worker(zip_path):
    global _zip
    _zip = zipfile.ZipFile(zip_path)


def _table_row(name, pca_atoms):
    mol = parse_csv(_zip.read(name).decode())
    coords, z = mol["coords"], mol["atomic_numbers"]
    angles = pca_angles(coords, z, pca_atoms)
    return (int(PurePosixPath(name).stem), mol["qm9_index"], mol["smiles"], angles, rotation_matrix(*angles),
            molecule_center(coords, z, "all"), molecule_center(coords, z, "heavy"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip_path", help="QM9S archive (qm9s_csv.zip).")
    parser.add_argument("out", help="Output .npz file.")
    parser.add_argument("--pca_atoms", choices=["all", "heavy"], default="all", help="Atoms used for the PCA normal.")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".csv"))
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    with multiprocessing.Pool(cpus, initializer=_init_worker, initargs=(args.zip_path,)) as pool:
        rows = pool.map(functools.partial(_table_row, pca_atoms=args.pca_atoms), names, chunksize=256)

    fields = ("qm9s_index", "qm9_index", "smiles", "euler_angles", "rotation_matrix", "centroid", "heavy_centroid")
    table = {field: np.array(values) for field, values in zip(fields, zip(*rows))}
    np.savez(args.out, **table, pca_atoms=np.array(args.pca_atoms))
    print(f"{len(rows)} molecules -> {args.out}")


if __name__ == "__main__":
    main()
