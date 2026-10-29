"""
Computes the orientation of every QM9S molecule once, with the simulator's PCA auto-rotation,
and writes it as a table. The simulator (--rotations) and other datasets (e.g. QH9) then use
exactly these values instead of recomputing the PCA, which is not reproducible for molecules
without a well-defined plane (see README.md).

Output .npz, one row per QM9S molecule, sorted by qm9s_index:
    qm9s_index, qm9_index, smiles
    euler_angles: (N, 3) -- (phi, theta, psi) in degrees, as taken by --molecule_rotation
    rotation_matrix: (N, 3, 3) -- image frame = coordinates @ rotation_matrix
    centroid: (N, 3) -- centroid of the QM9S coordinates (unit: A), to place other datasets' copies

Usage:
    python qm9s_adapter/compute_rotations.py /scratch/phys/sin/sethih1/qm9s-raw/qm9s_csv.zip rotations.npz [--pca_atoms all|heavy]
"""
import argparse
import functools
import multiprocessing
import os
import zipfile

import numpy as np

from _repo import bind_package
bind_package()

from ters_img_simulator.core.load_molecule import rotation_matrix
from ters_img_simulator.utils.utils import molecule_normal, normal_to_zyz_ters
from read_qm9s import BOHR_TO_ANGSTROM, parse_qm9s_csv


def pca_angles(coords, atomic_numbers, pca_atoms="all"):
    """
    Euler angles (degrees) of the simulator's auto-rotation, as in point_spectrum_generation.py.
    Usable for any dataset: coords (n, 3) in A. Translation does not matter; the PCA is done in
    bohr, as the .fchk pipeline does. rotation_matrix(*angles) gives R (image frame = coords @ R).
    """
    return normal_to_zyz_ters(molecule_normal(np.asarray(coords) / BOHR_TO_ANGSTROM, atomic_numbers, pca_atoms))


def load_rotations(path):
    """
    Rotation table -> {qm9s_index: euler_angles}.
    """
    with np.load(path) as data:
        return dict(zip(data["qm9s_index"].tolist(), data["euler_angles"]))


_zip = None  # one open zip handle per worker process


def init_worker(zip_path):
    global _zip
    _zip = zipfile.ZipFile(zip_path)


def row(name, pca_atoms):
    mol = parse_qm9s_csv(_zip.read(name).decode())
    angles = pca_angles(mol["coords"], mol["atomic_numbers"], pca_atoms)
    return int(name[-10:-4]), mol["qm9_index"], mol["smiles"], angles, rotation_matrix(*angles), mol["coords"].mean(axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=str, help="Path to qm9s_csv.zip.")
    parser.add_argument("out", type=str, help="Output .npz file.")
    parser.add_argument("--pca_atoms", choices=["all", "heavy"], default="all", help="Atoms used for the PCA normal.")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".csv"))

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    with multiprocessing.Pool(cpus, initializer=init_worker, initargs=(args.zip_path,)) as pool:
        rows = pool.map(functools.partial(row, pca_atoms=args.pca_atoms), names, chunksize=256)

    qm9s_index, qm9_index, smiles, angles, R, centroid = zip(*rows)
    np.savez(args.out, qm9s_index=np.array(qm9s_index), qm9_index=np.array(qm9_index), smiles=np.array(smiles),
             euler_angles=np.array(angles, dtype=float), rotation_matrix=np.array(R), centroid=np.array(centroid))
    print(f"{len(rows)} molecules -> {args.out}")
