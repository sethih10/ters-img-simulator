"""
Reads a QM9S molecule and returns it in the format of core.read_gaussian.read_gaussian.

QM9S stores the Hessian instead of normal modes, so the harmonic vibrational analysis performed by
Gaussian (Vib-E2, Vib-Modes) is repeated here. See README.md for the derivation and validation.
"""
import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903

# Most abundant isotope masses (amu), as used by Gaussian for frequencies (Vib-AtMass)
ISOTOPE_MASSES = {1: 1.00782504, 6: 12.0, 7: 14.0030740, 8: 15.9949146, 9: 18.9984032}

# Eigenvalue of the mass-weighted Hessian in eV/(Å² amu) -> wavenumber in cm⁻¹
EV_A2_AMU_TO_CM1 = np.sqrt(1.602176634e-19 / (1e-20 * 1.66053906660e-27)) / (2 * np.pi * 2.99792458e10)


def parse_csv(text):
    """
    Parses one QM9S molecule CSV.

    Returns a dict with qm9_index (int, original QM9 number), smiles, atomic_numbers (n,),
    coords (n, 3) in Å, polar_derivatives (n, 3, 6) in bohr³/Å with components
    (xx, xy, yy, xz, yz, zz), and hessian (3n, 3n) in eV/Å².
    """
    rows = [[c for c in line.split(",")[1:] if c != ""] for line in text.splitlines()[1:]]
    n = int(float(rows[0][3]))
    xyz = np.array(rows[10:10 + n], dtype=float)
    return {
        "qm9_index": int(float(rows[0][1])),
        "smiles": rows[0][2],
        "atomic_numbers": xyz[:, 0].astype(int),
        "coords": xyz[:, 1:],
        "polar_derivatives": np.array(rows[10 + 2*n:10 + 3*n], dtype=float).reshape(n, 3, 6),
        "hessian": np.array(rows[10 + 3*n:10 + 6*n], dtype=float),
    }


def normal_modes(atomic_numbers, coords, hessian):
    """
    Harmonic vibrational analysis with Gaussian's conventions. Translations and rotations are
    projected out exactly before diagonalisation.

    Returns frequencies (3n-6,) in cm⁻¹ (ascending, imaginary as negative), reduced masses (3n-6,)
    in amu, and modes (3n, 3n-6) as unit-length Cartesian displacements.
    """
    masses = np.array([ISOTOPE_MASSES[z] for z in atomic_numbers])
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    hessian_mw = hessian / np.outer(sqrt_m, sqrt_m)

    r = coords - masses @ coords / masses.sum()
    rigid = [(np.sqrt(masses)[:, None] * v).ravel() for e in np.eye(3) for v in (e, np.cross(e, r))]
    U, s, _ = np.linalg.svd(np.array(rigid).T)
    if np.sum(s > 1e-4 * s[0]) != 6:
        raise ValueError("Linear molecule (3n-5 modes); the simulator assumes 3n-6 modes")

    D = U[:, 6:]  # orthonormal basis of the vibrational subspace
    eigvals, eigvecs = np.linalg.eigh(D.T @ hessian_mw @ D)
    frequencies = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * EV_A2_AMU_TO_CM1
    l_cart = (D @ eigvecs) / sqrt_m[:, None]
    red_masses = 1 / np.sum(l_cart**2, axis=0)
    return frequencies, red_masses, l_cart * np.sqrt(red_masses)


def to_gaussian_format(atomic_numbers, coords, hessian, polar_derivatives):
    """
    Converts QM9S quantities to the units and layout returned by read_gaussian.
    """
    frequencies, red_masses, modes = normal_modes(atomic_numbers, coords, hessian)
    polar_au = (polar_derivatives * BOHR_TO_ANGSTROM).ravel()  # bohr³/Å -> bohr² (a.u.)
    return (red_masses, frequencies, polar_au, modes, (coords / BOHR_TO_ANGSTROM).ravel(),
            atomic_numbers, len(atomic_numbers), 0)


def read_qm9s(text):
    """
    Drop-in replacement for read_gaussian: QM9S CSV text -> read_gaussian tuple.
    """
    mol = parse_csv(text)
    return to_gaussian_format(mol["atomic_numbers"], mol["coords"], mol["hessian"], mol["polar_derivatives"])
