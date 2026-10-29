"""
QM9S -> TERS simulator adapter.

Turns one QM9S molecule CSV (from qm9s_csv.zip) into exactly the tuple that
core.read_gaussian.read_gaussian returns for a Gaussian .fchk file, so the rest of
the simulator (load_molecule, analytic_field, spectrum_real) runs unchanged.

QM9S stores the Hessian instead of normal modes, so the vibrational analysis that
Gaussian does (Vib-E2, Vib-Modes) is redone here. See README.md for the derivation
and the validation.
"""
import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903

# Most abundant isotope masses (amu), the masses Gaussian uses for frequencies (Vib-AtMass)
ISOTOPE_MASSES = {1: 1.00782504, 6: 12.0, 7: 14.0030740, 8: 15.9949146, 9: 18.9984032}

# Eigenvalue of the mass-weighted Hessian in eV/(A^2 amu) -> wavenumber in cm^-1
EV_A2_AMU_TO_CM1 = np.sqrt(1.602176634e-19 / (1e-20 * 1.66053906660e-27)) / (2 * np.pi * 2.99792458e10)


def parse_qm9s_csv(text):
    """
    Parses one QM9S molecule CSV (layout described in qm9s-raw/readme.txt).

    Returns a dict with:
        qm9_index: int -- molecule number in the original QM9 (to match other QM9-derived datasets)
        smiles: str
        atomic_numbers: np.ndarray (n,)
        coords: np.ndarray (n, 3) -- unit: A
        polar_derivatives: np.ndarray (n, 3, 6) -- d alpha / d R_(atom, x|y|z), components
                           (xx, xy, yy, xz, yz, zz); unit: Bohr^3/A
        hessian: np.ndarray (3n, 3n) -- unit: eV/A^2
    """
    # First line is the column header, first column is the row label
    rows = [[c for c in line.split(",")[1:] if c != ""] for line in text.splitlines()[1:]]
    n = int(float(rows[0][3]))

    xyz = np.array(rows[10:10 + n], dtype=float)
    polar = np.array(rows[10 + 2*n:10 + 3*n], dtype=float)
    hessian = np.array(rows[10 + 3*n:10 + 6*n], dtype=float)

    return {
        "qm9_index": int(float(rows[0][1])),
        "smiles": rows[0][2],
        "atomic_numbers": xyz[:, 0].astype(int),
        "coords": xyz[:, 1:],
        "polar_derivatives": polar.reshape(n, 3, 6),
        "hessian": hessian,
    }


def normal_modes(atomic_numbers, coords, hessian):
    """
    Harmonic vibrational analysis, following Gaussian's conventions.

    Translations and rotations are projected out of the mass-weighted Hessian, which is
    then diagonalized in the 3n-6 dimensional vibrational subspace.

    Arguments:
        atomic_numbers: np.ndarray (n,)
        coords: np.ndarray (n, 3) -- unit: A
        hessian: np.ndarray (3n, 3n) -- unit: eV/A^2

    Returns:
        frequencies: np.ndarray (3n-6,) -- ascending, imaginary modes negative (unit: cm^-1)
        red_masses: np.ndarray (3n-6,) -- reduced masses (unit: amu)
        modes: np.ndarray (3n, 3n-6) -- Cartesian displacements, each column of unit length
    """
    masses = np.array([ISOTOPE_MASSES[z] for z in atomic_numbers])
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    hessian_mw = hessian / np.outer(sqrt_m, sqrt_m)

    # Translations and infinitesimal rotations about the centre of mass, mass-weighted
    r = coords - masses @ coords / masses.sum()
    tr = []
    for e in np.eye(3):
        tr.append((np.sqrt(masses)[:, None] * e).ravel())
        tr.append((np.sqrt(masses)[:, None] * np.cross(e, r)).ravel())
    U, s, _ = np.linalg.svd(np.array(tr).T)
    if np.sum(s > 1e-4 * s[0]) != 6:
        raise ValueError("Linear molecule (3n-5 modes); the simulator assumes 3n-6 modes")

    D = U[:, 6:]  # orthonormal basis of the vibrational subspace, (3n, 3n-6)
    eigvals, eigvecs = np.linalg.eigh(D.T @ hessian_mw @ D)

    frequencies = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * EV_A2_AMU_TO_CM1
    l_cart = (D @ eigvecs) / sqrt_m[:, None]   # Cartesian displacements of unit mass-weighted modes
    red_masses = 1 / np.sum(l_cart**2, axis=0)
    modes = l_cart * np.sqrt(red_masses)       # normalized, as in Gaussian's Vib-Modes

    return frequencies, red_masses, modes


def to_gaussian_tuple(atomic_numbers, coords, hessian, polar_derivatives):
    """
    Converts QM9S quantities to the units and layout of read_gaussian.

    Returns:
        red_masses, frequencies, polar_derivatives (flat, a.u.), atom_an (3n, 3n-6),
        atom_positions (flat, bohr), atomic_numbers, N1, N2
    """
    frequencies, red_masses, modes = normal_modes(atomic_numbers, coords, hessian)

    # QM9S (n, 3, 6) is already Gaussian's order; only Bohr^3/A -> Bohr^2 (a.u.) is needed
    polar_au = (polar_derivatives * BOHR_TO_ANGSTROM).ravel()
    positions_bohr = (coords / BOHR_TO_ANGSTROM).ravel()

    return red_masses, frequencies, polar_au, modes, positions_bohr, atomic_numbers, len(atomic_numbers), 0


def read_qm9s(text):
    """
    Drop-in replacement for read_gaussian: QM9S CSV text -> read_gaussian tuple.
    """
    mol = parse_qm9s_csv(text)
    return to_gaussian_tuple(mol["atomic_numbers"], mol["coords"], mol["hessian"], mol["polar_derivatives"])
