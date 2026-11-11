# QM9S Adapter

Runs the TERS simulator on the [QM9S](https://doi.org/10.6084/m9.figshare.24235333) dataset
(129,817 molecules, distributed as `qm9s_csv.zip`) without Gaussian `.fchk` files.

QM9S provides coordinates, the Hessian and polarizability derivatives for every molecule. The
adapter converts these into exactly the data the simulator reads from a `.fchk` file, so the
physics (`load_molecule`, `analytic_field`, `spectrum_real`) runs unchanged.

## Contents

| File | Purpose |
| --- | --- |
| `reader.py` | Parses a QM9S molecule and returns the same tuple as `core/read_gaussian.py` |
| `generate.py` | Batch driver: reads directly from the zip archive, supports sharding and resuming, writes one `.npz` per molecule and a run manifest per shard |
| `rotations.py` | Computes the orientation of every molecule once and writes the rotation table; `pca_angles()` applies the same convention to other datasets |

## Usage

All commands are run from the repository root.

### 1. Rotation table

Computed once per dataset and shared by every run and every derived dataset (see
[Coordinate frame](#coordinate-frame)):

```bash
python qm9s_adapter/rotations.py qm9s_csv.zip qm9s_rotations.npz --pca_atoms heavy
```

### 2. TERS data

```bash
python qm9s_adapter/generate.py qm9s_csv.zip <output_dir> <log_dir> \
    --grid 64 --rotations qm9s_rotations.npz --center heavy
```

| Option | Description |
| --- | --- |
| `--grid G` | Image resolution, `G × G` points over the 18 Å scan (default 256) |
| `--rotations FILE` | Per-molecule orientation from the rotation table (recommended) |
| `--molecule_rotation φ θ ψ` | One fixed orientation for every molecule, in degrees |
| `--pca_atoms {all,heavy}` | Atoms used for on-the-fly PCA orientation when neither option above is given |
| `--center {none,all,heavy}` | Point moved to the grid centre before rotation: none (QM9S origin), centroid of all atoms, or centroid of the non-hydrogen atoms |
| `--plot_spectrum ν₁ ν₂ …` | Evaluate at these wavenumbers (cm⁻¹) instead of at the normal modes |
| `--shard k --num_shards N` | Process every N-th molecule starting at k, e.g. one shard per task of a job array |
| `--limit n` | Process at most n molecules (for test runs) |

Orientation precedence: `--molecule_rotation`, then `--rotations`, then on-the-fly PCA.

**Parallelism.** One worker process per core, taken from `SLURM_CPUS_PER_TASK` when set. Set
`OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) so that the workers do not each start their
own numpy threads. With 8 shards of 32 cores, the full dataset takes about 1 minute per shard at
64 px and 5 minutes at 256 px; memory use is about 100 MB per worker.

**Resuming.** Molecules whose `.npz` already exists are skipped. Files are written to `.partial/`
and renamed when complete, so an interrupted run never leaves a truncated file; rerunning the same
command continues where it stopped.

**Disk space.** Output is float64, `G × G × (3n − 6)` per molecule: about 3.0 TB for the full
dataset at 256 px and 0.2 TB at 64 px.

### Output

One `<qm9s_index>.npz` per molecule (e.g. `000123.npz`), with the keys written by
`scripts/point_spectrum_generation.py` (`atom_pos`, `atomic_numbers`, `x_pos`, `y_pos`,
`frequencies`, `spectrums`; see the top-level README for shapes and axis order) and two more:

| Key | Shape | Description |
| --- | --- | --- |
| `euler_angles` | `(3,)` | Orientation used, (φ, θ, ψ) in degrees |
| `center` | `(3,)` | Point moved to the grid centre, in Å in the QM9S frame |

Each shard also writes `run_manifest_shard<k>.json` with the command line, git commit and
working-tree state, all simulation parameters, the rotation table and its SHA-256 checksum, and the
software environment.

## Coordinate frame

### Image frame

```text
atom_pos = (coords − c) @ R,    R = rotation_matrix(φ, θ, ψ)    (core/load_molecule.py)
```

- `R` turns the PCA plane normal of the molecule to +z (`normal_to_zyz_ters`); the tip scans the
  plane 3 Å above the highest atom.
- The scan grid is centred on the origin. `c` is set by `--center`: 0, the centroid of all atoms,
  or the centroid of the non-hydrogen atoms.
- Without centring, molecules sit wherever the QM9S coordinates place them. In the uncentred
  dataset 62 molecules extend beyond the ±9 Å scan and are cropped. With `--center heavy` the
  largest extent is about 5.8 Å.

### Rotation table

`rotations.py` writes one row per QM9S molecule:

| Field | Description |
| --- | --- |
| `qm9s_index`, `qm9_index`, `smiles` | QM9S file number, original QM9 number and SMILES |
| `euler_angles` | (φ, θ, ψ) in degrees, as accepted by `--molecule_rotation` |
| `rotation_matrix` | 3 × 3 matrix `R`; image frame = `coords @ R` |
| `centroid` | Centroid of all atoms (Å): `c` for `--center all` |
| `heavy_centroid` | Centroid of the non-hydrogen atoms (Å): `c` for `--center heavy` |
| `pca_atoms` | Atoms used for the PCA normal (`all` or `heavy`) |

Orientations are taken from the table rather than recomputed because PCA is not reproducible:

- 20 molecules (CH₄ and other symmetric molecules) have an exactly degenerate plane, so their
  orientation is determined by floating-point noise.
- 121 molecules are nearly degenerate (relative eigenvalue gap < 10⁻³), so small geometry
  differences between datasets change their plane.
- The sign of a PCA axis depends on the numpy/LAPACK build. Recomputing the table in a different
  environment reproduced 255 of 300 sampled molecules and turned the other 45 by exactly 180°.

Using the stored values makes the TERS images and every dataset aligned to them agree exactly.

### Relation to QM9 and QH9

Measured on 295 randomly sampled molecules:

- QM9S, QM9 and QH9 list the same atoms in the same order.
- QM9S is in the QM9 frame (median rotation 0.0007°, centroid shift 10⁻⁵ Å). QM9S re-optimised
  the geometries at B3LYP/def2-TZVP: median atom displacement 0.017 Å, 99th percentile 0.5 Å.
- QH9 has the QM9 geometry and orientation (to 5 × 10⁻⁵ Å), translated by about 16 Å.

QH9 (or QM9) coordinates are therefore brought into the image frame with the table values:

```python
pos_image = (xyz - xyz.mean(axis=0) + centroid - c) @ R   # xyz: QH9 coordinates (Å), same atom order
                                                          # c: 0, centroid or heavy_centroid
```

The result agrees with `atom_pos` up to the QM9S re-optimisation. Rows are matched by `qm9_index`.

The same convention can be applied to a new geometry without the table. It then agrees with the
table except for the roughly 140 (nearly) degenerate molecules listed above:

```python
from ters_img_simulator.qm9s_adapter.rotations import pca_angles
from ters_img_simulator.core.load_molecule import rotation_matrix

R = rotation_matrix(*pca_angles(xyz, atomic_numbers, "heavy"))
```

### Conventions

- **Row vectors.** `R` acts on row vectors (`coords @ R`). Code that rotates column vectors
  (`xyz @ M.T`) needs `M = R.T`.
- **Proper rotations.** `det R = +1`, so molecules are never mirrored. Which face of a non-planar
  molecule points to the tip follows from the arbitrary sign of the PCA normal; it is a fixed
  convention per molecule, not a physical choice.
- **Different frames are not interchangeable.** All-atom PCA, heavy-atom PCA, centred and
  uncentred datasets use different frames. Data from different frames must be re-aligned before
  being combined; `euler_angles`, `center` and the run manifest identify the frame of a dataset.
- **Exact re-simulation.** `--rotations` (or `--molecule_rotation` with the stored angles)
  reproduces stored spectra bit for bit in the same software environment.

## Method

### Input conversion

| Simulator input (`read_gaussian`) | QM9S provides | Conversion |
| --- | --- | --- |
| Coordinates (bohr) | Coordinates (Å) | Å → bohr |
| Polarizability derivatives (a.u., bohr²), per atom `[d/dx, d/dy, d/dz] × (xx, xy, yy, xz, yz, zz)` | Same layout (bohr³/Å) | × 0.529177 |
| `Vib-E2`: frequencies (cm⁻¹) and reduced masses | Not stored | Computed from the Hessian |
| `Vib-Modes`: normalised Cartesian modes | Not stored | Computed from the Hessian |
| — | Hessian, 3n × 3n (eV/Å²) | Input to the vibrational analysis |

### Vibrational analysis

`normal_modes` repeats the harmonic analysis of Gaussian:

1. Masses of the most abundant isotopes (H 1.00783, C 12, N 14.00307, O 15.99491, F 18.99840 amu),
   as used by Gaussian (`Vib-AtMass`).
2. Mass-weighted Hessian `H_mw = H / sqrt(m_i m_j)`.
3. Translations and rotations (about the centre of mass) are projected out exactly: an orthonormal
   basis `D` of the 3n − 6 dimensional vibrational subspace is built from their complement.
4. Diagonalisation of `Dᵀ H_mw D` gives eigenvalues λ and mass-weighted modes `l_mw = D v`.
5. Frequencies `ν = sign(λ) · sqrt|λ| · 521.47` cm⁻¹ (λ in eV/(Å² amu)).
6. Gaussian normalisation: `l_cart = l_mw / sqrt(m)`, reduced mass `μ = 1 / Σ l_cart²`, stored
   mode `l_cart · sqrt(μ)` (unit length). `load_molecule` recovers `l_cart`, which enters the
   Raman tensor.

### Polarizability-derivative layout

The QM9S documentation gives the units but not the order of the 18 values per atom. Of the two
candidates, direction-major `[d][component]` and component-major `[component][d]`, only the
correct one satisfies the rotational sum rule

```text
Σ_i (ω × r_i) · ∂α/∂r_i = Ωα − αΩ,    Ωv = ω × v
```

with the static polarizability α from the same file. Direction-major order satisfies it to a
relative 4 × 10⁻⁴; component-major order is off by a factor of about 100. This also confirms the
units (bohr³/Å with coordinates in Å). Direction-major order with components
(xx, xy, yy, xz, yz, zz) is the order of Gaussian `.fchk` files.

## Validation

**Round trip against Gaussian.** Hessians, coordinates and polarizability derivatives from
Gaussian `.fchk` files were converted to the QM9S units and layout, passed through the adapter,
and compared with the values Gaussian wrote and with the full simulator output (32 × 32 grid,
100–4000 cm⁻¹):

| Quantity | Maximum deviation (5 molecules) |
| --- | --- |
| Masses | 0 |
| Frequencies | 2.5 × 10⁻⁵ cm⁻¹ |
| Reduced masses (relative) | 9 × 10⁻⁷ |
| Mode overlap \|⟨Gaussian\|adapter⟩\| (non-degenerate modes) | 1.000000 |
| TERS spectra, max\|ΔS\| / max\|S\| | 3 × 10⁻⁶ |

**Self-consistency of QM9S** (500 random molecules and a full pass over all 129,817):

- Translational sum rule Σᵢ ∂α/∂rᵢ = 0: relative residual ≤ 2 × 10⁻⁷.
- Rotational sum rule: relative residual ≤ 4 × 10⁻⁴.
- CH₄ frequencies 1346, 1562, 3031 and 3136 cm⁻¹ agree with B3LYP literature values, confirming
  that the Hessian is in eV/Å².
- No imaginary frequencies in any molecule; 3–29 atoms per molecule (mean 18).

## Limitations

- **Linear molecules are not supported.** The 6 linear molecules in QM9S (C#C, C#N, C#CC#C,
  C#CC#N, N#CC#CC#N, N#CC#CC#CC#N) have 3n − 5 modes, whereas `load_molecule` and
  `analytic_field` assume 3n − 6. They are reported as errors and skipped.
- **Level of theory.** QM9S is computed at B3LYP/def2-TZVP. Data generated from `.fchk` files at a
  different level of theory (e.g. RPBE-PBE/6-31G(d)) differs systematically in frequencies and
  intensities and should not be combined without accounting for this.

## Changes to the core simulator

- `load_molecule()` and `generate_ters_data()` accept an optional `reader` argument (default
  `read_gaussian`), through which the adapter supplies its data without duplicating simulation code.
- The rotation-matrix formula was moved unchanged into `rotation_matrix(phi, theta, psi)` in
  `core/load_molecule.py`, so the rotation table and the simulator share one implementation.

Results for `.fchk` input are unchanged.
