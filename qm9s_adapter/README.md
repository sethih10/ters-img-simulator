# QM9S adapter

Runs the TERS simulator on the [QM9S](https://doi.org/10.6084/m9.figshare.24235333) dataset
(`/scratch/phys/sin/sethih1/qm9s-raw/qm9s_csv.zip`, 129,817 molecules) without Gaussian `.fchk` files.

| File | Purpose |
|---|---|
| `read_qm9s.py` | The adapter: QM9S CSV → the exact tuple `core/read_gaussian.py` returns |
| `qm9s_spectrum_generation.py` | Batch driver: reads straight from the zip, shards, resumes, writes one `.npz` per molecule |
| `compute_rotations.py` | Computes each molecule's orientation once (PCA) and writes the rotation table; `pca_angles()` for other datasets |
| `run_qm9s.sbatch` | SLURM job: array of 8 × 32 cores on `batch-milan` |
| `_repo.py` | Makes `import ters_img_simulator` resolve to this checkout, whatever the directory is called |

Checks and plots live in `dev/` at the repository root. It is git-ignored, so it's local only:
`dev/validate_adapter.py` (physics) and `dev/check_frames.py` (orientation and frames).

## Why an adapter is needed

The simulator reads Gaussian `.fchk` files. QM9S holds the same physics, but stored differently:

| Simulator needs (`read_gaussian`) | QM9S CSV provides | Adapter does |
|---|---|---|
| coordinates, bohr | coordinates, Å | Å → bohr |
| `Polarizability Derivatives`, a.u. (bohr²), per atom `[d/dx, d/dy, d/dz] × (xx, xy, yy, xz, yz, zz)` | same layout, Bohr³/Å | ×0.529177 (Bohr³/Å → bohr²) |
| `Vib-E2`: frequencies (cm⁻¹) and reduced masses | — (not stored) | computed from the Hessian |
| `Vib-Modes`: normalized Cartesian modes | — (not stored) | computed from the Hessian |
| — | Hessian, 3n×3n, eV/Å² | input to the vibrational analysis |

So the adapter redoes Gaussian's harmonic vibrational analysis from the QM9S Hessian. The rest of
the simulator (`load_molecule`, `analytic_field`, `spectrum_real`) runs unchanged.

## How the vibrational analysis is done (`normal_modes`)

1. Masses: most abundant isotopes (H 1.00783, C 12, N 14.00307, O 15.99491, F 18.99840 amu), the
   same masses Gaussian uses (`Vib-AtMass`).
2. Mass-weight the Hessian: `H_mw = H / sqrt(m_i m_j)`.
3. Build the 3 translations and 3 rotations (about the centre of mass) in mass-weighted coordinates
   and take an orthonormal basis `D` of the remaining 3n−6 dimensional vibrational space.
   Like Gaussian, this removes translations and rotations exactly, instead of relying on 6 near-zero
   eigenvalues.
4. Diagonalize `Dᵀ H_mw D` → eigenvalues λ, mass-weighted modes `l_mw = D v`.
5. Frequencies `ν = sign(λ) · sqrt|λ| · 521.47` cm⁻¹ (with λ in eV/(Å² amu)).
6. Gaussian's conventions: `l_cart = l_mw / sqrt(m)`, reduced mass `μ = 1 / Σ l_cart²`,
   stored mode `= l_cart · sqrt(μ)` (unit length).
   `load_molecule` then uses `mode / sqrt(μ) = l_cart`, which is what enters the Raman tensor.

## How the QM9S layout was determined

The readme gives the units but not the order of the 18 polarizability-derivative numbers per atom.
The two candidates are direction-major `[d][component]` and component-major `[component][d]`.
For the right order, the rotational sum rule holds:

    Σ_i (ω × r_i) · ∂α/∂r_i = Ωα − αΩ,    Ωv = ω × v

This uses the static polarizability α from the same CSV. The direction-major order satisfies it to
~1e-3 (relative 4e-4). The other order is off by ~100. The fact that it holds at all also confirms
Bohr³/Å with coordinates in Å. Direction-major with (xx, xy, yy, xz, yz, zz) is exactly
Gaussian's `.fchk` order.

## Validation (`python dev/validate_adapter.py`)

**1. Round trip against Gaussian.** Take Gaussian `.fchk` files (`data_files/all_group`), convert
their raw Hessian, coordinates and polarizability derivatives to QM9S units and layout, and run them
through the adapter. Then compare with what Gaussian itself wrote, and with the full simulator output
(32×32 grid, 100–4000 cm⁻¹):

| quantity | max deviation (5 molecules) |
|---|---|
| masses | 0 |
| frequencies | 2.5e-5 cm⁻¹ |
| reduced masses (relative) | 9e-7 |
| mode overlap \|⟨Gaussian\|adapter⟩\| (non-degenerate) | 1.000000 |
| TERS spectra, max\|ΔS\| / max\|S\| | 3e-6 |

**2. QM9S self-consistency** (500 random molecules, plus a full pass over all 129,817):
- translational sum rule Σᵢ ∂α/∂rᵢ = 0: residual ≤ 2e-7 relative
- rotational sum rule: residual ≤ 4e-4 relative
- CH₄: 1346, 1562, 3031, 3136 cm⁻¹ (B3LYP-level literature values), which confirms the Hessian is in eV/Å²
- no imaginary frequencies in any of the 129,817 molecules; 3–29 atoms (mean 18)

## Limitations

- **Linear molecules are skipped** (6 in QM9S: C#C, C#N, C#CC#C, C#CC#N, N#CC#CC#N, N#CC#CC#CC#N).
  They have 3n−5 modes, and `load_molecule`/`analytic_field` assume 3n−6. They are logged as errors.
- **Level of theory**: QM9S is B3LYP/def2-TZVP. The existing `.fchk` data in `data_files/all_group`
  is RPBE-PBE/6-31G(d). Frequencies and intensities therefore differ systematically between the two
  sets, so don't mix them in one training set without accounting for this.

## Running

From the repository root. Test first, on a few molecules:

    SLURM_CPUS_PER_TASK=4 python qm9s_adapter/qm9s_spectrum_generation.py \
        /scratch/phys/sin/sethih1/qm9s-raw/qm9s_csv.zip /tmp/qm9s_test /tmp/qm9s_test_logs --limit 8

Rotation table, once (about a minute on 8 cores; see "Orientation" below):

    python qm9s_adapter/compute_rotations.py /scratch/phys/sin/sethih1/qm9s-raw/qm9s_csv.zip \
        /scratch/phys/sin/sethih1/data_files/qm9s_rotations.npz

Full dataset, oriented by the table:

    sbatch qm9s_adapter/run_qm9s.sbatch --rotations /scratch/phys/sin/sethih1/data_files/qm9s_rotations.npz

- **Jobs**: an array of 8 × 32 cores on `batch-milan` (`--shard/--num_shards` from the array).
  A whole 128-core node would have to wait until it is completely free, which can take days;
  32-core pieces fit into partly used nodes and start at once. Triton rejects `--mem=0`, so memory
  is requested explicitly (~100 MB per worker is enough).
- **Time**: the full run (job 21237186, 2026-10-27) took ~5 min per task. All 8 tasks started at once.
- **Disk**: output is float64 `256×256×(3n−6)` → **3.0 TB** for 129,811 files. Check the `sin`
  group quota first with `lfs quota -h -g sin /scratch`.
- **Resume**: finished molecules are skipped, and files are written to `.partial/` and then renamed,
  so a killed job never leaves a truncated `.npz`. Resubmit to continue.
- `OMP_NUM_THREADS=1` in the job script matters: without it, every worker starts its own numpy
  threads and they oversubscribe the cores.

Files are named by QM9S index (`000001.npz`). They have the same fields as
`scripts/point_spectrum_generation.py` writes: `atom_pos`, `atomic_numbers`, `x_pos`, `y_pos`,
`frequencies`, `spectrums`.

Orientation, in order of precedence:
1. `--molecule_rotation φ θ ψ`: the same rotation for every molecule.
2. `--rotations table.npz`: per-molecule rotations from the table.
3. Otherwise: PCA auto-rotation, computed on the fly.

## Orientation, and using the same frame for other datasets (e.g. QH9)

**The image frame.**
- Atoms are rotated as `atom_pos = coordinates @ R`, with `R = rotation_matrix(φ, θ, ψ)` from
  `core/load_molecule.py`.
- The angles come from the auto-rotation: the PCA plane normal (all atoms, or heavy atoms with
  `--pca_atoms heavy`) goes to +z, using `normal_to_zyz_ters`.
- The tip is above the molecule, along +z, and the grid is `x_pos` × `y_pos`.
- There is no translation, so the molecule is not centred on the grid.

**The rotation table** (`compute_rotations.py` → `qm9s_rotations.npz`) holds one row per QM9S
molecule:

| Field | Meaning |
|---|---|
| `qm9s_index`, `qm9_index`, `smiles` | identity: the QM9S file number, the original QM9 number, and the SMILES |
| `euler_angles` | (φ, θ, ψ) in degrees, as taken by `--molecule_rotation` |
| `rotation_matrix` | 3×3 `R`, image frame = `coordinates @ R` |
| `centroid` | centroid of the QM9S coordinates (Å), to place copies from other datasets |

**Why use a table instead of recomputing the PCA.**
- The PCA is not reproducible for molecules without a well-defined plane. 20 QM9S molecules (CH₄
  and other symmetric ones) have an exactly degenerate plane, so their orientation depends on
  floating-point noise. Recomputing in a different Python/numpy environment turned CH₄ into a
  different orientation.
- 121 more molecules are nearly degenerate (relative gap < 1e-3), so a slightly different geometry
  from another dataset gives them a different plane.
- All other molecules agree to about 1e-14.

Computing once and sharing the numbers makes the TERS images and anything built on them
(e.g. STM from QH9) agree exactly.

**How the datasets relate** (`dev/check_frames.py`, 295 random molecules):
- **Same atom order:** QM9S, QM9 and QH9 list the same atoms in the same order for every molecule.
- **QM9S and QM9 share a frame:** rotation median 0.0007°, centroid shift 1e-5 Å. QM9S re-optimised
  the geometry (B3LYP/def2-TZVP): median 0.017 Å, 99th percentile 0.5 Å. An apparent rotation of up to
  13° appears only where the conformer changed.
- **QH9 is exactly QM9's geometry** (5e-5 Å) and orientation, but **shifted by about 16 Å**.

So the table's `R` applies directly to QH9 coordinates. Only the translation needs fixing:

    pos_image = (xyz - xyz.mean(axis=0) + centroid) @ R        # xyz: QH9 coordinates (A), same atom order

This matches the TERS image's `atom_pos` up to the QM9S re-optimisation. Look up the row by
`qm9_index` (QM9 number). For QH9, use the QH9↔QM9 map of the QH9 pipeline.

**Using our orientation code directly.** Another pipeline can call the same function instead of
reading the table:

    from compute_rotations import pca_angles            # qm9s_adapter/ on sys.path
    from ters_img_simulator.core.load_molecule import rotation_matrix
    R = rotation_matrix(*pca_angles(xyz, Z))             # pos_image = xyz @ R (plus the shift above)

This gives the same convention, but recomputes on that dataset's geometry. It agrees with the table
except for the ~140 (nearly) degenerate molecules above, so use the table when images must match.

**Conventions that are easy to get wrong:**
- **Row vectors.** `R` acts on row vectors: `coordinates @ R`. Code that rotates as `xyz @ M.T`
  needs `M = R.T`.
- **No mirroring.** `R` is a proper rotation (det +1), so the molecule is never mirrored and the
  tip always sees the same face.
- **Frames differ between datasets.** This frame (all-atom PCA, uncentred) differs from other TERS
  datasets that used heavy-atom PCA and centring. Don't mix images from different frame
  conventions without re-aligning.
- **Older datasets.** The datasets generated before the table existed (2026-10-27) used on-the-fly
  PCA. For the ~20 degenerate molecules they may not match the table. Regenerate those with
  `--rotations` if exact agreement matters.
- **Exact re-simulation.** `--molecule_rotation <euler_angles>` (or `--rotations`) reproduces stored
  spectra bit for bit.

## Change to the core code

- `load_molecule(...)` and `generate_ters_data(...)` take an optional `reader` argument (default
  `read_gaussian`), so the adapter plugs in without copying any simulation code.
- The rotation-matrix formula moved unchanged into `rotation_matrix(phi, theta, psi)` in
  `core/load_molecule.py`, so the table and the simulator use the same code.

Behaviour for `.fchk` input is unchanged.
