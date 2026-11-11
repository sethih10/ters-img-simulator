"""
Generates TERS data for QM9S molecules, read directly from qm9s_csv.zip.

Uses the physics and constants of scripts/point_spectrum_generation.py with the QM9S reader. Writes
one <qm9s_index>.npz per molecule and one run_manifest_shard<k>.json per shard. Besides the keys
of point_spectrum_generation.py, each file stores euler_angles (degrees) and center (Å), with
atom_pos = (coords - center) @ rotation_matrix(*euler_angles).

Orientation: --molecule_rotation, else --rotations (table from rotations.py), else PCA.
Molecules whose output already exists are skipped, so an interrupted run can be resumed.

Usage:
    python qm9s_adapter/generate.py qm9s_csv.zip <output_dir> <log_dir> \\
        --grid 64 --rotations qm9s_rotations.npz --center heavy
"""
import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm import tqdm

import ters_img_simulator.scripts.point_spectrum_generation as psg
from ters_img_simulator.qm9s_adapter.reader import parse_csv, to_gaussian_format
from ters_img_simulator.qm9s_adapter.rotations import load_table, molecule_center, pca_angles

TIP_HEIGHT = 3.0  # Å above the highest atom, as set in psg.generate_ters_data

_zip = None  # zip archive, opened once per worker process
_grid = None


def _init_worker(zip_path, log_file, grid):
    global _zip, _grid
    _zip, _grid = zipfile.ZipFile(zip_path), grid
    psg.X_COUNT = psg.Y_COUNT = grid
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _simulate(task):
    """
    Simulates and saves one molecule. Returns True on success.
    """
    member, out_dir, rotation, plot_spectrum, pca_atoms, center_mode = task
    logging.info(f"Started processing file: {member.name}")
    try:
        mol = parse_csv(_zip.read(str(member)).decode())
        if rotation is None:
            rotation = pca_angles(mol["coords"], mol["atomic_numbers"], pca_atoms)
        center = molecule_center(mol["coords"], mol["atomic_numbers"], center_mode)

        # Translation moves the molecule on the grid; modes and polarizability derivatives are unchanged
        def reader(_):
            return to_gaussian_format(mol["atomic_numbers"], mol["coords"] - center, mol["hessian"], mol["polar_derivatives"])

        atom_pos, atomic_numbers, x_pos, y_pos, frequencies, spectrums, _ = \
            psg.generate_ters_data(member, rotation, plot_spectrum, reader=reader)
        if spectrums.shape[:2] != (_grid, _grid):
            raise RuntimeError(f"Image is {spectrums.shape[:2]}, expected {_grid} x {_grid}")

        # Write, then rename: an interrupted run never leaves a truncated file
        partial = out_dir / ".partial" / f"{member.stem}.npz"
        np.savez(partial, atom_pos=atom_pos, atomic_numbers=atomic_numbers, x_pos=x_pos, y_pos=y_pos,
                 frequencies=frequencies, spectrums=spectrums,
                 euler_angles=np.asarray(rotation, dtype=float), center=center)
        os.replace(partial, out_dir / partial.name)
    except Exception as e:
        logging.exception(f"Error processing file: {member.name} - {e}")
        return False
    logging.info(f"Finished processing file: {member.name}")
    return True


def _log_path(log_file, shard):
    log_file = Path(log_file)
    if log_file.is_dir() or not log_file.suffix:
        log_file = log_file / f"qm9s_shard{shard}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(args, log_file, n_molecules, n_ok, n_failed, seconds):
    """
    Records the code version, parameters, inputs and environment that produced this shard.
    """
    def git(*cmd):
        return subprocess.run(["git", "-C", str(Path(__file__).parent), *cmd], capture_output=True, text=True)

    commit = git("rev-parse", "HEAD").stdout.strip() or None
    table_pca_atoms = None
    if args.rotations:
        with np.load(args.rotations) as table:
            table_pca_atoms = str(table["pca_atoms"]) if "pca_atoms" in table.files else None
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "git_commit": commit,
        "git_dirty": git("diff", "--quiet", "HEAD").returncode != 0 if commit else None,
        "shard": args.shard, "num_shards": args.num_shards,
        "molecules": n_molecules, "succeeded": n_ok, "failed": n_failed, "seconds": round(seconds, 1),
        "log_file": str(log_file),
        "input": str(args.zip_path),
        "rotation": {
            "molecule_rotation": args.molecule_rotation,
            "table": str(args.rotations) if args.rotations else None,
            "table_sha256": _sha256(args.rotations) if args.rotations else None,
            "table_pca_atoms": table_pca_atoms,
            "fallback_pca_atoms": args.pca_atoms,
        },
        "center": args.center, "grid": args.grid, "plot_spectrum": args.plot_spectrum,
        "simulator": {
            "X_WIDTH_A": psg.X_WIDTH, "Y_WIDTH_A": psg.Y_WIDTH,
            "TIP_WIDTH_A": np.asarray(psg.TIP_WIDTH).tolist(), "TIP_HEIGHT_A": TIP_HEIGHT,
            "E_TYPE": psg.E_TYPE, "PEAK_WIDTH_cm-1": psg.PEAK_WIDTH, "LAMBDA_0_nm": psg.LAMBDA_0,
            "T_K": psg.T, "CUTOFF_FREQUENCY_cm-1": psg.CUTOFF_FREQUENCY,
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    path = args.save_path / f"run_manifest_shard{args.shard}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip_path", type=Path, help="QM9S archive (qm9s_csv.zip).")
    parser.add_argument("save_path", type=Path, help="Output directory for the .npz files.")
    parser.add_argument("log_file", help="Log file, or a directory for a timestamped log file.")
    parser.add_argument("--grid", type=int, default=psg.X_COUNT, help=f"Grid points per axis over the {psg.X_WIDTH} Å scan.")
    parser.add_argument("--molecule_rotation", type=float, nargs=3, metavar=("PHI", "THETA", "PSI"),
                        help="Euler angles in degrees, used for every molecule.")
    parser.add_argument("--rotations", type=Path, help="Rotation table from rotations.py.")
    parser.add_argument("--pca_atoms", choices=["all", "heavy"], default="all",
                        help="Atoms used for PCA orientation when no rotation is given.")
    parser.add_argument("--center", choices=["none", "all", "heavy"], default="none",
                        help="Point moved to the grid centre: origin, centroid of all atoms or of the non-hydrogen atoms.")
    parser.add_argument("--plot_spectrum", type=float, nargs="+", help="Wavenumbers (cm⁻¹); default: the normal modes.")
    parser.add_argument("--shard", type=int, default=0, help="Index of this shard.")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards.")
    parser.add_argument("--limit", type=int, help="Process at most this many molecules.")
    args = parser.parse_args()

    log_file = _log_path(args.log_file, args.shard)
    (args.save_path / ".partial").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path) as z:
        members = sorted(PurePosixPath(n) for n in z.namelist() if n.endswith(".csv"))

    # Interleaved shards: molecule size grows with the index, so every shard gets a similar mix
    members = members[args.shard::args.num_shards][:args.limit]
    todo = [m for m in members if not (args.save_path / f"{m.stem}.npz").exists()]
    print(f"{len(members)} molecules, {len(members) - len(todo)} already done, "
          f"{len(todo)} to run on {psg.cpus} workers. Log: {log_file}")

    table = load_table(args.rotations) if args.rotations and args.molecule_rotation is None else {}
    missing = [m.stem for m in todo if table and int(m.stem) not in table]
    if missing:
        print(f"Warning: {len(missing)} molecules are not in the rotation table and use PCA, e.g. {missing[:5]}")

    tasks = [(m, args.save_path, table.get(int(m.stem), args.molecule_rotation), args.plot_spectrum,
              args.pca_atoms, args.center) for m in todo]
    start, n_ok = time.time(), 0
    with multiprocessing.Pool(psg.cpus, initializer=_init_worker, initargs=(args.zip_path, log_file, args.grid)) as pool:
        for ok in tqdm(pool.imap_unordered(_simulate, tasks, chunksize=4), total=len(tasks), mininterval=30):
            n_ok += ok

    seconds = time.time() - start
    manifest = _write_manifest(args, log_file, len(members), n_ok, len(tasks) - n_ok, seconds)
    print(f"Done: {n_ok} succeeded, {len(tasks) - n_ok} failed (see log) in {seconds:.0f} s. Manifest: {manifest}")


if __name__ == "__main__":
    main()
