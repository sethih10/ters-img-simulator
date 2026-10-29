"""
Generates TERS data for QM9S molecules, read directly from qm9s_csv.zip (no extraction).

Same physics, constants and output (.npz per molecule) as scripts/point_spectrum_generation.py;
only the reader differs (read_qm9s instead of read_gaussian). Output files are named by the
QM9S index, e.g. 000001.npz.

Orientation: --molecule_rotation (same for all molecules), else --rotations (per-molecule table
from compute_rotations.py: deterministic, shared with other datasets), else PCA auto-rotation.

Workers = SLURM_CPUS_PER_TASK (as in point_spectrum_generation.py). --shard/--num_shards
splits the molecules over the tasks of a job array. Molecules whose .npz already exists are
skipped, so a resubmitted job resumes.

Arguments:
    zip_path: str -- path to qm9s_csv.zip
    save_path: str -- directory for the .npz files
    log_file: str -- log file, or a directory for a timestamped log file

Example:
    python qm9s_adapter/qm9s_spectrum_generation.py /scratch/phys/sin/sethih1/qm9s-raw/qm9s_csv.zip out/ logs/
"""
import argparse
import logging
import multiprocessing
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from _repo import bind_package
bind_package()

import ters_img_simulator.scripts.point_spectrum_generation as psg
from compute_rotations import load_rotations, pca_angles
from read_qm9s import parse_qm9s_csv, read_qm9s

_zip = None  # one open zip handle per worker process


def init_worker(zip_path, log_file, grid):
    global _zip
    _zip = zipfile.ZipFile(zip_path)
    psg.X_COUNT = psg.Y_COUNT = grid
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_member(member):
    """
    Reader passed to generate_ters_data: zip member -> read_gaussian tuple.
    """
    return read_qm9s(_zip.read(str(member)).decode())


def process_member(args):
    """
    Generates and saves the TERS data of one QM9S molecule. Returns True on success.
    """
    member, save_path, molecule_rotation, plot_spectrum, pca_atoms = args
    logging.info(f"Started processing file: {member.name}")

    try:
        current_rotation = molecule_rotation
        if current_rotation is None:
            mol = parse_qm9s_csv(_zip.read(str(member)).decode())
            current_rotation = pca_angles(mol["coords"], mol["atomic_numbers"], pca_atoms)

        image_data = psg.generate_ters_data(member, current_rotation, plot_spectrum, reader=read_member)
        if image_data is None:
            raise ValueError("Adapter returned no data")

        # Write to a staging directory, then rename: a killed job never leaves a truncated .npz
        partial_path = save_path / ".partial"
        psg.save_ters_data(*image_data, partial_path)
        os.replace(partial_path / f"{member.stem}.npz", save_path / f"{member.stem}.npz")
    except Exception as e:
        logging.error(f"Error processing file: {member.name} - {e}")
        return False

    logging.info(f"Finished processing file: {member.name}")
    return True


def resolve_log_file(log_file, shard):
    log_file = Path(log_file)
    if log_file.is_dir() or log_file.suffix == "":
        log_file.mkdir(parents=True, exist_ok=True)
        log_file = log_file / f"qm9s_shard{shard}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path, help="Path to qm9s_csv.zip.")
    parser.add_argument("save_path", type=Path, help="Directory to save the .npz files.")
    parser.add_argument("log_file", type=str, help="Log file, or a directory for a timestamped log file.")
    parser.add_argument("--shard", type=int, default=0, help="Index of this shard (e.g. SLURM_ARRAY_TASK_ID).")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (e.g. SLURM_ARRAY_TASK_COUNT).")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many molecules (for test runs).")
    parser.add_argument("--grid", type=int, default=psg.X_COUNT, help="Image resolution, grid x grid points over the same 18 A frame.")
    parser.add_argument(
        "--molecule_rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("PHI", "THETA", "PSI"),
        help="Euler angles in degrees, for all molecules. If omitted, --rotations or PCA-based auto-rotation is used.",
    )
    parser.add_argument("--rotations", type=Path, default=None, help="Per-molecule rotation table from compute_rotations.py.")
    parser.add_argument("--plot_spectrum", type=float, nargs='+', default=None, help="Wavenumbers where the spectrum is calculated. Default: the normal modes of the molecule.")
    parser.add_argument("--pca_atoms", choices=["all", "heavy"], default="all", help="Atoms used for the PCA normal in auto-rotation. 'heavy' falls back to all atoms if the heavy atoms define no plane.")
    args = parser.parse_args()

    log_file = resolve_log_file(args.log_file, args.shard)
    args.save_path.mkdir(parents=True, exist_ok=True)
    (args.save_path / ".partial").mkdir(exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as z:
        members = sorted(PurePosixPath(n) for n in z.namelist() if n.endswith(".csv"))

    # Interleaved: molecule size grows with the index, so every shard gets a similar mix
    members = members[args.shard::args.num_shards][:args.limit]
    todo = [m for m in members if not (args.save_path / f"{m.stem}.npz").exists()]
    print(f"{len(members)} molecules, {len(members) - len(todo)} already done, "
          f"{len(todo)} to run on {psg.cpus} workers. Log: {log_file}")

    t_0 = time.time()
    n_ok = 0
    table = load_rotations(args.rotations) if args.rotations and args.molecule_rotation is None else {}
    tasks = [(m, args.save_path, table.get(int(m.stem), args.molecule_rotation), args.plot_spectrum, args.pca_atoms) for m in todo]
    with multiprocessing.Pool(processes=psg.cpus, initializer=init_worker, initargs=(args.zip_path, log_file, args.grid)) as pool:
        for ok in tqdm(pool.imap_unordered(process_member, tasks, chunksize=4), total=len(tasks), mininterval=30):
            n_ok += ok

    print(f"Done: {n_ok} succeeded, {len(tasks) - n_ok} failed (see log) in {time.time() - t_0:.0f} s")
