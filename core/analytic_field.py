import numpy as np

def analytic_field(atom_polarizabilities, atoms, frequencies, E_type, atom_pos=None, R=None, tip_xyz=None, tip_width=None):
    """
    Calculates and returns intensities of each vibrational mode and polarized dipoles of each 
    atom for each vibrational mode.

    Arguments:
    atom_polarizabilities: np.ndarray -- the polarizability of each atom
    atoms: int -- the number of atoms in the molecule
    frequencies: np.ndarray -- the vibrational frequencies of each mode (unit: cm^-1)
    E_type: int -- type of the incident field (0: plane wave, 1: 2D Gaussian, 2: 3D Gaussian)
    atom_pos: np.ndarray -- the cartesian coordinates of the atoms (unit: A)
    R: np.ndarray -- the rotation matrix
    tip_xyz: np.ndarray -- position of the tip (unit: A), shape (3,); or many tips at once, shape (M, 3)
    tip_width: np.ndarray -- width of the Gaussian field (unit: A)

    Returns:
        intensity: np.ndarray -- intensity of each vibrational mode; shape (modes, 2), or (M, modes, 2) for M tips
        dipole: np.ndarray -- polarized dipole of each atom for each mode; shape (modes, 3*atoms); None for M tips (not computed)
    """
    # One tip (3,) or many tips (M, 3), e.g. a whole row of the image grid in one call
    many_tips = tip_xyz is not None and np.ndim(tip_xyz) == 2
    tips = np.atleast_2d(tip_xyz) if tip_xyz is not None else np.zeros((1, 3))
    E_local = np.zeros((tips.shape[0], atoms, 3))

    match E_type:
        case 0: # plane wave
            theta = 0/180*np.pi #Angle between incident direction and z-axis
            phi = 0/180*np.pi   #Angle of incident direction around z-axis
            E_local[:] = \
                np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

        case 1: # 2D Gaussian
            atom_tip_distance = np.dot(atom_pos, R)[None, :, :] - tips[:, None, :]  # (tips, atoms, 3)
            E_local[:] = \
                np.exp(-4*np.log(2)*np.sum(((atom_tip_distance[..., 0:2])/(tip_width[0:2]))**2, axis=-1))[..., None]*np.array([0, 0, 1])

        case 2: # 3D Gaussian
            atom_tip_distance = np.dot(atom_pos, R)[None, :, :] - tips[:, None, :]  # (tips, atoms, 3)
            E_local[:] = \
                np.exp(-4*np.log(2)*np.sum(((atom_tip_distance)/(tip_width))**2, axis=-1))[..., None]*(atom_tip_distance/np.linalg.norm(atom_tip_distance, axis=-1, keepdims=True))

        case _:
            raise ValueError(f"Function analytic field only supports field types 0, 1, 2. Input values was {E_type}")

    if not many_tips:
        E_local = E_local[0].ravel()  # (3*atoms,), same layout as before

        # Calculating mode intensities and dipoles
        mode_intensities = np.zeros((3*atoms-6, 2))
        dipoles = np.tensordot(atom_polarizabilities, E_local, axes=([2], [0]))
        E_scattering = np.einsum('i,ij->j', E_local, dipoles.T)

        mode_intensities[:, 0] = frequencies
        mode_intensities[:, 1] = E_scattering**2

        return mode_intensities, dipoles

    # Many tips: same E . (alpha E), using only each atom's 3x3 block (no atom-atom coupling, so the rest of alpha is zero)
    n_modes = atom_polarizabilities.shape[0]
    blocks = np.stack([atom_polarizabilities[:, 3*i:3*i+3, 3*i:3*i+3] for i in range(atoms)], axis=1)  # (modes, atoms, 3, 3)
    EE = (E_local[..., :, None] * E_local[..., None, :]).reshape(tips.shape[0], 9*atoms)               # (tips, 9*atoms)
    E_scattering = EE @ blocks.reshape(n_modes, 9*atoms).T                                             # (tips, modes)

    mode_intensities = np.zeros((tips.shape[0], 3*atoms-6, 2))
    mode_intensities[..., 0] = frequencies
    mode_intensities[..., 1] = E_scattering**2

    # Dipoles are not needed here. If needed, e.g. for a far-field Green's function:
    # dipoles = np.einsum('knab,tnb->tkna', blocks, E_local).reshape(tips.shape[0], n_modes, 3*atoms)  # (tips, modes, 3*atoms)
    return mode_intensities, None