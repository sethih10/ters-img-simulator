import numpy as np

def spectrum_real(frequencies, wavenumber_range, peak_width, lambda_0, T):
    """
    Calculates and returns the Raman line shape of every vibrational mode in a given range.

    NOTE: this used to take mode_intensities and return the summed spectrum at one tip position.
    The line shapes do not depend on the tip position, so they are now computed once per molecule
    and the spectrum at a tip position is:
        spectrum = mode_intensities[:, 1] @ spectrum_real(frequencies, wavenumber_range, ...)
    (identical to the old spectrum_real(mode_intensities, wavenumber_range, ...)).

    Arguments:
        frequencies: np.ndarray -- the vibrational frequencies of each mode (unit: cm^-1)
        wavenumber_range: np.ndarray -- the range of frequencies
        peak_width: int -- The full width at half maximum (FWHM) of the Raman vibrational peaks,
                                plotted as Lorentzian lines (unit: cm^-1)
        lambda_0: int -- incident wavelength (unit: nm)
        T: int -- Temperature, used to calculate the phonon population for each vibrational mode (unit: K)

    Returns:
        lineshape: np.ndarray -- shape (modes, wavenumbers), zero rows for modes with frequency <= 0
    """

    parm = 0.6950302506112777  # Unit convert: 1K = 0.6950302506112777 cm-1
    lineshape = np.zeros((frequencies.size, wavenumber_range.size))
    ok = frequencies > 0
    f = frequencies[ok]

    T_factor = 1 - np.exp(-f / (parm * T))
    lineshape[ok] = (
        peak_width**2 / (2 * np.pi) *
        1.0 / ((wavenumber_range[None, :] - f[:, None])**2 + 0.25 * peak_width**2) *
        ((10**7 / lambda_0 - f)**4 / (f * T_factor * 1e9))[:, None]
    )

    return lineshape
