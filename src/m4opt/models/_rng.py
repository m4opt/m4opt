"""
RNG utilities for use in the ``m4opt.models`` package for sampling.
"""

import numpy as np
from numpy.typing import NDArray


def get_rng(
    rng: np.random.Generator | int | None = None,
) -> np.random.Generator:
    """
    Return a NumPy random-number generator.

    This function accepts either an existing random-number generator, an
    integer seed, or ``None``. It provides a consistent mechanism for
    constructing reproducible random-number generators throughout the package.

    Parameters
    ----------
    rng : numpy.random.Generator, int, or None, optional
        Random-number source.

        - If a :class:`numpy.random.Generator` is provided, it is returned
          unchanged.
        - If an integer is provided, a new generator is initialized using
          that value as the random seed.
        - If ``None`` is provided, a new generator is initialized using
          NumPy's default entropy source.

    Returns
    -------
    numpy.random.Generator
        Random-number generator suitable for stochastic operations.

    Raises
    ------
    TypeError
        If ``rng`` is not a :class:`numpy.random.Generator`, an integer
        seed, or ``None``.

    Notes
    -----
    Passing an existing generator is recommended when several stochastic
    operations should share the same reproducible random stream.
    """
    if isinstance(rng, np.random.Generator):
        return rng

    if rng is None:
        return np.random.default_rng()

    # Although ``bool`` is technically a subclass of ``int`` in Python, it
    # is almost certainly a programming error if it appears here. Reject it
    # explicitly rather than silently treating ``True`` and ``False`` as the
    # integer seeds ``1`` and ``0``.
    if isinstance(rng, bool):
        raise TypeError("`rng` must be a NumPy Generator, an integer seed, or None.")

    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))

    raise TypeError("`rng` must be a NumPy Generator, an integer seed, or None.")


def get_seed_sequence(
    seed: np.random.SeedSequence | int | None = None,
) -> np.random.SeedSequence:
    """
    Return a NumPy `~numpy.random.SeedSequence`.

    Companion to :func:`get_rng`, for callers that need to *spawn* further
    independent streams (see :func:`split_root_seed`/:func:`spawn_seeds`) rather than
    draw numbers directly.

    Parameters
    ----------
    seed : numpy.random.SeedSequence, int, or None, optional
        - If a `~numpy.random.SeedSequence` is provided, it is returned unchanged.
        - If an integer is provided, it seeds a new `SeedSequence`.
        - If ``None``, a new `SeedSequence` is initialized from NumPy's default
          entropy source.

    Returns
    -------
    numpy.random.SeedSequence
    """
    if isinstance(seed, np.random.SeedSequence):
        return seed

    if seed is None:
        return np.random.SeedSequence()

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(
            "`seed` must be a NumPy SeedSequence, an integer seed, or None."
        )

    return np.random.SeedSequence(int(seed))


def split_root_seed(
    seed: np.random.SeedSequence | int | None = None,
) -> tuple[np.random.Generator, np.random.SeedSequence]:
    """
    Split a root seed into an "immediate draws" generator and a "per-event seeds" spawner.

    Used by Monte Carlo samplers that need two decorrelated streams from a single root
    seed: one to drive draws consumed immediately within the sampling call itself (how
    many events, their positions/redshifts/times), and a second to spawn independent,
    individually storable per-event seeds (see :func:`spawn_seeds`) for lazily
    regenerating each event's physical parameters later. Keeping these separate means
    replaying a single event's parameter draw can never depend on how many other
    events were sampled alongside it in the same call.

    Parameters
    ----------
    seed : numpy.random.SeedSequence, int, or None, optional
        Root seed; see :func:`get_seed_sequence`.

    Returns
    -------
    rng : numpy.random.Generator
        Generator for this call's immediate draws.
    spawn_sequence : numpy.random.SeedSequence
        Pass to :func:`spawn_seeds` to derive per-event seeds.

    See Also
    --------
    spawn_seeds : Derives the actual per-event seed array from `spawn_sequence`.
    """
    draw_sequence, spawn_sequence = get_seed_sequence(seed).spawn(2)

    return np.random.default_rng(draw_sequence), spawn_sequence


def spawn_seeds(spawn_sequence: np.random.SeedSequence, n: int) -> NDArray[np.uint64]:
    """
    Derive ``n`` independent, reproducible, individually storable seeds.

    Each returned seed is a single ``uint64``, directly usable with :func:`get_rng`
    (e.g. as ``Lightcurve.sample_parameters(rng=seed)``) to reproducibly regenerate
    that one event's parameters later without needing to replay the draws for any
    other event -- the compression trick behind only ever storing a seed per event
    rather than its sampled parameters.

    Uses `~numpy.random.SeedSequence.spawn`, which is NumPy's recommended mechanism
    for generating many independent reproducible streams from one root; a naive
    ``base_seed + i`` scheme is not guaranteed to avoid subtle stream correlations the
    way spawning is.

    Parameters
    ----------
    spawn_sequence : numpy.random.SeedSequence
        Typically the second element returned by :func:`split_root_seed`.
    n : int
        Number of seeds to derive.

    Returns
    -------
    numpy.ndarray
        ``uint64`` array of shape ``(n,)``.
    """
    return np.array(
        [
            child.generate_state(1, dtype=np.uint64)[0]
            for child in spawn_sequence.spawn(n)
        ],
        dtype=np.uint64,
    )
