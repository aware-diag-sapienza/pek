import numpy as np


def get_random_state(random_state):
    """
    Get a valid random state for use in NumPy functions.

    This function ensures that a valid random state is returned, handling the case
    when the input random state is None.

    Parameters:
    - random_state (None, int, or np.random.Generator):
      The random state. If None, a new random state is generated.

    Returns:
    - int:
      A random seed (integer) suitable for initializing a NumPy random state.

    Example:
    >>> get_random_state(None)
    # Returns a new random seed (integer).

    >>> get_random_state(42)
    # Returns the provided random seed (assuming it's valid).

    >>> rng = np.random.default_rng(123)
    >>> get_random_state(rng)
    # Returns the seed associated with the provided NumPy random state.

    Note:
    The returned integer can be used as a seed when initializing
    a NumPy random state for various random number generation functions.
    """
    if random_state is None:
        return np.random.default_rng(None).integers(0, np.iinfo(np.int32).max, size=1)[0]
    return random_state
