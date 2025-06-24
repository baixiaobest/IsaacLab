from isaaclab.utils import configclass
from dataclasses import MISSING

@configclass
class SingleTerrainGeneratorCfg:
    """Configuration for the single terrain generator."""

    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed from the
    current NumPy's random state is used.
    """

    size: tuple[float, float] = MISSING
    """The width (along x) and length (along y) of the terrain (in m).
    """