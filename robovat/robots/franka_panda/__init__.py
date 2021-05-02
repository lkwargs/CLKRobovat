from .franka_panda_real import FrankaPandaReal
from .franka_panda_sim import FrankaPandaSim


def factory(simulator=None, config=None):
    if simulator is None:
        # Always use the default real-world Sawyer configuration.
        return FrankaPandaReal(config=None)
    else:
        return FrankaPandaSim(simulator=simulator, config=config
)
