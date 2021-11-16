"""Interface for poison recipes."""
from village.shop.forgemaster_functional import ForgemasterFunctional
from .forgemaster_untargeted import ForgemasterUntargeted
from .forgemaster_targeted import ForgemasterTargeted
from .forgemaster_targetedrandom import ForgemasterTargetedRandom
from .forgemaster_explosion import ForgemasterExplosion
from .forgemaster_tensorclog import ForgemasterTensorclog
from .forgemaster_functional import ForgemasterFunctional

import torch


def Forgemaster(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'grad_explosion':
        return ForgemasterExplosion(args, setup)
    elif args.recipe == 'tensorclog':
        return ForgemasterTensorclog(args, setup)
    elif args.recipe == 'untargeted':
        return ForgemasterUntargeted(args, setup)
    elif args.recipe == 'targeted':
        return ForgemasterTargeted(args, setup)
    elif args.recipe == 'targeted_random':
        return ForgemasterTargetedRandom(args, setup)
    elif args.recipe == 'functional':
        return ForgemasterFunctional(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Forgemaster']
