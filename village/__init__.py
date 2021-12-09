"""Library of simple routines."""

from village.shop import Forgemaster
from village.materials import Furnace, CIFAR10, CIFAR20, CIFAR100
from village.clients import Client

from .options import options



__all__ = ['Client', 'Forgemaster', 'Furnace', 'CIFAR10', 'CIFAR20', 'CIFAR100', 'options']
