import numpy as np
import torch
from ..consts import NON_BLOCKING, BENCHMARK
import time
torch.backends.cudnn.benchmark = BENCHMARK
import pdb

from .forgemaster_base import _Forgemaster

class ForgemasterClasswiseRandom(_Forgemaster):

    def _forge(self, client, furnace):
        """Run generalized iterative routine."""
        print(f'Starting forgeing procedure ...')

        poison_delta = furnace.initialize_poison(initializer='zero')
        if self.args.full_data:
            dataloader = furnace.trainloader
        else:
            dataloader = furnace.poisonloader

        # Create random noise
        num_classes = len(furnace.trainset.classes)
        binary_prob = torch.ones((num_classes, *furnace.trainset[0][0].shape)) * 0.5
        classwise_noise = torch.bernoulli(binary_prob)
        self.classwise_noise = classwise_noise.to(**self.setup) * (self.args.eps / 255) / furnace.ds

        for batch, example in enumerate(dataloader):
            if batch == 0:
                start = time.time()
            elif batch % 100 == 0:
                end = time.time()
                avg = (end-start)/100
                start = end
                print(f'average time per epoch: {len(dataloader) * avg}')
            self._perturb_batch(batch, example, poison_delta, furnace)

        # Return a tensor of poison_delta
        return poison_delta

    def _perturb_batch(self, batch_idx, example, poison_delta, furnace):
        if batch_idx % 50 == 0:
            print(f'[Batch idx {batch_idx}]')
        inputs, labels, ids = example
        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

        # Perturb using noise, selected based on class label, then normalize
        perturbation_tensors = self.classwise_noise[labels]
        
        # Save adversarial perturbations to poison_delta
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(ids.tolist()):
            lookup = furnace.poison_lookup.get(image_id)
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        # Return slice to CPU:
        poison_delta[poison_slices] = perturbation_tensors.detach().to(device=torch.device('cpu'))