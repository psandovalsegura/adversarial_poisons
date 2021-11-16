
import torch
from ..consts import NON_BLOCKING, BENCHMARK
import time
from recoloradv.utils import get_attack_from_name
from recoloradv.mister_ed.utils import pytorch_utils as utils
torch.backends.cudnn.benchmark = BENCHMARK
import pdb


torch.backends.cudnn.benchmark = BENCHMARK

from .forgemaster_base import _Forgemaster

class ForgemasterFunctional(_Forgemaster):

    def _forge(self, client, furnace):
        """Run generalized iterative routine."""
        print(f'Starting forgeing procedure ...')

        poison_delta = furnace.initialize_poison(initializer='zero')
        if self.args.full_data:
            dataloader = furnace.trainloader
        else:
            dataloader = furnace.poisonloader

        for batch, example in enumerate(dataloader):
            if batch == 0:
                start = time.time()
            elif batch % 100 == 0:
                end = time.time()
                avg = (end-start)/100
                start = end
                print(f'average time per epoch: {len(dataloader) * avg}')
            self._batched_step(batch, example, poison_delta, client, furnace)

        # Return a tensor of poison_delta
        return poison_delta

    def _batched_step(self, batch_idx, example, poison_delta, client, furnace):
        model = client.model
        inputs, labels, ids = example
        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

        # Perform the ReColorAdv + StAdv + Delta attack
        # from https://arxiv.org/abs/1906.00001
        normalizer = utils.DifferentiableNormalize(mean=furnace.trainset.data_mean,
                                                   std=furnace.trainset.data_std)
        attack = get_attack_from_name('recoloradv+stadv+delta', model, normalizer, verbose=False)
        print(f'[Batch idx: {batch_idx}] Num iterations: {attack.attack_kwargs["num_iterations"]}')
        # Default settings for the attack can be overwritten by uncommenting the following line
        # attack.attack_kwargs['num_iterations'] = self.args.attackiter

        # Attack expects unnormalized inputs
        unnormalized_inputs = (inputs * furnace.ds) + furnace.dm
        _,_,_,_, perturbation = attack.attack(unnormalized_inputs, labels)

        # Get adversarial perturbation, then normalize
        perturbation_tensors = perturbation.adversarial_tensors()
        perturbation_tensors = (perturbation_tensors - furnace.dm) / furnace.ds
        
        # Save adversarial perturbations to poison_delta
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(ids.tolist()):
            lookup = furnace.poison_lookup.get(image_id)
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        # Return slice to CPU:
        poison_delta[poison_slices] = perturbation_tensors.detach().to(device=torch.device('cpu'))
