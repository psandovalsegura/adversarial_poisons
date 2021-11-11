"""Main class, holding information about models and training/testing routines."""

import math
import torch
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK
import pdb
from .forgemaster_base import _Forgemaster

class ForgemasterRotation(_Forgemaster):

    def forge(self, client, furnace):
        """Compute and save sample features, then
           perform feature transformation.
        """
        self.angle = math.pi * self.args.rotation_angle_factor
        self.usv, self.num_principal_dirs, self.center_translation = self._perform_pca(client, furnace)
        poison_delta = super().forge(client, furnace)
        return poison_delta

    # def _create_feature_extractor(self, model):
    #     """Create feature extractor for model."""
    #     model.eval()

    #     # TODO: ensure most models have a node named 'flatten'
    #     pre_linear_node_name = 'flatten'
    #     linear_node_name = 'fc'
    #     feature_extractor = create_feature_extractor(model, return_nodes=[pre_linear_node_name, linear_node_name])
    #     return feature_extractor, pre_linear_node_name, linear_node_name
    
    def _get_representation_layer_features(self, client, furnace):
        """For every sample in poisonloader, compute feature"""
        print('==> Computing representation layer features')
        model = client.model
        features = []

        # Define hook
        def log_feature(module, input, output):
            batch_size = output.size(0)
            features.append(output.reshape(batch_size, -1))
            return None

        handle = model.avgpool.register_forward_hook(log_feature)
        for x, _y, _id in furnace.poisonloader:
            x = x.cuda()
            out = model(x)
        handle.remove()

        features = torch.cat(features, dim=0) 
        return features

    def _perform_pca(self, client, furnace):
        """Perform PCA
        """
        with torch.no_grad():
            features = self._get_representation_layer_features(client, furnace)
            num_principal_dirs = 2
            pca_iters = 100

            # Perform PCA
            C = features.mean(dim=(-2,), keepdim=True)
            features = features - C
            usv = torch.pca_lowrank(features, q=num_principal_dirs, center=False, niter=pca_iters)

            # Check covariance of principal dirs 
            S = usv[1]
            cov = S ** 2 / (features.size(0) - 1)
            print(f'==> Covariance of principal components: {cov}') 
            return usv, num_principal_dirs, C

    def _rotate_input_features(self, input_features):
        V = self.usv[2].detach() 

        # Project onto first two principal components
        input_features = input_features - self.center_translation.detach()
        proj_features = torch.matmul(input_features, V[:, :self.num_principal_dirs]) 

        # Rotate every projected feature
        rot = torch.tensor([[math.cos(self.angle), -math.sin(self.angle)], [math.sin(self.angle), math.cos(self.angle)]]).cuda()
        rot_proj_features = torch.matmul(proj_features, rot)

        # Unproject back rotated features by adding back non-principal components
        rot_proj_features = torch.matmul(rot_proj_features, V[:, :self.num_principal_dirs].T) 
        rot_features = rot_proj_features + (input_features - torch.matmul(proj_features, V[:, :self.num_principal_dirs].T)) 

        # Add back centering translation C
        rot_features = rot_features + self.center_translation.detach()
        return rot_features


    def _define_objective(self, inputs, labels):
        """Implement the closure here."""
        def closure(model, criterion, optimizer):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            features = []

            # Define hook
            def log_feature(module, input, output):
                batch_size = output.size(0)
                features.append(output.reshape(batch_size, -1))
                return None

            # Forward 
            handle = model.avgpool.register_forward_hook(log_feature)
            outputs = model(inputs)
            handle.remove()

            # Compute loss
            input_features = features[0]
            rot_input_features = self._rotate_input_features(input_features)
            loss = torch.pow(torch.norm((input_features - rot_input_features), p=2), 2) / input_features.size(0)
            loss.backward(retain_graph=self.retain)

            prediction = (outputs.data.argmax(dim=1) != labels).sum()
            return loss.detach().cpu(), prediction.detach().cpu()
        return closure
