#!/usr/bin/env python3

from abc import ABC, abstractmethod

import torch

from ..._utils.common import _format_tensor_into_tuples


class PropagationRule(ABC):
    """
    Base class for all propagation rule classes, also called Z-Rule.
    STABILITY_FACTOR is used to assure that no zero divison occurs.
    """

    STABILITY_FACTOR = 1e-9

    def forward_hook(self, module, inputs, outputs):
        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        inputs = _format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
        for input in inputs:
            if not hasattr(input, "hook_registered"):
                input_hook = self._create_backward_hook_input(input.data)
                self._handle_input_hooks.append(input.register_hook(input_hook))
                input.hook_registered = True
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)
        return outputs.clone()

    @staticmethod
    def backward_hook_activation(module, grad_input, grad_output):
        """Backward hook to propagate relevance over non-linear activations."""
        if (
            isinstance(grad_input, tuple)
            and isinstance(grad_output, tuple)
            and len(grad_input) > len(grad_output)
        ):
            # Adds any additional elements of grad_input if applicable
            # This occurs when registering a backward hook on nn.Dropout
            # modules, which has an additional element of None in
            # grad_input
            return grad_output + grad_input[len(grad_output) :]
        return grad_output

    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            relevance = grad * inputs
            device = grad.device
            if self._has_single_input:
                self.relevance_input[device] = relevance.data
            else:
                self.relevance_input[device].append(relevance.data)
            return relevance

        return _backward_hook_input

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad / (outputs + sign * self.STABILITY_FACTOR)
            self.relevance_output[grad.device] = grad.data
            return relevance

        return _backward_hook_output

    def forward_hook_weights(self, module, inputs, outputs):
        """Save initial activations a_j before modules are changed"""
        device = inputs[0].device if isinstance(inputs, tuple) else inputs.device
        if hasattr(module, "activations") and device in module.activations:
            raise RuntimeError(
                "Module {} is being used more than once in the network, which "
                "is not supported by LRP. "
                "Please ensure that module is being used only once in the "
                "network.".format(module)
            )
        module.activations[device] = tuple(input.data for input in inputs)
        self._manipulate_weights(module, inputs, outputs)

    @abstractmethod
    def _manipulate_weights(self, module, inputs, outputs):
        raise NotImplementedError

    def forward_pre_hook_activations(self, module, inputs):
        """Pass initial activations to graph generation pass"""
        device = inputs[0].device if isinstance(inputs, tuple) else inputs.device
        for input, activation in zip(inputs, module.activations[device]):
            input.data = activation
        return inputs


class EpsilonRule(PropagationRule):
    """
    Rule for relevance propagation using a small value of epsilon
    to avoid numerical instabilities and remove noise.

    Use for middle layers.

    Args:
        epsilon (integer, float): Value by which is added to the
        discriminator during propagation.
    """

    def __init__(self, epsilon=1e-9) -> None:
        self.STABILITY_FACTOR = epsilon

    def _manipulate_weights(self, module, inputs, outputs):
        pass


class GammaRule(PropagationRule):
    """
    Gamma rule for relevance propagation, gives more importance to
    positive relevance.

    Use for lower layers.

    Args:
        gamma (float): The gamma parameter determines by how much
        the positive relevance is increased.
    """

    def __init__(self, gamma=0.25, set_bias_to_zero=False) -> None:
        self.gamma = gamma
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = (
                module.weight.data + self.gamma * module.weight.data.clamp(min=0)
            )
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class Alpha1_Beta0_Rule(PropagationRule):
    """
    Alpha1_Beta0 rule for relevance backpropagation, also known
    as Deep-Taylor. Only positive relevance is propagated, resulting
    in stable results, therefore recommended as the initial choice.

    Warning: Does not work for BatchNorm modules because weight and bias
    are defined differently.

    Use for lower layers.
    """

    def __init__(self, set_bias_to_zero=False) -> None:
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data.clamp(min=0)
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class IdentityRule(EpsilonRule):
    """
    Identity rule for skipping layer manipulation and propagating the
    relevance over a layer. Only valid for modules with same dimensions for
    inputs and outputs.

    Can be used for BatchNorm2D.
    """

    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            return self.relevance_output[grad.device]

        return _backward_hook_input
