#!/usr/bin/env python3
from typing import Any, Callable, Tuple

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _select_targets,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    undo_gradient_requirements,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.robust._core.perturbation import Perturbation
from torch import Tensor


class FGSM(Perturbation):
    r"""
    Fast Gradient Sign Method is an one-step method that can generate
    adversarial examples. For non-targeted attack, the formulation is
    x' = x + epsilon * sign(gradient of L(theta, x, y)).
    For targeted attack on t, the formulation is
    x' = x - epsilon * sign(gradient of L(theta, x, t)).
    L(theta, x, y) is the model's loss function with respect to model
    parameters, inputs and labels.

    More details on Fast Gradient Sign Method can be found in the original
    paper:
    https://arxiv.org/pdf/1412.6572.pdf
    """

    def __init__(
        self,
        forward_func: Callable,
        loss_func: Callable = None,
        lower_bound: float = float("-inf"),
        upper_bound: float = float("inf"),
    ) -> None:
        r"""
        Args:
            forward_func (callable): The pytorch model for which the attack is
                        computed.
            loss_func (callable, optional): Loss function of which the gradient
                        computed. The loss function should take in outputs of the
                        model and labels, and return a loss tensor.
                        The default loss function is negative log.
            lower_bound (float, optional): Lower bound of input values.
            upper_bound (float, optional): Upper bound of input values.
                        e.g. image pixels must be in the range 0-255

        Attributes:
            bound (Callable): A function that bounds the input values based on
                        given lower_bound and upper_bound. Can be overwritten for
                        custom use cases if necessary.
            zero_thresh (float): The threshold below which gradient will be treated
                        as zero. Can be modified for custom use cases if necessary.
        """
        super().__init__()
        self.forward_func = forward_func
        self.loss_func = loss_func
        self.bound = lambda x: torch.clamp(x, min=lower_bound, max=upper_bound)
        self.zero_thresh = 10 ** -6

    def perturb(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        epsilon: float,
        target: Any,
        additional_forward_args: Any = None,
        targeted: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method computes and returns the perturbed input for each input tensor.
        It supports both targeted and non-targeted attacks.

        Args:

            inputs (tensor or tuple of tensors): Input for which adversarial
                        attack is computed. It can be provided as a single
                        tensor or a tuple of multiple tensors. If multiple
                        input tensors are provided, the batch sizes must be
                        aligned accross all tensors.
            epsilon (float): Step size of perturbation.
            target (any): True labels of inputs if non-targeted attack is
                        desired. Target class of inputs if targeted attack
                        is desired. Target will be passed to the loss function
                        to compute loss, so the type needs to match the
                        argument type of the loss function.

                        If using the default negative log as loss function,
                        labels should be of type int, tuple, tensor or list.
                        For general 2D outputs, labels can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the label for the corresponding example.

                        For outputs with > 2 dimensions, labels can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This label index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          label for the corresponding example.
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. These arguments are provided to
                        forward_func in order following the arguments in inputs.
                        Default: None.
            targeted (bool, optional): If attack should be targeted.
                        Default: False.


        Returns:

            - **perturbed inputs** (*tensor* or tuple of *tensors*):
                        Perturbed input for each
                        input tensor. The perturbed inputs have the same shape and
                        dimensionality as the inputs.
                        If a single tensor is provided as inputs, a single tensor
                        is returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
        """
        is_inputs_tuple = _is_tuple(inputs)
        inputs: Tuple[Tensor, ...] = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        def _forward_with_loss() -> Tensor:
            additional_inputs = _format_additional_forward_args(additional_forward_args)
            outputs = self.forward_func(  # type: ignore
                *(*inputs, *additional_inputs)  # type: ignore
                if additional_inputs is not None
                else inputs
            )
            if self.loss_func is not None:
                return self.loss_func(outputs, target)
            else:
                loss = -torch.log(outputs)
                return _select_targets(loss, target)

        grads = compute_gradients(_forward_with_loss, inputs)
        undo_gradient_requirements(inputs, gradient_mask)
        perturbed_inputs = self._perturb(inputs, grads, epsilon, targeted)
        perturbed_inputs = tuple(
            self.bound(perturbed_inputs[i]) for i in range(len(perturbed_inputs))
        )
        return _format_output(is_inputs_tuple, perturbed_inputs)

    def _perturb(
        self,
        inputs: Tuple,
        grads: Tuple,
        epsilon: float,
        targeted: bool,
    ) -> Tuple:
        r"""
        A helper function to calculate the perturbed inputs given original
        inputs, gradient of loss function and epsilon. The calculation is
        different for targetd v.s. non-targeted as described above.
        """
        multiplier = -1 if targeted else 1
        inputs = tuple(
            torch.where(
                torch.abs(grad) > self.zero_thresh,
                inp + multiplier * epsilon * torch.sign(grad),
                inp,
            )
            for grad, inp in zip(grads, inputs)
        )
        return inputs
