#!/usr/bin/env python3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.stat import Stat
from captum.attr._utils.summarizer import Summarizer
from captum.log import log_usage
from torch import Tensor


class ClassSummarizer(Summarizer):
    r"""
    Used to keep track of summaries for associated classes. The
    classes/labels can be of any type that are supported by `dict`.

    This also keeps track of an aggregate of all class summaries.
    """

    @log_usage()
    def __init__(self, stats: List[Stat]) -> None:
        Summarizer.__init__.__wrapped__(self, stats)
        self.summaries: Dict[Any, Summarizer] = defaultdict(
            lambda: Summarizer(stats=stats)
        )

    def update(  # type: ignore
        self,
        x: TensorOrTupleOfTensorsGeneric,
        labels: TargetType = None,
    ):
        r"""
        Updates the stats of the summarizer, optionally associated to classes.

        This accepts either a single tensor to summarise or a tuple of tensors.

        Args:
            x (Tensor or Tuple[Tensor, ...]):
                The input tensor to be summarised. The first
                dimension of this input must be associated to
                the batch size of the inputs.
            labels (int, tuple, tensor or list, optional):
                The associated labels for `x`. If Any, we
                assume `labels` represents the label for all inputs in `x`.

                If this is None we simply aggregate the total summary.
        """
        if labels is None:
            super().update(x)
            return

        x = _format_tensor_into_tuples(x)

        num_labels = 1

        labels_typed: Union[List[Any], Tensor]
        if isinstance(labels, list) or isinstance(labels, Tensor):
            labels_typed = labels
            num_labels = len(labels)  # = labels.size(0) if tensor
        else:
            labels_typed = [labels]

        # mypy doesn't realise I have made the int a list
        if len(labels_typed) > 1:
            for x_i in x:
                assert x_i.size(0) == num_labels, (
                    "batch size does not equal amount of labels; "
                    "please ensure length of labels is equal to 1 "
                    "or to the `batch_size` corresponding to the "
                    "number of examples in the input(s)"
                )

        batch_size = x[0].size(0)

        for i in range(batch_size):
            tensors_to_summarize = tuple(tensor[i] for tensor in x)
            tensors_to_summarize_copy = tuple(tensor[i].clone() for tensor in x)
            label = labels_typed[0] if len(labels_typed) == 1 else labels_typed[i]

            self.summaries[label].update(tensors_to_summarize)
            super().update(tensors_to_summarize_copy)

    @property
    def class_summaries(
        self,
    ) -> Dict[
        Any, Union[None, Dict[str, Optional[Tensor]], List[Dict[str, Optional[Tensor]]]]
    ]:
        r"""
        Returns:
             The summaries for each class.
        """
        return {key: value.summary for key, value in self.summaries.items()}
