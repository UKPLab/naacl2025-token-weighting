from typing import Optional, List

import torch
from logger_setup import logger
from torch.nn import Softmax
from torch.nn.functional import normalize, softmax
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer


class ChunksDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, idx):
        return self.chunks[idx]


def has_row_with_high_zero_ratio(numbers: torch.Tensor, k: float) -> (bool, int):
    """
    Checks if any row in a 2D tensor has a ratio of zeros greater than k.

    :param numbers: 2D PyTorch tensor
    :param k: A float indicating the threshold (0 <= k <= 1)
    :return: True if any row has a zero ratio greater than k, otherwise False
    """
    zero_counts = torch.sum(numbers == 0, dim=1)  # Count zeros per row
    row_lengths = numbers.shape[1]  # Number of columns in each row
    zero_ratios = zero_counts / row_lengths  # Compute ratios of zeros per row
    return torch.any(zero_ratios > k).item(), zero_counts  # Check if any row exceeds the threshold


def get_top_kappa(numbers: torch.Tensor, k: float, device_no: int = -1) -> torch.tensor:
    """
    Filters the top-kappa percentage of values row-wise for a given tensor, with an option for
    random selection of zero-value indices if their ratio exceeds a threshold. The output is a tensor
    where filtered values from the original tensor are replaced with 1, and others are set to 0.

    :param numbers: A 2-dimensional tensor containing numerical values. Each row represents a
        separate entity for evaluation.
    :type numbers: torch.Tensor
    :param k: A float representing the fraction (between 0 and 1) of the top-kappa elements to retain
        in each row. For example, if `k=0.2`, the top 20% of values in each row are kept.
    :type k: float
    :param device_no: The device number of the GPU where computations should occur. If less than
        0 (default), the device of the `numbers` tensor is automatically detected and used. CPU is
        used if `numbers` is on the CPU.
    :type device_no: int
    :return: A tensor of the same shape as `numbers`, with the top-kappa elements in each row set to
        1 and other elements replaced with 0.
    :rtype: torch.Tensor
    :raises ValueError: If the value of `k` is not greater than 0 and less than or equal to 1.
    """
    if not 0 < k <= 1:
        raise ValueError("k must be between 0 and 1")
    device_no = numbers.get_device() if device_no < 0 else None
    if device_no < 0:
        device_no = None

    too_many_zeros, zero_count = has_row_with_high_zero_ratio(numbers, 1 - k)

    sorted, indices = torch.sort(numbers, dim=1, stable=True)

    if too_many_zeros:
        print(f"We have to pick randomly.")
        for i in range(len(indices)):
            indices[i, :zero_count[i]] = indices[i, torch.randperm(zero_count[i], device=device_no)]
    else:
        print("No random picking was needed.")

    indices_to_keep = indices[:, -int(k * numbers.shape[1]):]

    transformed_numbers = torch.zeros_like(numbers)
    transformed_numbers[torch.arange(numbers.shape[0])[:, None], indices_to_keep] = 1

    return transformed_numbers


def calculate_base_loss(model, batch: torch.Tensor, size: int,
                        base_model_stride: int, tokenizer=None, use_first_chunk=False,
                        minibatch_size=1) -> torch.tensor:
    """
    Splits the batch into overlapping chunks of length size. If the original sequences are prepended with BOS token,
    also prepend the BOS token to the shorter sequences. Then calculate loss for the short sequences
    :param model: the model
    :param batch: the original batch of long context sequences
    :param size: size of the short context
    :param base_model_stride: evaluation stride of base model. Must be chosen such that base_model_stride divides (batch.shape[1] - size).
    can most of the times be chosen such that (batch.shape[1] - size)/base_model_stride = batch.shape[1]/size + 1
    :param use_first_chunk: if short context loss should be calculated for the first chunk. As this loss will be the same as the long context loss,
    it is usually not needed separately.
    :param minibatch_size: how many short context sequences should be processed in the same batch
    :return:
    """

    chunks = batch.unfold(dimension=1, size=size, step=base_model_stride)
    if (batch.shape[1] - size) % base_model_stride != 0:
        raise ValueError(
            "Please choose a base_model_stride such that (long_context_length - short_context_length) can be divided by short_model_stride without rest!")

    # prepend bos token
    if tokenizer is not None and torch.all(batch[:, 0] == tokenizer.bos_token_id):
        logger.info("Prepend Bos Token")
        # change overlap token to bos

        chunks = chunks[:, :, 1:]

        bos_tokens = torch.ones((chunks.shape[0], chunks.shape[1], 1),
                                device=batch.get_device(),
                                requires_grad=False, dtype=torch.long) * tokenizer.bos_token_id
        chunks = torch.cat((bos_tokens, chunks), dim=2)
    else:
        logger.info("Don't prepend Bos Token")

    return forward_passes(chunks, model, size, base_model_stride, use_first_chunk, minibatch_size=minibatch_size,
                          device=batch.device,
                          batch_shape=batch.shape)


def forward_passes(chunks, model, size, base_model_stride, use_first_chunk=False, minibatch_size=1,
                   device=-1, batch_shape=None) -> torch.tensor:
    offset = 0 if use_first_chunk else 1

    losses_base_together = torch.zeros(tuple(batch_shape) if not isinstance(batch_shape, tuple) else batch_shape,
                                       device=device, requires_grad=False)

    # Convert chunks into a dataset
    chunks_dataset = ChunksDataset(
        torch.concat([chunks[:, chunk_no, :] for chunk_no in range(offset, chunks.shape[1])], dim=0))

    # Create a DataLoader for iterating through the dataset
    chunks_loader = DataLoader(chunks_dataset, batch_size=minibatch_size, shuffle=False)

    chunks_processed = 0
    for chunk_batch in chunks_loader:
        # Perform operations on each chunk batch here

        loss_value_base_together = calculate_loss_per_token(model, chunk_batch)

        for pos_in_minibatch in range(len(chunk_batch)):
            pos_of_ctx = chunks_processed // batch_shape[0]
            pos_in_batch = chunks_processed % batch_shape[0]

            if pos_of_ctx == 0:
                if offset == 0:
                    losses_base_together[pos_in_batch, 1:size] = loss_value_base_together[pos_in_minibatch, :]
                else:
                    losses_base_together[pos_in_batch,
                    size + (pos_of_ctx + offset - 1) * base_model_stride:
                    size + (pos_of_ctx + offset) * base_model_stride] = loss_value_base_together[pos_in_minibatch,
                                                                        -base_model_stride:]
            else:
                losses_base_together[pos_in_batch,
                size + (pos_of_ctx + offset - 1) * base_model_stride:
                size + (pos_of_ctx + offset) * base_model_stride] = loss_value_base_together[pos_in_minibatch,
                                                                    -base_model_stride:]
            chunks_processed += 1

    return losses_base_together[:, 1:]


def calculate_loss_per_token(model, input_ids: torch.tensor) -> torch.tensor:
    """
    :param model: the LLM
    :param input_ids: the input
    :return: the loss
    """
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    softmax = Softmax(dim=1)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

        logits = outputs.logits

        # Shift the logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate the loss and then perplexity for each token
        loss = loss_func(torch.transpose(shift_logits, 1, 2), shift_labels)
        entropy = loss_func(torch.transpose(shift_logits, 1, 2), softmax(torch.transpose(shift_logits, 1, 2)))

        loss = loss.detach()
        entropy = entropy.detach()

    return loss


class ShortLongLoss:
    def __init__(self,
                 base_length: int,
                 base_stride: int,
                 logit_comparison: str = "difference",
                 transforms: Optional[List[str]] = None,
                 truncation: Optional[float] = None,
                 sparsification: Optional[float] = None,
                 normalization: Optional[str] = None,
                 interpolation: Optional[float] = None,
                 random_seed: int = 42):
        """
        Initializes an instance of the class with provided configurations and
        validates the parameters. Each parameter serves a specific function
        such as setting the strategy for logit comparison, applying
        transformations, or defining truncation thresholds. Several
        assertions are performed to ensure that the provided values fall
        within valid ranges or valid choices.

        :param base_length: the context length of the short-context model
        :param base_stride: the stride with which the short-context gets created from the long context
        :param logit_comparison: Strategy for comparing logit values.
        :param transforms: List of transformations to apply; valid
            options are "minus", "absolute", "exp", "max" and "shift x" where x is the shift.
        :param truncation: A threshold value to truncate data, or None
            if no truncation is applied.
        :param sparsification: A float between 0.0 and 1.0 representing the
            degree of sparsification.
        :param normalization: Specifies the normalization method,
            if any, to be applied. Options are "L1" and "softmax"
        :param interpolation: A float between 0.0 and 1.0 defining
            the interpolation weight.
        :param random_seed: An integer value used as a random seed
            for reproducibility.

        :raises AssertionError:
            If `transforms` contains invalid elements, if
            `sparsification` is not between 0.0 and 1.0,
            or if `interpolation` is not between 0.0 and 1.0.
        """

        self.base_length = base_length
        self.base_stride = base_stride


        self.logit_comparison = logit_comparison

        self.transforms = [] if transforms is None else transforms

        for transform in self.transforms:
            assert transform in ["minus", "absolute", "exp", "max"]

        self.truncation = truncation
        self.sparsification = sparsification
        assert self.sparsification is None or 0.0 <= self.sparsification <= 1.0, f"interpolation {self.sparsification} is not in [0,1]"
        self.normalization = normalization
        self.interpolation = interpolation
        assert self.interpolation is None or 0.0 <= self.interpolation <= 1.0, f"interpolation {self.interpolation} is not in [0,1]"
        self.random_seed = random_seed

    def normalize_weights(self, unnormalized_weights: torch.tensor) -> torch.tensor:
        """
        Normalizes the provided unnormalized weights tensor based on the specified
        normalization method. This function ensures non-zero weight rows by adjusting
        any row containing all zero values. The normalization can be either L1
        (normalize weights using L1 norm) or softmax (apply softmax function to
        compute probabilities).

        Additionally, an optional interpolation step allows blending the calculated
        weights with a predefined interpolation value.

        :param unnormalized_weights: A tensor representing the unnormalized weights to
            be processed, where each element corresponds to a weight value in the
            tensor.
        :type unnormalized_weights: torch.tensor

        :return: A tensor containing the normalized weights with adjusted probabilities
            based on the specified normalization and optional interpolation.
        :rtype: torch.tensor
        """

        zero_counts = torch.count_nonzero(unnormalized_weights, dim=-1).detach()
        for i in range(unnormalized_weights.shape[0]):
            if zero_counts[i] == 0:
                unnormalized_weights[i] = (1 / unnormalized_weights.shape[-1])
        if self.normalization == "L1":
            weights = normalize(unnormalized_weights, p=1.0, dim=-1).detach()
            weights = (weights * unnormalized_weights.shape[-1]).detach()
        elif self.normalization == "softmax":
            weights = softmax(unnormalized_weights, dim=-1).detach()
            weights = (weights * unnormalized_weights.shape[-1]).detach()
        else:
            raise ValueError(f"normalization has to be 'L1' or 'softmax', but input is {self.normalization}")

        if self.interpolation is not None:
            weights = (self.interpolation + (1 - self.interpolation) * weights).detach()

        return weights

    def calculate_weights(self, loss: torch.tensor, base_loss: torch.tensor) -> torch.tensor:
        """

        :param loss: full length loss
        :param base_loss: loss based on shorter context
        :return: the weights, calculated given the class attributes.
        """

        if self.logit_comparison == "quotient":
            weights = ((loss + 1e-5) / (base_loss + 1e-5)).detach()
        elif self.logit_comparison == "difference":
            weights = (loss - base_loss).detach()
        else:
            raise ValueError(
                f"logit comparison {self.logit_comparison} is not supported, choose 'difference' or 'quotient'")

        if self.transforms is not None:
            for transform in self.transforms:
                if transform == "absolute":
                    weights = torch.abs(weights).detach()
                elif transform == "minus":
                    weights = -weights.detach()
                elif transform == "max":
                    weights[weights < 0] = 0
                elif transform == "exp":
                    weights = torch.exp(weights)
                elif transform.startswith("shift"):
                    shift_value = float(transform.split(" ")[1])
                    assert shift_value > 0, f"given shift value {shift_value} is <= 0. This is not possible because of the log"
                    weights -= torch.log(torch.ones_like(weights) * shift_value)
                else:
                    raise ValueError(
                        f"The requested transform {transform} is not supported: 'absolute', 'minus', 'max', 'exp', 'shift x'")

        if self.truncation is not None:
            weights = torch.clip(weights, max=self.truncation)

        if self.sparsification is not None:
            weights = get_top_kappa(weights, self.sparsification).detach()

        if self.normalization is not None:
            weights = self.normalize_weights(weights)

        return weights


class CustomTrainer(Trainer):
    def __init__(self, settings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_frozen_base = settings.use_frozen_base

        self.precision = settings.precision
        self.custom_loss = settings.custom_loss
        self.custom_loss.random_seed = settings.seed
        self.random_seed = settings.seed
        self.tokenizer = settings.tokenizer
        self.scoring_minibatch = settings.scoring_minibatch

        self.forward_pass_counter = 0
        self.vanilla_loss_for_logging = 0
        self.std_for_logging = 0
        self.model_path = settings.model_path
        self.log_perplexity = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        batch = inputs["input_ids"]

        if self.custom_loss is not None and not self.use_frozen_base:
            logger.info(f"calculate small model weights with unfrozen model")

            with torch.no_grad():
                base_loss = calculate_base_loss(model, batch, self.custom_loss.base_length,
                                                self.custom_loss.base_stride, self.tokenizer,
                                                minibatch_size=self.scoring_minibatch,
                                                use_first_chunk=False)

        else:
            logger.info(f"fetches small model weights from frozen model")
            base_loss = inputs["cross_entropy"].detach()
            if len(base_loss.shape) == 3:
                if base_loss.shape[0] == 1:
                    base_loss = torch.squeeze(base_loss, dim=0)
                elif base_loss.shape[1] == 1:
                    base_loss = torch.squeeze(base_loss, dim=1)
            assert len(base_loss.shape) == 2

        outputs = model(batch)

        logits = outputs.logits

        # Shift the logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        loss = self.loss_func(torch.transpose(shift_logits, 1, 2), shift_labels)

        loss_value = loss.detach()
        logger.info(f"loss value: {loss_value}")
        if torch.isnan(loss_value).any():
            raise Exception("NaN loss")
        if torch.isinf(loss_value).any():
            raise Exception("Inf loss")

        # this saves us a forward pass as the first 4k tokens have the same context and thus the same loss
        if self.custom_loss is not None and not self.use_frozen_base:
            base_loss[:, :self.custom_loss.base_length] = loss_value[:, :self.custom_loss.base_length]
            logger.info("Updated base loss with main model loss")
            logger.info(f"base loss: {base_loss}")

        if self.custom_loss is not None:
            weights = self.custom_loss.calculate_weights(loss_value, base_loss)
            logger.info(f"normalized weights: {weights}")
            logger.info(f"average weight: {torch.mean(weights, dim=1).detach()}")
            logger.info(f"weight std: {torch.std(weights, dim=1).detach()}")

            if torch.isnan(weights).any():
                raise Exception("NaN weights")
            if torch.isinf(weights).any():
                raise Exception("Inf weights")

            weighted_loss = torch.mean(loss * weights)

            self.vanilla_loss_for_logging += torch.mean(loss_value).detach().cpu().item()
            self.std_for_logging += torch.mean(torch.std(weights, dim=1)).detach().cpu().item()
            self.forward_pass_counter += 1

            if self.forward_pass_counter == self.args.gradient_accumulation_steps:
                if self.state.is_world_process_zero:
                    self.log({
                        "global_step": int(self.state.global_step / self.args.gradient_accumulation_steps),
                        "unweighted_loss": self.vanilla_loss_for_logging / self.args.gradient_accumulation_steps,
                        "std": self.std_for_logging / self.args.gradient_accumulation_steps
                    })
                self.forward_pass_counter = 0
                self.vanilla_loss_for_logging = 0
                self.std_for_logging = 0

        else:
            logger.info("Don't use custom loss and set weights uniformly to 1")
            weighted_loss = torch.mean(loss)

        logger.info(f"loss*weights for backpropagation: {weighted_loss}")

        return (weighted_loss, outputs) if return_outputs else weighted_loss
