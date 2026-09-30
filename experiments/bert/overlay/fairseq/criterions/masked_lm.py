# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from omegaconf import II

import torch
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class MaskedLmConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("masked_lm", dataclass=MaskedLmConfig)
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: MaskedLmConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        valid_targets = targets.ne(self.padding_idx)
        ncorrect = (
            logits.argmax(dim=-1)[valid_targets]
            .eq(targets[valid_targets])
            .long()
            .sum()
        )

        nsentences = sample["nsentences"]
        task_cfg = self.task.cfg
        dup_factor = getattr(task_cfg, "dup_factor", 0)
        dynamic_masking = int(getattr(task_cfg, "dynamic_masking", False))
        training_seed = getattr(task_cfg, "seed", 0)
        configured_mask_seed = getattr(task_cfg, "mask_seed", None)
        mask_seed = (
            configured_mask_seed
            if configured_mask_seed is not None
            else training_seed
        )

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": nsentences,
            "sample_size": sample_size,
            "ncorrect": ncorrect,
            # Weight constant experiment metadata by the number of sequences so
            # logging outputs remain safely summable across workers.
            "dup_factor_x_sequences": dup_factor * nsentences,
            "dynamic_masking_x_sequences": dynamic_masking * nsentences,
            "mask_prob_x_sequences": getattr(task_cfg, "mask_prob", 0) * nsentences,
            "seed_x_sequences": training_seed * nsentences,
            "mask_seed_x_sequences": mask_seed * nsentences,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        mlm_loss = loss_sum / sample_size / math.log(2)
        metrics.log_scalar("loss", mlm_loss, sample_size, round=3)
        metrics.log_scalar("mlm_loss", mlm_loss, sample_size, round=3)
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
        metrics.log_scalar(
            "pred_rate",
            float(sample_size) / ntokens,
            ntokens,
            priority=40,
            round=3,
        )
        metrics.log_scalar(
            "masked_accuracy",
            float(ncorrect) / sample_size if sample_size > 0 else 0.0,
            sample_size,
            priority=45,
            round=4,
        )
        metrics.log_scalar_sum("sequence_presentations", nsentences, priority=46)
        metrics.log_scalar_sum("tokens_processed", ntokens, priority=47)

        if nsentences > 0:
            experiment_metadata = {
                "dup_factor": "dup_factor_x_sequences",
                "dynamic_masking": "dynamic_masking_x_sequences",
                "mask_prob": "mask_prob_x_sequences",
                "seed": "seed_x_sequences",
                "mask_seed": "mask_seed_x_sequences",
            }
            for metric_name, logging_name in experiment_metadata.items():
                weighted_sum = sum(
                    log.get(logging_name, 0) for log in logging_outputs
                )
                metrics.log_scalar(
                    metric_name,
                    weighted_sum / nsentences,
                    weight=0,
                    priority=48,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
