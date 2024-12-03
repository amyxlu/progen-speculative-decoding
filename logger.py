import logging

from pythonjsonlogger import jsonlogger
from vllm.engine.metrics_types import StatLoggerBase, Stats, SupportsMetricsInfo
from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics


def init_logger(name):
    return logging.getLogger(name)


def init_vllm_json_file_logger(path):
    logger = init_logger("progen_vllm")
    logger.setLevel(logging.INFO)
    log_handler = logging.FileHandler(path)
    formatter = jsonlogger.JsonFormatter()
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    return logger


class VllmStatLogger(StatLoggerBase):
    """VLLM stat logger."""

    def __init__(self, logger: logging.Logger, local_interval: float = 0.1) -> None:
        super().__init__(local_interval)
        self.logger = logger

    def log(self, stats: Stats) -> None:
        self.maybe_update_spec_decode_metrics(stats)
        if self.spec_decode_metrics is not None:
            metrics_dict = self._format_spec_decode_metrics_str(
                self.spec_decode_metrics
            )
            self.logger.info(metrics_dict)

        self.spec_decode_metrics = None

    def _format_spec_decode_metrics_str(self, metrics: SpecDecodeWorkerMetrics) -> dict:
        return {
            key: getattr(metrics, key)
            for key in [
                "draft_acceptance_rate",
                "system_efficiency",
                "num_spec_tokens",
                "accepted_tokens",
                "draft_tokens",
                "emitted_tokens",
            ]
        }

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError
