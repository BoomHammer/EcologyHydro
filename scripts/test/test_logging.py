"""Log setup must not duplicate records when reused in a batch process."""

import logging

from ecologyhydro.logging_utils import configure_logging


def test_logging_reconfiguration_and_utf8(tmp_path):
    root_handlers = logging.getLogger().handlers[:]
    log_path = tmp_path / "nested" / "run.log"
    configure_logging(log_path)
    logger = configure_logging(log_path)
    logger.info("黄河流域")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    assert log_path.read_text(encoding="utf-8").count("黄河流域") == 1
    assert logging.getLogger().handlers == root_handlers
