import logging
import os

from utils.conf import print_conf


def set_logger(cfg, file_name=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """

    if 'loglevel' in cfg:
        loglevel = eval('logging.' + cfg['loglevel'])
    else:
        loglevel = logging.INFO

    outname = '{}_{}'.format(cfg.dataset, cfg.netname)

    if file_name is not None:
        outname += '_' + file_name

    if cfg.evaluate:
        outname += '_test.log'
    else:
        outname += '_train.log'

    outdir = cfg['result_root']

    log_path = os.path.join(outdir, outname)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info(print_conf(cfg))
    logging.info('writting logs to file {}'.format(log_path))
