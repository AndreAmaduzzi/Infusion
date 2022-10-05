import logging
logger = logging.getLogger()
import os
import sys
import torch

def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(log_format)
    logger.addHandler(err_handler)
    logger.setLevel(logging.INFO)

    return logger

def normalize_cloud(pcd: torch.Tensor):
    bc = torch.mean(pcd, dim=1, keepdim=True)
    dist = torch.cdist(pcd, bc)
    max_dist = torch.max(dist, dim=1)[0]
    new_pcd = (pcd - bc) / torch.unsqueeze(max_dist, dim=2)

    return new_pcd