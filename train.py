"""
Training script for CAM_DGD
"""
from run import CAM_DGD_run

CAM_DGD_run(model_name='CAM_DGD', dataset_name='mosei', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_distill=True)
