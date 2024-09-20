import fire
import torch

def main(bs_per_gpu, acc_num, tgt_num):
    gpu_count = torch.cuda.device_count()
    return bs_per_gpu * acc_num * gpu_count == tgt_num

if __name__ == '__main__':
    fire.Fire(main)
