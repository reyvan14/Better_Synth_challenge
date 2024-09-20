import fire
import random
import jsonlines as jl
from loguru import logger

random.seed(1024)

def main(input_dataset, output_dataset, sample_num=200000):
    with jl.open(input_dataset) as reader:
        with jl.open(output_dataset, 'w') as writer:
            ds = []
            for s in reader:
                ds.append(s)
            logger.info(f'Total number of samples in the input dataset: {len(ds)}')
            if sample_num < len(ds):
                logger.info(f'Sampled number from the input dataset: {sample_num}')
                sampled_ds = random.sample(ds, sample_num)
            else:
                logger.info(f'Keep all samples in the input dataset.')
                sampled_ds = ds
            writer.write_all(sampled_ds)

if __name__ == '__main__':
    fire.Fire(main)
