import pandas as pd
from tqdm import tqdm
import numpy as np
import fire

import wandb

def get_run_id(project, run_name, entity='dail'):
    api = wandb.Api()
    runs = api.runs(path=f'{entity}/{project}')
    for run in runs:
        if run.name == run_name:
            return run.id
    return ''

def record_result(res, project, run_name):
    run_id = get_run_id(project, run_name)
    
    wandb.login()
    run = wandb.init(project=project, id=run_id, resume='must')
    wandb.log(res)
    
    # write to dir
    import json
    import os
    import time
    mmbench_res_dir = '/mnt/workspace/lielin.hyl/projects/MGM/work_dirs/mmbench'
    if not os.path.exists(mmbench_res_dir):
        os.makedirs(mmbench_res_dir)
    res_fn = f'{project}_{run_name}_{time.time_ns()}.json'
    json.dump(res, open(os.path.join(mmbench_res_dir, res_fn), 'w'))

def main(res_excel, split=None, wandb_project=None, wandb_run_name=None):
    data = pd.read_excel(res_excel)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in 'ABCD' else k] = data.pop(k)
    data_main = data[data['index'] < int(1e6)]

    result = {}
    hit, tot = 0, 0
    lt = len(data_main)
    for i in tqdm(range(lt)):
        item_main = data_main.iloc[i]
        idx = item_main['index']
        if idx in result:
            correct = result[idx]
            hit += correct
            tot += 1
            continue
        sub_data = data[data['index'] % int(1e6) == idx]
        all_correct = 1
        for j in range(len(sub_data)):
            item = sub_data.iloc[j]
            if item['prediction'] != item['answer']:
                all_correct = 0
                break
        result[idx] = all_correct
        hit += all_correct
        tot += 1

    score = 100.0 * hit / tot
    print('%.2f%%' % (score))
    
    if wandb_project and wandb_run_name:
        if 'cn' in split:
            split_name = 'MMBench-CN'
        else:
            split_name = 'MMBench'
        record_result({f'{split_name}': score}, wandb_project, wandb_run_name)

if __name__ == '__main__':
    fire.Fire(main)