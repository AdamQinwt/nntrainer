from tensorboardX import SummaryWriter
import os.path as osp
import time
import logging
from pathlib import Path

def create_logger(output_dir,unique_name,need_tb=True,need_timestamp=True):
    '''
    create loggers
    :param output_dir: root of the output path
    :param unique_name: name
    :param need_tb: whether tensorboard is needed
    :param need_timestamp: whether timestamp is needed
    :return: logger,log dir,summary writer
    '''
    root_output_dir = Path(output_dir)
    if not root_output_dir.exists():
        print(f'=> creating {root_output_dir}')
    unique_name = osp.basename(unique_name).split('.')[0]
    final_output_dir = root_output_dir / unique_name
    print(f'=> creating {final_output_dir}')
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    if need_timestamp:
        log_file = '{}_{}.log'.format(unique_name, time_str)
    else:
        log_file = '{}.log'.format(unique_name)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head,filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    summary_writer=None
    if need_tb:
        tensorboard_log_dir = Path(root_output_dir) / (unique_name + "_" + time_str) if need_timestamp else Path(root_output_dir) / (unique_name)
        print(f"=> creating {tensorboard_log_dir}")
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    logger.info(time_str)
    return logger, str(final_output_dir), summary_writer

import sys
def cmd_params():
    s=sys.argv
    return str(s)

def enumerable2str(en,indent=0):
    s=''
    ind='\t'*indent
    if isinstance(en, str) or isinstance(en, int)or isinstance(en, float):
        s += f'{ind}{en}\n'
    elif isinstance(en, dict):
        for k,v in en.items():
            s += f'{ind}{k}:\n'
            s+=enumerable2str(v,indent+1)
    elif isinstance(en, list) or isinstance(en, tuple):
        s += f'{ind}[\n'
        for v in en:
            s+=enumerable2str(v,indent+1)
        s+=f'{ind}]\n'
    return s

if __name__=='__main__':
    import easydict
    a=easydict.EasyDict()
    a.model=easydict.EasyDict()
    a.model.model_a='hello'
    a.model.model_b=10
    a.model.model_c=[-5,6,-7]
    a.data=easydict.EasyDict()
    a.data.i1='b1'
    s=enumerable2str(a)
    print(s)