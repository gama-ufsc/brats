import os
import subprocess

from pathlib import Path


def predict(input_dir, output_dir, model, fold, task, trainer,
            chk='model_best', overwrite=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if fold.startswith('fold_'):
        fold = fold[-1]
    elif fold == 'ensemble':
        raise NotImplementedError  # TODO: implement ensemble prediction

    cmd = (f'nnUNet_predict -i {input_dir} -o {output_dir} -t {task} '
           f'-tr {trainer} -m {model} -f {fold} -chk {chk}')

    if overwrite:
        cmd += ' --overwrite_existing'

    try:
        _ = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        return False

    return True
