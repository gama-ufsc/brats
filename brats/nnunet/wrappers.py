import os
import subprocess

from pathlib import Path


# check nnUNet required environment variables
assert 'nnUNet_raw_data_base' in os.environ, '`nnUNet_raw_data_base` is not set'
assert Path(os.environ['nnUNet_raw_data_base']).exists(), '`nnUNet_raw_data_base` does not exist'

assert 'nnUNet_preprocessed' in os.environ, '`nnUNet_preprocessed` is not set'
assert Path(os.environ['nnUNet_preprocessed']).exists(), '`nnUNet_preprocessed` does not exist'

assert 'RESULTS_FOLDER' in os.environ, '`RESULTS_FOLDER` is not set'
assert Path(os.environ['RESULTS_FOLDER']).exists(), '`RESULTS_FOLDER` does not exist'


def predict(input_dir, output_dir, model, fold, task, trainer,
            chk='model_best'):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    cmd = (f'nnUNet_predict -i {input_dir} -o {output_dir} -t {task} '
           f'-tr {trainer} -m {model} -f {fold} --overwrite_existing '
           f'-chk {chk}')

    try:
        _ = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        return False

    return True
