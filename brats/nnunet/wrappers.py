import subprocess

from pathlib import Path


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
