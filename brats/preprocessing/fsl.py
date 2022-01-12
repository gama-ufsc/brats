import nibabel as nib

from pathlib import Path
from typing import Dict
from nipype.interfaces.fsl.utils import Reorient2Std

from .base import Step


class StdReorientation(Step):
    """Reorient all modalities to the standard orientation.

    See nipype.interfaces.fsl.utils.Reorient2Std.
    """
    def run(self, context: Dict) -> Dict:
        modalities = context['modalities'][-1]

        out_modalities = dict()
        for mod, image in modalities.items():
            mod_name = Path(image.get_filename()).name

            reor = Reorient2Std()
            reor.inputs.in_file = image.get_filename()
            reor.inputs.out_file = self.tmpdir/('reoriented_'+mod_name)

            res = reor.run()

            out_modalities[mod] = nib.load(res.outputs.out_file)
        
        context['modalities'].append(out_modalities)

        return context
