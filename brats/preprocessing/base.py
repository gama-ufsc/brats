from pathlib import Path
import nibabel as nib

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union


class Step(ABC):
    """Steps are the building blocks of a pipeline, they should contain an
    operation to be performed on the scans.

    Steps can be run outside a Pipeline (probably through the __call__
    interface), but it is not advisable unless you know what you are doing.

    Attributes:
        tmpdir: directory for the temporary files, provided by the Pipeline.
    """
    def __init__(self, tmpdir=None) -> None:
        if tmpdir is not None:
            self._tmpdir = Path(tmpdir)
        else:
            self._tmpdir = tmpdir

    @abstractmethod
    def run(self, context: Dict) -> Dict:
        """Apply the step.

        Args:
            context: data, outputs and parameters of previous steps.
        
        Returns:
            context: data, outputs and parameters of steps (including this one).
        """

    def __call__(self, **context: Dict) -> Dict:
        return self.run(context)

    @property
    def tmpdir(self):
        if self._tmpdir is None:
            raise AttributeError(
                '`tmpdir` was not set. If you are running this step without a '
                'Pipeline, you must set `self.tmpdir` to an existing directory.'
            )
        else:
            return self._tmpdir

    @tmpdir.setter
    def tmpdir(self, value):
        self._tmpdir = Path(value)

class Pipeline():
    """Sequence of steps to be applied to a multi-modality image.

    Orders the application and provides a shared context so that information
    can flow and one step can use the results of the previous. Runs solely on
    nifti images, so conversion from other formats should be done beforehand.

    Attributes:
        steps: list of operations to be applied in sequence.
        tmpdir: directory for temporary files.
    """
    def __init__(self, steps: List[Step], tmpdir: Union[str, Path]) -> None:
        self.tmpdir = Path(tmpdir)

        self.steps = list()
        for step in steps:
            try:
                step.tmpdir
                # so it doesn't overwrite user-defined tmpdir
            except AttributeError:
                # avoids one step overwriting the files of an previous
                step.tmpdir = self.tmpdir/type(step).__name__
                step.tmpdir.mkdir(exist_ok=True)
            self.steps.append(step)

    def run(self, modalities: Dict[str, Union[str, nib.Nifti1Image]],
        return_context=False) -> Dict:
        """Apply pipeline's steps in the provided modalities.

        Args:
            modalities: modalities can be provided either as loaded nifti
            images or as paths to the images.
            return_context: if False (default), returns the modalities after
            the pipeline; otherwise, returns full context.
        """
        # load modalities
        modalities_ = dict()
        for mod, img in modalities.items():
            if not isinstance(img, nib.Nifti1Image):
                img = Path(img)  # doubles as filepath-check
                img = nib.load(str(img))

            modalities_[mod.lower()] = img

        # initializes context
        context = dict()
        context['modalities'] = [modalities_]  # modalities at each step
        context['transforms'] = {m: list() for m in modalities_.keys()}

        # run
        for step in self.steps:
            context = step.run(context)

        if return_context:
            return context
        else:
            return context['modalities'][-1]
