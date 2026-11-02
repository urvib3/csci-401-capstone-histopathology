"""Honor the cluster cache location when loading TIAToolbox models."""
import os
from pathlib import Path


def get_pretrained_model(*args, **kwargs):
    import tiatoolbox
    from tiatoolbox.models.architecture import get_pretrained_model as load

    # TIAToolbox 2.x does not read this environment variable itself.
    default = Path(__file__).resolve().parents[2] / "data/tiatoolbox"
    tiatoolbox.rcParam["TIATOOLBOX_HOME"] = Path(os.environ.get("TIATOOLBOX_HOME", default))
    return load(*args, **kwargs)
