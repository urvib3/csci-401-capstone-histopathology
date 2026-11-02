"""Exercise real pretrained models, native output, losses and backward on GPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from wcsp.classifiers.tia_model import get_pretrained_model
from wcsp.classifiers.fullres_unet import FullResUNet, MODEL_NAME
from wcsp.classifiers.training_loss import expected_pairwise_cost

assert torch.cuda.is_available()
torch.set_num_threads(4)
print(torch.__version__, torch.cuda.get_device_name(), flush=True)
base, _ = get_pretrained_model(MODEL_NAME)
base = base.cuda().eval()
x = torch.randint(256, (1, 3, 512, 512), device="cuda").float()
y = torch.randint(5, (1, 512, 512), device="cuda")
y[:, :5] = 255
with torch.no_grad():
    baseline_logits = base(base.preproc(x))
model = FullResUNet(base, freeze_base=False).cuda().eval()
logits = model(x)
assert logits.shape == (1, 5, 512, 512)
assert torch.allclose(logits, torch.nn.functional.interpolate(baseline_logits, size=(512, 512), mode="bilinear", align_corners=False), atol=1e-5)
for arch, network in (("baseline", base), ("fullres", model)):
    network.train()
    logits = network(x)
    probs = torch.nn.functional.interpolate(logits.softmax(1), size=(512, 512), mode="bilinear", align_corners=False)
    ce = torch.nn.functional.nll_loss(probs.clamp_min(1e-7).log(), y, ignore_index=255)
    pair = expected_pairwise_cost(probs[..., 128:384, 128:384], y[..., 128:384, 128:384])
    loss = ce + .1 * pair
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    print(arch, 'shape', logits.shape, 'loss', loss.item(), 'peak_GPU_GB', torch.cuda.max_memory_allocated()/1e9, flush=True)
    network.zero_grad(set_to_none=True)
print("GPU smoke passed", flush=True)
