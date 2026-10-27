import pytest

torch = pytest.importorskip("torch")

from pixel.train import TrainConfig, edit, load_model, parse_direction, train  # noqa: E402


def test_smoke_train_loss_falls_and_edit_runs(tmp_path):
    cfg = TrainConfig(out_dir=str(tmp_path / "run"), steps=40, batch=4, val_every=20, ckpt_every=1000, smoke=True, seed=0)
    res = train(cfg)
    assert res["steps"] == 40
    assert res["loss_last"] < res["loss_first"]
    assert res["val"]["n"] > 0 and res["val"]["psnr_output"] > 0
    model, _cfg = load_model(res["checkpoint"])
    import numpy as np

    img = np.random.default_rng(0).random((70, 90, 3)).astype(np.float32)
    out, pdir = edit(model, img, parse_direction("blur=-1,noise=-0.5"))
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0
    assert pdir.shape == (12,)
    with pytest.raises(KeyError):
        parse_direction("nope=1")
