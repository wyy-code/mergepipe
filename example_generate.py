# 生成一个简单的 base.npz 和 expert.npz
import numpy as np

np.savez_compressed(
    "base.npz",
    **{
        "model.layers.0.attn.o_proj.weight": np.random.randn(128, 64).astype(np.float32),
        "model.layers.1.mlp.down_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "model.embed_tokens.weight": np.random.randn(32, 64).astype(np.float32),
    },
)
# expert 模拟为 base 的细微漂移
data = np.load("base.npz")
exp = {k: data[k] + 0.01 * np.sign(data[k]) for k in data.files}
np.savez_compressed("./expert.npz", **exp)
