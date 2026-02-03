# mergepipe/storage/io.py
from __future__ import annotations

import glob
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, Union

import numpy as np
from safetensors import safe_open as st_safe_open
from safetensors.numpy import save_file as st_save_file

# Optional torch import (lazy)
_TORCH_OK = False
try:
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ------------------ Utilities ------------------


def _ensure_torch():
    if not _TORCH_OK:
        raise RuntimeError("PyTorch is required but not installed. Please `pip install torch`.")


def _to_float32_np(x: ArrayLike) -> np.ndarray:
    if _TORCH_OK and isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float32, device="cpu").numpy()
    return np.asarray(x, dtype=np.float32)


def _to_float32_torch(x: ArrayLike, device: str | None = None) -> torch.Tensor:
    _ensure_torch()
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x
    t = t.to(dtype=torch.float32)
    if device:
        t = t.to(device)
    return t


def _read_tensor_numpy_or_torch(
    file_path: str, key: str, backend: str = "np", device: str | None = None
) -> ArrayLike:
    """
    读取单个 tensor：
    - backend="np": 优先 numpy；遇到 bf16 等不支持 dtype 时回退 torch -> float32 numpy
    - backend="pt": 直接用 torch 返回 float32 torch.Tensor（可放到 device）
    """
    if backend == "pt":
        _ensure_torch()
        with st_safe_open(file_path, framework="pt") as f:
            t = f.get_tensor(key).to(dtype=torch.float32)
        if device:
            t = t.to(device)
        return t

    # backend == "np": try numpy first
    try:
        with st_safe_open(file_path, framework="numpy") as f:
            arr = f.get_tensor(key)
        return arr.astype(np.float32, copy=True)
    except TypeError as e:
        # e.g., "data type 'bfloat16' not understood"
        if "bfloat16" not in str(e).lower():
            raise

    # fallback to torch then convert to numpy float32
    _ensure_torch()
    with st_safe_open(file_path, framework="pt") as f:
        t = f.get_tensor(key).to(dtype=torch.float32, device="cpu")
    return t.numpy()


# ------------------ Storage Base & Implementations ------------------


class StorageBase:
    """
    抽象存储接口（张量级/块级读写）。
    返回与写入的数据类型由 backend 决定：
      - backend="np": 使用 numpy.ndarray(float32)
      - backend="pt": 使用 torch.Tensor(float32)（可配置 device）
    """

    backend: str = "np"

    def iter_tensors(self) -> Iterable[tuple[str, ArrayLike]]:
        raise NotImplementedError

    def load_block(self, name: str, start_idx: int, end_idx: int) -> ArrayLike:
        raise NotImplementedError

    def write_block(self, name: str, start_idx: int, end_idx: int, data: ArrayLike) -> None:
        raise NotImplementedError

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        raise NotImplementedError

    def dtype(self, name: str) -> str:
        raise NotImplementedError

    def flush(self, out_path: str | None = None) -> None:
        # Optional
        return


class FakeStorage(StorageBase):
    """内存中的张量字典，适合单测。"""

    def __init__(
        self, tensors: dict[str, ArrayLike], backend: str = "np", device: str | None = None
    ):
        self.backend = backend
        self.device = device
        # 统一缓存为 float32 对应后端
        self.tensors: dict[str, ArrayLike] = {}
        for k, v in tensors.items():
            if backend == "pt":
                self.tensors[k] = _to_float32_torch(v, device=device)
            else:
                self.tensors[k] = _to_float32_np(v)

    def iter_tensors(self):
        yield from self.tensors.items()

    def _flat(self, name: str) -> ArrayLike:
        arr = self.tensors[name]
        return arr.reshape(-1)

    def load_block(self, name: str, start_idx: int, end_idx: int) -> ArrayLike:
        flat = self._flat(name)
        if self.backend == "pt":
            return flat[start_idx:end_idx].to(dtype=torch.float32)
        return flat[start_idx:end_idx].astype(np.float32, copy=True)

    def write_block(self, name: str, start_idx: int, end_idx: int, data: ArrayLike) -> None:
        if self.backend == "pt":
            flat = self.tensors[name].reshape(-1)
            flat[start_idx:end_idx] = _to_float32_torch(data, device=self.device)
            self.tensors[name] = flat.reshape(self.tensors[name].shape)
        else:
            flat = self.tensors[name].reshape(-1)
            flat[start_idx:end_idx] = _to_float32_np(data)
            self.tensors[name] = flat.reshape(self.tensors[name].shape)

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.tensors[name].shape)  # type: ignore[attr-defined]

    def dtype(self, name: str) -> str:
        v = self.tensors[name]
        return str(v.dtype)


class DiskNPZStorage(StorageBase):
    """以 .npz 文件模拟权重容器（仅 numpy 后端有意义）。"""

    def __init__(
        self, path: str, writable: bool = False, backend: str = "np", device: str | None = None
    ):
        assert backend == "np", "DiskNPZStorage only supports numpy backend."
        self.path = path
        self._archive = np.load(path, allow_pickle=False)
        self._writable = writable
        self.backend = "np"
        self.device = None
        self._cache = {k: self._archive[k].astype(np.float32).copy() for k in self._archive.files}

    def iter_tensors(self):
        for k in self._cache.keys():
            yield k, self._cache[k]

    def _flat(self, name: str) -> np.ndarray:
        return self._cache[name].reshape(-1)

    def load_block(self, name: str, start_idx: int, end_idx: int) -> np.ndarray:
        flat = self._flat(name)
        return flat[start_idx:end_idx].astype(np.float32, copy=True)

    def write_block(self, name: str, start_idx: int, end_idx: int, data: ArrayLike) -> None:
        if not self._writable:
            raise RuntimeError("Storage opened as read-only")
        flat = self._flat(name)
        flat[start_idx:end_idx] = _to_float32_np(data)
        self._cache[name] = flat.reshape(self._cache[name].shape)

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._cache[name].shape)

    def dtype(self, name: str) -> str:
        return str(self._cache[name].dtype)

    def flush(self, out_path: str | None = None):
        np.savez_compressed(out_path or self.path, **self._cache)


# --- SafeTensors 单文件实现（按需读取 + bf16 回退 + backend 切换 + RO Cache） ---


class SafeTensorsStorage(StorageBase):
    """
    safetensors 单文件存储：
    - backend="np": 返回/写入 numpy.float32；遇到 bf16 读会自动 torch 回退再转 np.float32
    - backend="pt": 返回/写入 torch.float32；可将数据放在 device 上
    - 写：块写整张量缓存；flush 时 save_file()
    - 读：带只读缓存（RO cache），同一张量只读一次
    """

    def __init__(
        self, path: str, writable: bool = False, backend: str = "np", device: str | None = None
    ):
        self.path = path
        self._writable = writable
        self.backend = backend
        self.device = device

        # 仅建立索引（张量名 -> (shape, dtype_str)）
        self._index: dict[str, dict[str, Any]] = {}
        try:
            with st_safe_open(self.path, framework="numpy") as f:
                for k in f.keys():
                    t = f.get_tensor(k)  # 可能在 bf16 下抛 TypeError
                    self._index[k] = {"shape": tuple(t.shape), "dtype": str(t.dtype)}
        except TypeError:
            _ensure_torch()
            with st_safe_open(self.path, framework="pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    self._index[k] = {"shape": tuple(t.shape), "dtype": str(t.dtype)}

        # 脏缓存（整张量）
        self._dirty_np: dict[str, np.ndarray] = {}
        self._dirty_pt: dict[str, torch.Tensor] = {}
        # 只读缓存（避免分析阶段重复 IO）
        self._ro_cache_np: dict[str, np.ndarray] = {}
        self._ro_cache_pt: dict[str, torch.Tensor] = {}

    def iter_tensors(self):
        for name in self._index.keys():
            if self.backend == "pt":
                if name in self._dirty_pt:
                    yield name, self._dirty_pt[name]
                elif name in self._ro_cache_pt:
                    yield name, self._ro_cache_pt[name]
                else:
                    arr = _read_tensor_numpy_or_torch(
                        self.path, name, backend="pt", device=self.device
                    )
                    self._ro_cache_pt[name] = arr
                    yield name, arr
            else:
                if name in self._dirty_np:
                    yield name, self._dirty_np[name]
                elif name in self._ro_cache_np:
                    yield name, self._ro_cache_np[name]
                else:
                    arr = _read_tensor_numpy_or_torch(self.path, name, backend="np")
                    self._ro_cache_np[name] = arr
                    yield name, arr

    def _flat(self, name: str) -> ArrayLike:
        if self.backend == "pt":
            if name in self._dirty_pt:
                return self._dirty_pt[name].reshape(-1)
            if name in self._ro_cache_pt:
                return self._ro_cache_pt[name].reshape(-1)
            arr = _read_tensor_numpy_or_torch(self.path, name, backend="pt", device=self.device)
            self._ro_cache_pt[name] = arr
            return arr.reshape(-1)
        else:
            if name in self._dirty_np:
                return self._dirty_np[name].reshape(-1)
            if name in self._ro_cache_np:
                return self._ro_cache_np[name].reshape(-1)
            arr = _read_tensor_numpy_or_torch(self.path, name, backend="np")
            self._ro_cache_np[name] = arr
            return arr.reshape(-1)

    def load_block(self, name: str, start_idx: int, end_idx: int) -> ArrayLike:
        flat = self._flat(name)
        if self.backend == "pt":
            return flat[start_idx:end_idx].to(dtype=torch.float32)
        else:
            return flat[start_idx:end_idx].astype(np.float32, copy=True)

    def write_block(self, name: str, start_idx: int, end_idx: int, data: ArrayLike) -> None:
        if not self._writable:
            raise RuntimeError("SafeTensorsStorage opened as read-only")
        if self.backend == "pt":
            if name not in self._dirty_pt:
                full = _read_tensor_numpy_or_torch(
                    self.path, name, backend="pt", device=self.device
                )
                self._dirty_pt[name] = full
            flat = self._dirty_pt[name].reshape(-1)
            flat[start_idx:end_idx] = _to_float32_torch(data, device=self.device)
            self._dirty_pt[name] = flat.reshape(self._dirty_pt[name].shape)
            # 同步 RO 缓存，后续读能看到最新
            self._ro_cache_pt[name] = self._dirty_pt[name]
        else:
            if name not in self._dirty_np:
                full = _read_tensor_numpy_or_torch(self.path, name, backend="np")
                self._dirty_np[name] = full
            flat = self._dirty_np[name].reshape(-1)
            flat[start_idx:end_idx] = _to_float32_np(data)
            self._dirty_np[name] = flat.reshape(self._dirty_np[name].shape)
            self._ro_cache_np[name] = self._dirty_np[name]

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._index[name]["shape"])

    def dtype(self, name: str) -> str:
        return str(self._index[name]["dtype"])

    def flush(self, out_path: str | None = None) -> None:
        tensors: dict[str, Any] = {}
        # 统一保存 float32
        for name in self._index.keys():
            if self.backend == "pt":
                if name in self._dirty_pt:
                    tensors[name] = (
                        self._dirty_pt[name].to(dtype=torch.float32, device="cpu").numpy()
                    )
                elif name in self._ro_cache_pt:
                    tensors[name] = (
                        self._ro_cache_pt[name].to(dtype=torch.float32, device="cpu").numpy()
                    )
                else:
                    arr = _read_tensor_numpy_or_torch(self.path, name, backend="pt")
                    tensors[name] = arr.to(dtype=torch.float32, device="cpu").numpy()
            else:
                if name in self._dirty_np:
                    tensors[name] = self._dirty_np[name].astype(np.float32, copy=False)
                elif name in self._ro_cache_np:
                    tensors[name] = self._ro_cache_np[name].astype(np.float32, copy=False)
                else:
                    tensors[name] = _read_tensor_numpy_or_torch(self.path, name, backend="np")
        st_save_file(tensors, out_path or self.path)
        self._dirty_np.clear()
        self._dirty_pt.clear()
        self._ro_cache_np.clear()
        self._ro_cache_pt.clear()


# --- Sharded SafeTensors（HF 分片，带 RO Cache） ---


class ShardedSafeTensorsStorage(StorageBase):
    """
    兼容 HuggingFace 分片布局：
    - 目录下：model.safetensors 或 model-00001-of-000xx.safetensors
    - 按需读取；只写“脏分片”；带只读缓存（RO cache）
    """

    SHARD_RE = re.compile(r".*model-(\d+)-of-(\d+)\.safetensors$")

    def __init__(
        self, dir_path: str, writable: bool = False, backend: str = "np", device: str | None = None
    ):
        self.dir = dir_path
        self._writable = writable
        self.backend = backend
        self.device = device

        self._is_single = os.path.isfile(os.path.join(dir_path, "model.safetensors"))
        self._index: dict[str, dict[str, Any]] = (
            {}
        )  # name -> {"shard": path, "shape":..., "dtype":...}
        self._dirty_np: dict[str, dict[str, np.ndarray]] = {}
        self._dirty_pt: dict[str, dict[str, torch.Tensor]] = {}
        self._ro_cache_np: dict[str, dict[str, np.ndarray]] = {}  # shard -> { name: arr }
        self._ro_cache_pt: dict[str, dict[str, torch.Tensor]] = {}  # shard -> { name: tensor }

        if self._is_single:
            single_path = os.path.join(dir_path, "model.safetensors")
            try:
                with st_safe_open(single_path, framework="numpy") as f:
                    for k in f.keys():
                        t = f.get_tensor(k)  # bf16 时可能抛 TypeError
                        self._index[k] = {
                            "shard": single_path,
                            "shape": tuple(t.shape),
                            "dtype": str(t.dtype),
                        }
            except TypeError:
                _ensure_torch()
                with st_safe_open(single_path, framework="pt") as f:
                    for k in f.keys():
                        t = f.get_tensor(k)
                        self._index[k] = {
                            "shard": single_path,
                            "shape": tuple(t.shape),
                            "dtype": str(t.dtype),
                        }
        else:
            shard_paths = sorted(glob.glob(os.path.join(dir_path, "model-*-of-*.safetensors")))
            if not shard_paths:
                raise FileNotFoundError(f"No safetensors found under {dir_path}")
            for spath in shard_paths:
                try:
                    with st_safe_open(spath, framework="numpy") as f:
                        for k in f.keys():
                            t = f.get_tensor(k)
                            self._index[k] = {
                                "shard": spath,
                                "shape": tuple(t.shape),
                                "dtype": str(t.dtype),
                            }
                except TypeError:
                    _ensure_torch()
                    with st_safe_open(spath, framework="pt") as f:
                        for k in f.keys():
                            t = f.get_tensor(k)
                            self._index[k] = {
                                "shard": spath,
                                "shape": tuple(t.shape),
                                "dtype": str(t.dtype),
                            }

    def _resolve_name(self, name: str) -> str:
        # Fast path
        if name in self._index:
            return name

        cands = []

        # 1) 尝试加/去 model 前缀（覆盖不同HF封装层级）
        prefixes = ["", "model.", "model.model."]
        # 如果 name 本身带 model.，也尝试去掉一层或两层
        stripped = name
        if stripped.startswith("model.model."):
            cands.append(stripped[len("model.model."):])
        if stripped.startswith("model."):
            cands.append(stripped[len("model."):])

        for p in prefixes:
            cands.append(p + name)

        # 2) tie-embedding 特判：lm_head <-> embed_tokens（并做同样前缀扩展）
        def _expand(x: str):
            out = []
            for p in prefixes:
                out.append(p + x)
            if x.startswith("model.model."):
                out.append(x[len("model.model."):])
            if x.startswith("model."):
                out.append(x[len("model."):])
            return out

        if name.endswith("lm_head.weight"):
            cands.extend(_expand("model.embed_tokens.weight"))
            cands.extend(_expand("embed_tokens.weight"))
        if name.endswith("embed_tokens.weight"):
            cands.extend(_expand("lm_head.weight"))

        # 3) 返回第一个命中的
        for k in cands:
            if k in self._index:
                return k

        return name


    # ---- 基本查询 ----
    def iter_tensors(self):
        yielded = set()
        for name, meta in self._index.items():
            if name in yielded:
                continue
            shard = meta["shard"]
            if self.backend == "pt":
                cache = self._ro_cache_pt.get(shard, {})
                if name in cache:
                    arr = cache[name]
                else:
                    arr = _read_tensor_numpy_or_torch(shard, name, backend="pt", device=self.device)
                    self._ro_cache_pt.setdefault(shard, {})[name] = arr
            else:
                cache = self._ro_cache_np.get(shard, {})
                if name in cache:
                    arr = cache[name]
                else:
                    arr = _read_tensor_numpy_or_torch(shard, name, backend="np")
                    self._ro_cache_np.setdefault(shard, {})[name] = arr
            yield name, arr
            yielded.add(name)

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        real = self._resolve_name(name)
        return tuple(self._index[real]["shape"])

    def dtype(self, name: str) -> str:
        real = self._resolve_name(name)
        return str(self._index[real]["dtype"])

    # ---- 块级读写 ----
    def _flat_from_shard(self, shard_path: str, name: str) -> ArrayLike:
        if self.backend == "pt":
            if shard_path in self._dirty_pt and name in self._dirty_pt[shard_path]:
                return self._dirty_pt[shard_path][name].reshape(-1)
            if shard_path in self._ro_cache_pt and name in self._ro_cache_pt[shard_path]:
                return self._ro_cache_pt[shard_path][name].reshape(-1)
            arr = _read_tensor_numpy_or_torch(shard_path, name, backend="pt", device=self.device)
            self._ro_cache_pt.setdefault(shard_path, {})[name] = arr
            return arr.reshape(-1)
        else:
            if shard_path in self._dirty_np and name in self._dirty_np[shard_path]:
                return self._dirty_np[shard_path][name].reshape(-1)
            if shard_path in self._ro_cache_np and name in self._ro_cache_np[shard_path]:
                return self._ro_cache_np[shard_path][name].reshape(-1)
            arr = _read_tensor_numpy_or_torch(shard_path, name, backend="np")
            self._ro_cache_np.setdefault(shard_path, {})[name] = arr
            return arr.reshape(-1)

    def load_block(self, name: str, start_idx: int, end_idx: int) -> ArrayLike:
        real = self._resolve_name(name)

        def _read(real_name: str) -> ArrayLike:
            meta = self._index[real_name]
            shard = meta["shard"]
            flat = self._flat_from_shard(shard, real_name)
            if self.backend == "pt":
                return flat[start_idx:end_idx].to(dtype=torch.float32)
            else:
                return flat[start_idx:end_idx].astype(np.float32, copy=True)

        try:
            return _read(real)
        except Exception as e:
            # 关键：处理 “索引里有，但文件实际缺 tensor” 的 SafetensorError
            msg = str(e).lower()
            is_missing = ("does not contain tensor" in msg) or isinstance(e, KeyError)

            if not is_missing:
                raise

            # tie-embedding fallback: lm_head <-> embed_tokens
            alt = None
            if real == "lm_head.weight" and "model.embed_tokens.weight" in self._index:
                alt = "model.embed_tokens.weight"
            elif real == "model.embed_tokens.weight" and "lm_head.weight" in self._index:
                alt = "lm_head.weight"

            if alt is not None:
                try:
                    return _read(alt)
                except Exception as e2:
                    # fallback 也读不到 -> 真缺
                    raise KeyError(name) from e2

            # 没有 fallback -> 真缺
            raise KeyError(name) from e

    def write_block(self, name: str, start_idx: int, end_idx: int, data: ArrayLike) -> None:
        if not self._writable:
            raise RuntimeError("ShardedSafeTensorsStorage opened as read-only")
        meta = self._index[name]
        shard = meta["shard"]

        if self.backend == "pt":
            if shard not in self._dirty_pt or name not in self._dirty_pt[shard]:
                full = _read_tensor_numpy_or_torch(shard, name, backend="pt", device=self.device)
                self._dirty_pt.setdefault(shard, {})[name] = full
            flat = self._dirty_pt[shard][name].reshape(-1)
            flat[start_idx:end_idx] = _to_float32_torch(data, device=self.device)
            self._dirty_pt[shard][name] = flat.reshape(self._dirty_pt[shard][name].shape)
            # 同步 RO 缓存
            self._ro_cache_pt.setdefault(shard, {})[name] = self._dirty_pt[shard][name]
        else:
            if shard not in self._dirty_np or name not in self._dirty_np[shard]:
                full = _read_tensor_numpy_or_torch(shard, name, backend="np")
                self._dirty_np.setdefault(shard, {})[name] = full
            flat = self._dirty_np[shard][name].reshape(-1)
            flat[start_idx:end_idx] = _to_float32_np(data)
            self._dirty_np[shard][name] = flat.reshape(self._dirty_np[shard][name].shape)
            self._ro_cache_np.setdefault(shard, {})[name] = self._dirty_np[shard][name]

    def flush(self, out_dir: str | None = None) -> None:
        if self._is_single:
            # 退化为单文件
            target = os.path.join(out_dir or self.dir, "model.safetensors")
            tensors: dict[str, Any] = {}
            for name, meta in self._index.items():
                shard = meta["shard"]
                if self.backend == "pt":
                    if shard in self._dirty_pt and name in self._dirty_pt[shard]:
                        tensors[name] = (
                            self._dirty_pt[shard][name]
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                        )
                    elif shard in self._ro_cache_pt and name in self._ro_cache_pt[shard]:
                        tensors[name] = (
                            self._ro_cache_pt[shard][name]
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                        )
                    else:
                        tensors[name] = (
                            _read_tensor_numpy_or_torch(shard, name, backend="pt")
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                        )
                else:
                    if shard in self._dirty_np and name in self._dirty_np[shard]:
                        tensors[name] = self._dirty_np[shard][name].astype(np.float32, copy=False)
                    elif shard in self._ro_cache_np and name in self._ro_cache_np[shard]:
                        tensors[name] = self._ro_cache_np[shard][name].astype(
                            np.float32, copy=False
                        )
                    else:
                        tensors[name] = _read_tensor_numpy_or_torch(shard, name, backend="np")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            st_save_file(tensors, target)
            self._dirty_np.clear()
            self._dirty_pt.clear()
            self._ro_cache_np.clear()
            self._ro_cache_pt.clear()
            return

        target_dir = out_dir or self.dir
        os.makedirs(target_dir, exist_ok=True)

        # shard -> keys
        shard_keys: dict[str, list[str]] = {}
        for name, meta in self._index.items():
            shard_keys.setdefault(meta["shard"], []).append(name)

        for shard, keys in shard_keys.items():
            out_path = os.path.join(target_dir, os.path.basename(shard))
            if self.backend == "pt":
                dirty = self._dirty_pt.get(shard, {})
                if dirty:
                    tensors: dict[str, Any] = {}
                    for k in keys:
                        if k in dirty:
                            tensors[k] = dirty[k].to(dtype=torch.float32, device="cpu").numpy()
                        elif shard in self._ro_cache_pt and k in self._ro_cache_pt[shard]:
                            tensors[k] = (
                                self._ro_cache_pt[shard][k]
                                .to(dtype=torch.float32, device="cpu")
                                .numpy()
                            )
                        else:
                            tensors[k] = (
                                _read_tensor_numpy_or_torch(shard, k, backend="pt")
                                .to(dtype=torch.float32, device="cpu")
                                .numpy()
                            )
                    st_save_file(tensors, out_path)
                else:
                    # 未修改：若 out_dir 不同则复制
                    if os.path.abspath(target_dir) != os.path.abspath(self.dir):
                        with open(shard, "rb") as sf, open(out_path, "wb") as df:
                            df.write(sf.read())
            else:
                dirty = self._dirty_np.get(shard, {})
                if dirty:
                    tensors: dict[str, Any] = {}
                    for k in keys:
                        if k in dirty:
                            tensors[k] = dirty[k].astype(np.float32, copy=False)
                        elif shard in self._ro_cache_np and k in self._ro_cache_np[shard]:
                            tensors[k] = self._ro_cache_np[shard][k].astype(np.float32, copy=False)
                        else:
                            tensors[k] = _read_tensor_numpy_or_torch(shard, k, backend="np")
                    st_save_file(tensors, out_path)
                else:
                    if os.path.abspath(target_dir) != os.path.abspath(self.dir):
                        with open(shard, "rb") as sf, open(out_path, "wb") as df:
                            df.write(sf.read())

        self._dirty_np.clear()
        self._dirty_pt.clear()
        self._ro_cache_np.clear()
        self._ro_cache_pt.clear()


# ------------------ DeltaProvider & Implementations ------------------


class DeltaProvider(Protocol):
    backend: str

    def delta_block(
        self, tensor_name: str, start_idx: int, end_idx: int, base_block: ArrayLike
    ) -> ArrayLike:
        """返回该张量块的 ΔW（与 base_block 同类型/设备）。"""
        ...


# NumPy providers
class WeightsAsDeltaProviderNP:
    backend = "np"

    def __init__(self, expert_storage: StorageBase):
        self.es = expert_storage

    def delta_block(self, tensor_name: str, s: int, e: int, base_block: np.ndarray) -> np.ndarray:
        try:
            ex_blk = self.es.load_block(tensor_name, s, e)  # np
        except KeyError:
            # expert 缺 tensor：视为 Δ=0
            return np.zeros_like(base_block, dtype=np.float32)
        return _to_float32_np(ex_blk) - _to_float32_np(base_block)


class DirectDeltaStorageProviderNP:
    backend = "np"

    def __init__(self, delta_storage: StorageBase):
        self.ds = delta_storage

    def delta_block(self, tensor_name: str, s: int, e: int, base_block: np.ndarray) -> np.ndarray:
        return _to_float32_np(self.ds.load_block(tensor_name, s, e))


class LoRAProviderNP:
    backend = "np"

    def __init__(self, lora_map: dict[str, dict[str, Any]]):
        self.lora_map = lora_map

    def delta_block(self, tensor_name: str, s: int, e: int, base_block: np.ndarray) -> np.ndarray:
        info = self.lora_map.get(tensor_name)
        if info is None:
            return np.zeros_like(base_block, dtype=np.float32)
        A = _to_float32_np(info["A"])
        B = _to_float32_np(info["B"])
        alpha = float(info.get("alpha", 1.0))
        flat = (A @ B.T * alpha).reshape(-1).astype(np.float32, copy=False)
        return flat[s:e]


# Torch providers
class WeightsAsDeltaProviderPT:
    backend = "pt"

    def __init__(self, expert_storage: StorageBase, device: str | None = None):
        _ensure_torch()
        self.es = expert_storage
        self.device = device

    def delta_block(self, tensor_name: str, s: int, e: int, base_block: torch.Tensor) -> torch.Tensor:
        try:
            ex_blk = self.es.load_block(tensor_name, s, e)  # torch
        except KeyError:
            # expert 缺 tensor：视为 Δ=0（与NP版本对齐）
            return torch.zeros_like(base_block, dtype=torch.float32, device=base_block.device)

        return _to_float32_torch(ex_blk, device=self.device) - _to_float32_torch(
            base_block, device=self.device
        )


class DirectDeltaStorageProviderPT:
    backend = "pt"

    def __init__(self, delta_storage: StorageBase, device: str | None = None):
        _ensure_torch()
        self.ds = delta_storage
        self.device = device

    def delta_block(
        self, tensor_name: str, s: int, e: int, base_block: torch.Tensor
    ) -> torch.Tensor:
        return _to_float32_torch(self.ds.load_block(tensor_name, s, e), device=self.device)


class LoRAProviderPT:
    backend = "pt"

    def __init__(self, lora_map: dict[str, dict[str, Any]], device: str | None = None):
        _ensure_torch()
        self.device = device
        # 预构造 torch 参数（float32）
        self.lora_torch: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in lora_map.items():
            A = _to_float32_torch(v["A"], device=device)
            B = _to_float32_torch(v["B"], device=device)
            alpha = float(v.get("alpha", 1.0))
            self.lora_torch[k] = {"A": A, "B": B, "alpha": alpha}

    def delta_block(
        self, tensor_name: str, s: int, e: int, base_block: torch.Tensor
    ) -> torch.Tensor:
        _ensure_torch()
        info = self.lora_torch.get(tensor_name)
        if info is None:
            return torch.zeros_like(base_block, dtype=torch.float32, device=base_block.device)
        A, B, alpha = info["A"], info["B"], float(info["alpha"])
        flat = (A @ B.t() * alpha).reshape(-1)
        return flat[s:e]


# ------------------ DeltaStorage（ΔW 容器） ------------------


class DeltaStorage(ShardedSafeTensorsStorage):
    """ΔW 容器：结构与分片 safetensors 一致。"""

    pass

# ------------------ Delta Iterator（统一迭代入口） ------------------

class DeltaIterator:
    """
    统一从“专家 Provider（weights/delta/lora）”读取 ΔW 的迭代器。
    - 构造时注入 base_storage 与 {expert_id: provider}
    - 提供:
        delta_for(expert_id, tensor_name, start, end) -> 与 base_block 同后端/设备的 ΔW
        iter_all(expert_ids, tensor_name, start, end) -> 逐个专家返回 (expert_id, ΔW)
    """
    def __init__(
        self,
        base_storage: StorageBase,
        providers: dict[str, DeltaProvider],
        backend: str = "np",
        device: str | None = None,
    ):
        self.base = base_storage
        self.providers = providers
        self.backend = backend
        self.device = device
        self.base_storage = base_storage
        

    def delta_for(self, expert_id: str, tensor_name: str, start: int, end: int) -> ArrayLike:
        prov = self.providers.get(expert_id)
        if prov is None:
            raise KeyError(f"DeltaIterator: provider for expert_id={expert_id} not found")
        base_block = self.base.load_block(tensor_name, start, end)
        return prov.delta_block(tensor_name, start, end, base_block)

    def iter_all(
        self, expert_ids: list[str], tensor_name: str, start: int, end: int
    ) -> Iterable[tuple[str, ArrayLike]]:
        for eid in expert_ids:
            yield eid, self.delta_for(eid, tensor_name, start, end)


def iter_deltas(
    iterator: "DeltaIterator",
    expert_ids: list[str],
    tensor_name: str,
    start: int,
    end: int,
) -> Iterable[tuple[str, ArrayLike]]:
    """
    语法糖：基于 DeltaIterator 的统一入口；与历史代码风格一致。
    返回 (expert_id, ΔW) 可直接喂给 AVG/TIES/DARE。
    """
    yield from iterator.iter_all(expert_ids, tensor_name, start, end)


# ------------------ LoRA Loader ------------------


def load_lora_from_dir(path: str, alpha: float = 1.0) -> dict[str, dict[str, Any]]:
    """
    支持两类常见布局：
    1) adapter.safetensors：键名 'lora_A.<tensor>' / 'lora_B.<tensor>' 或 '<tensor>.lora_A' / '.lora_B'
    2) 目录下 .npy：A_<safe_name>.npy / B_<safe_name>.npy（safe_name 用 __ 替换 .）
    返回：{tensor_name: {"A": np.ndarray, "B": np.ndarray, "alpha": float}}
    """
    lora_map: dict[str, dict[str, Any]] = {}

    # npy 情况
    A_paths = glob.glob(os.path.join(path, "A_*.npy"))
    B_paths = glob.glob(os.path.join(path, "B_*.npy"))
    if A_paths and B_paths:

        def unsanitize(n: str) -> str:
            return n.replace("__", ".")

        A_dict = {
            unsanitize(os.path.splitext(os.path.basename(p))[0][2:]): np.load(p).astype(np.float32)
            for p in A_paths
        }
        B_dict = {
            unsanitize(os.path.splitext(os.path.basename(p))[0][2:]): np.load(p).astype(np.float32)
            for p in B_paths
        }
        for k in set(A_dict) & set(B_dict):
            lora_map[k] = {"A": A_dict[k], "B": B_dict[k], "alpha": alpha}
        if lora_map:
            return lora_map

    # adapter.safetensors
    st_paths = glob.glob(os.path.join(path, "*.safetensors"))
    for sp in st_paths:
        try:
            with st_safe_open(sp, framework="numpy") as f:
                keys = list(f.keys())
                A1 = [k for k in keys if k.startswith("lora_A.")]
                B1 = [k for k in keys if k.startswith("lora_B.")]
                A2 = [k for k in keys if k.endswith(".lora_A")]
                B2 = [k for k in keys if k.endswith(".lora_B")]
                if A1 and B1:
                    names = set(k.split("lora_A.")[1] for k in A1) & set(
                        k.split("lora_B.")[1] for k in B1
                    )
                    for name in names:
                        A = f.get_tensor(f"lora_A.{name}").astype(np.float32)
                        B = f.get_tensor(f"lora_B.{name}").astype(np.float32)
                        lora_map[name] = {"A": A, "B": B, "alpha": alpha}
                if A2 and B2:

                    def base(n: str) -> str:
                        return n[:-7]

                    names = set(base(k) for k in A2) & set(base(k) for k in B2)
                    for name in names:
                        A = f.get_tensor(f"{name}.lora_A").astype(np.float32)
                        B = f.get_tensor(f"{name}.lora_B").astype(np.float32)
                        lora_map[name] = {"A": A, "B": B, "alpha": alpha}
        except Exception:
            # fallback to torch then convert
            _ensure_torch()
            with st_safe_open(sp, framework="pt") as f:
                keys = list(f.keys())
                A1 = [k for k in keys if k.startswith("lora_A.")]
                B1 = [k for k in keys if k.startswith("lora_B.")]
                A2 = [k for k in keys if k.endswith(".lora_A")]
                B2 = [k for k in keys if k.endswith(".lora_B")]
                if A1 and B1:
                    names = set(k.split("lora_A.")[1] for k in A1) & set(
                        k.split("lora_B.")[1] for k in B1
                    )
                    for name in names:
                        A = (
                            f.get_tensor(f"lora_A.{name}")
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                        )
                        B = (
                            f.get_tensor(f"lora_B.{name}")
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                        )
                        lora_map[name] = {"A": A, "B": B, "alpha": alpha}
                if A2 and B2:

                    def base(n: str) -> str:
                        return n[:-7]

                    names = set(base(k) for k in A2) & set(base(k) for k in B2)
                    for name in names:
                        A = (
                            f.get_tensor(f"{name}.lora_A")
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                        )
                        B = (
                            f.get_tensor(f"{name}.lora_B")
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                        )
                        lora_map[name] = {"A": A, "B": B, "alpha": alpha}
    return lora_map


# ------------------ Coverage / Touchmap (NEW) ------------------


def _bitmap_set(bitarr: bytearray, idx: int) -> None:
    byte_i = idx // 8
    bit_i = idx % 8
    bitarr[byte_i] |= 1 << bit_i


def _bitmap_new(n_bits: int) -> bytearray:
    n_bytes = (n_bits + 7) // 8
    return bytearray(n_bytes)


def _bitmap_ratio(bitarr: bytes, n_bits: int) -> float:
    if n_bits <= 0:
        return 0.0
    cnt = 0
    for b in bitarr:
        cnt += int(b).bit_count()
    return float(cnt) / float(n_bits)


@dataclass
class CoverageResult:
    touchmaps: dict[str, bytes]  # tensor_name -> bitmap (bytes)
    coverage_json: dict[str, dict[str, float]]  # {"tensor_name":{"ratio":...}}


def _num_blocks_from_shape(shape: tuple[int, ...], block_size: int) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(math.ceil(n / float(block_size)))


def _list_tensor_names(storage: StorageBase) -> list[str]:
    """
    仅获取张量名列表，尽量不触发整张量加载：
    - 优先使用存储实现的 _index / tensors 字段
    - 最后退化到 iter_tensors()（小模型/单测可接受）
    """
    # ShardedSafeTensorsStorage / SafeTensorsStorage
    idx = getattr(storage, "_index", None)
    if isinstance(idx, dict) and idx:
        return list(idx.keys())
    # FakeStorage
    tdict = getattr(storage, "tensors", None)
    if isinstance(tdict, dict) and tdict:
        return list(tdict.keys())
    # 兜底：可能会触发加载，但仅在 Fake/NPZ 场景出现
    names: list[str] = []
    try:
        for n, _ in storage.iter_tensors():
            names.append(n)
    except Exception:
        pass
    return names


def compute_delta_coverage(
    provider: DeltaProvider,
    base_storage: StorageBase,
    block_size: int = 4096,
    backend: str = "np",
    device: str | None = None,
    only_tensors: list[str] | None = None,
) -> CoverageResult:
    """
    统一统计任意 DeltaProvider 的块级触达：
      - 对 base_storage 的每个张量，逐块取 delta_block(start,end, base_block)
      - 判定“该块是否 touched”：np.any(delta != 0) / torch.any(delta != 0)
    返回：
      - touchmaps: {tensor_name: bitmap(bytes)}  1bit/块
      - coverage_json: {"tensor_name":{"ratio": touched_blocks/total_blocks}}
    说明：
      - 这是“下限估计”：只要 delta 非全零即视为触达；精度足够用于 Planner 与可视化。
      - 对 LoRAProvider：其 delta 是即时合成（A@B），此法同样适用。
    """
    tensors = _list_tensor_names(base_storage)
    if only_tensors:
        allow = set(only_tensors)
        tensors = [t for t in tensors if t in allow]

    touchmaps: dict[str, bytes] = {}
    coverage: dict[str, dict[str, float]] = {}

    for tname in tensors:
        shape = base_storage.tensor_shape(tname)
        n_elem = int(np.prod(shape))
        n_blocks = _num_blocks_from_shape(shape, block_size)
        bm = _bitmap_new(n_blocks)
        touched = 0
        bid = 0
        for start in range(0, n_elem, block_size):
            end = min(start + block_size, n_elem)
            base_blk = base_storage.load_block(tname, start, end)
            delta = provider.delta_block(tname, start, end, base_blk)
            # 判断非零
            if _TORCH_OK and isinstance(delta, torch.Tensor):
                is_touched = bool((delta != 0).any().item())
            else:
                darr = np.asarray(delta)
                is_touched = bool(np.any(darr != 0))
            if is_touched:
                _bitmap_set(bm, bid)
                touched += 1
            bid += 1
        touchmaps[tname] = bytes(bm)
        coverage[tname] = {"ratio": float(touched / max(1, n_blocks))}
    return CoverageResult(touchmaps=touchmaps, coverage_json=coverage)

# ------------------ DB-driven DeltaIterator builder (NEW) ------------------

def _open_storage_from_uri(
    uri: str,
    writable: bool = False,
    backend: str = "np",
    device: str | None = None,
) -> StorageBase:
    """
    约定：
      - 若 uri 指向目录（含 model.safetensors 或 model-00001-of-xxxx.safetensors），用 ShardedSafeTensorsStorage
      - 若 uri 指向单个 .safetensors 文件，用 SafeTensorsStorage
      - 若 uri 指向 .npz，仅用于小测试（DiskNPZStorage）
    """
    if os.path.isdir(uri):
        return ShardedSafeTensorsStorage(uri, writable=writable, backend=backend, device=device)
    if uri.endswith(".safetensors"):
        return SafeTensorsStorage(uri, writable=writable, backend=backend, device=device)
    if uri.endswith(".npz"):
        # 仅 numpy
        return DiskNPZStorage(uri, writable=writable, backend="np")
    # 兜底：当作目录
    return ShardedSafeTensorsStorage(uri, writable=writable, backend=backend, device=device)


def delta_iterator_from_db(
    con,  # sqlite3.Connection（避免循环依赖，不做类型导入）
    base_model_id: str,
    expert_ids: list[str],
    expert_kind: str,   # "weights" | "delta" | "lora"
    backend: str = "np",
    device: str | None = None,
):
    """
    从 DB 读取 base/expert 的 URI，自动选择 Provider，构造 DeltaIterator。
    - weights: expert_id 来自 models 表，Provider=WeightsAsDeltaProvider*
    - delta:   expert_id 指向“ΔW 容器”的 safetensors 目录/文件，Provider=DirectDeltaStorageProvider*
              （你可以把这些容器以“模型”身份注册，也可以独立存储，只要 uri 可用即可）
    - lora:    expert_id 来自 adapters 表（type='lora'），从 uri 目录加载 LoRA（A/B），Provider=LoRAProvider*
    返回: (iterator, used_expert_ids)  —— 其中 used_expert_ids 是成功构造的专家子集（uri 缺失的会被忽略）
    """
    # —— 本地导入，避免硬依赖 catalog.db 顶层
    from mergepipe.catalog import db as catalog_db  # type: ignore

    base_uri = catalog_db.get_model_uri(con, base_model_id)
    if not base_uri:
        raise RuntimeError(f"Base model `{base_model_id}` has no uri in DB.")
    base_storage = _open_storage_from_uri(base_uri, writable=False, backend=backend, device=device)

    providers: dict[str, DeltaProvider] = {}
    used: list[str] = []

    if expert_kind == "weights":
        m_uris = catalog_db.get_models_uri(con, expert_ids)
        for eid in expert_ids:
            uri = m_uris.get(eid)
            if not uri:
                continue
            es = _open_storage_from_uri(uri, writable=False, backend=backend, device=device)
            if backend == "pt":
                providers[eid] = WeightsAsDeltaProviderPT(es, device=device)
            else:
                providers[eid] = WeightsAsDeltaProviderNP(es)
            used.append(eid)

    elif expert_kind == "delta":
        # 这里假设 expert_ids 也在 models 表中（或你改成自己维护的一张 diffs/adapters 表也行），只要能拿到 uri 即可
        m_uris = catalog_db.get_models_uri(con, expert_ids)
        for eid in expert_ids:
            uri = m_uris.get(eid)
            if not uri:
                continue
            ds = _open_storage_from_uri(uri, writable=False, backend=backend, device=device)
            if backend == "pt":
                providers[eid] = DirectDeltaStorageProviderPT(ds, device=device)
            else:
                providers[eid] = DirectDeltaStorageProviderNP(ds)
            used.append(eid)

    elif expert_kind == "lora":
        # lora 情况：expert_ids 是 adapter_id；从 adapters 表拿 uri，再从目录加载 A/B
        adapters = catalog_db.get_adapters(con, expert_ids)
        for eid in expert_ids:
            meta = adapters.get(eid)
            if not meta or (meta.get("type") != "lora"):
                continue
            uri = meta.get("uri")
            if not uri:
                continue
            lora_map = load_lora_from_dir(uri, alpha=1.0)
            if not lora_map:
                continue
            if backend == "pt":
                providers[eid] = LoRAProviderPT(lora_map, device=device)
            else:
                providers[eid] = LoRAProviderNP(lora_map)
            used.append(eid)
    else:
        raise ValueError(f"Unknown expert_kind: {expert_kind}")

    iterator = DeltaIterator(
        base_storage=base_storage, providers=providers, backend=backend, device=device
    )
    return iterator, used


def iter_deltas_db(
    con,
    base_model_id: str,
    expert_ids: list[str],
    expert_kind: str,    # "weights" | "delta" | "lora"
    tensor_name: str,
    start: int,
    end: int,
    backend: str = "np",
    device: str | None = None,
):
    """
    统一 DB 入口：一行代码把 (base, experts) 路通，返回 (expert_id, ΔW) 序列。
    """
    iterator, used = delta_iterator_from_db(
        con,
        base_model_id=base_model_id,
        expert_ids=expert_ids,
        expert_kind=expert_kind,
        backend=backend,
        device=device,
    )
    for eid in used:
        yield eid, iterator.delta_for(eid, tensor_name, start, end)

@dataclass
class IOBudgetGate:
    """
    一个轻量级预算门：
    - budget_bytes: 总预算（bytes）
    - used_bytes:   已使用（bytes）
    - skipped_reads: 因预算不足而跳过的“读盘”次数（仅计数）
    """
    budget_bytes: int
    used_bytes: int = 0
    skipped_reads: int = 0

    @property
    def remaining_bytes(self) -> int:
        return max(0, int(self.budget_bytes - self.used_bytes))

    def try_charge(self, nbytes: int) -> bool:
        """
        若剩余预算足够则扣费并返回 True，否则不扣费返回 False。
        """
        nbytes = int(nbytes)
        if nbytes <= 0:
            return True
        if self.used_bytes + nbytes <= self.budget_bytes:
            self.used_bytes += nbytes
            return True
        return False

    def mark_skipped(self) -> None:
        self.skipped_reads += 1


def estimate_block_bytes(n_elems: int, dtype_bytes: int = 4) -> int:
    """
    估计一个 block 的 IO 字节数。
    目前 merge 路径里我们统一 float32 计算，所以默认 4 bytes/elem。
    """
    n_elems = int(n_elems)
    if n_elems <= 0:
        return 0
    return n_elems * int(dtype_bytes)


def zeros_block(
    backend: str,
    n_elems: int,
    device: str | None = None,
) -> ArrayLike:
    """
    不读盘，构造一个“长度为 n_elems 的 float32 零向量 block”：
    - backend="np" -> np.ndarray(float32)
    - backend="pt" -> torch.Tensor(float32, device=device or cpu)
    """
    n_elems = int(n_elems)
    if backend == "pt":
        _ensure_torch()
        dev = device or "cpu"
        return torch.zeros((n_elems,), dtype=torch.float32, device=dev)
    return np.zeros((n_elems,), dtype=np.float32)


def load_block_budgeted(
    storage: StorageBase,
    name: str,
    start_idx: int,
    end_idx: int,
    gate: IOBudgetGate | None,
    *,
    backend: str | None = None,
    device: str | None = None,
    dtype_bytes: int = 4,
) -> ArrayLike:
    """
    带 budget 的块读取：
    - gate is None: 直接读
    - gate not None:
        * 先按 (end-start)*dtype_bytes 估算一次 block 成本
        * 若预算足够 -> 扣费 & storage.load_block()
        * 若预算不足 -> 不读盘，直接返回 zeros_block，并 gate.mark_skipped()
    注意：
    - 这不是“精确 IO 计量”（因为 safetensors 可能整 tensor 读取），
        但它提供了一个 **可控的裁剪机制**：预算不足时直接不读盘。
    - 要让 budget 真正影响 IO，merge_engine.py 必须在读取 base/expert block 时用这个函数。
    """
    s = int(start_idx)
    e = int(end_idx)
    if e <= s:
        # 退化：空块
        b = backend or getattr(storage, "backend", "np")
        return zeros_block(b, 0, device=device)

    b = backend or getattr(storage, "backend", "np")
    n_elems = e - s
    est = estimate_block_bytes(n_elems, dtype_bytes=dtype_bytes)

    if gate is None:
        return storage.load_block(name, s, e)

    if gate.try_charge(est):
        return storage.load_block(name, s, e)

    # 预算不足：跳过读盘，返回零块
    gate.mark_skipped()
    return zeros_block(b, n_elems, device=device)


def io_budget_summary(gate: IOBudgetGate | None) -> dict[str, float]:
    """
    统一输出一个可写入 exec_stats 的摘要（给 merge_engine/planner 记录用）。
    """
    if gate is None:
        return {"budget_mb": 0.0, "used_mb": 0.0, "skipped_reads": 0.0}
    return {
        "budget_mb": float(gate.budget_bytes) / (1024.0 * 1024.0),
        "used_mb": float(gate.used_bytes) / (1024.0 * 1024.0),
        "skipped_reads": float(gate.skipped_reads),
    }