"""JAX backend selection: Apple Metal, CUDA, or CPU.

Auto-detection order follows the hardware actually present:

* Apple Silicon (Darwin/arm64) -> ``METAL`` via ``jax-metal``, else CPU.
  macOS has no CUDA support, so CUDA is never probed there.
* Everything else             -> ``cuda`` when an NVIDIA GPU is visible, else CPU.

Metal lacks a QR kernel, so orthogonal initialisers must be evaluated on CPU
and the resulting parameters transferred to the accelerator (see `init_device`).
"""

from __future__ import annotations

import platform
import sys

import jax

# Platform strings to try for each logical backend, in order. jaxlib registers
# the Metal plugin as 'METAL' and renamed the NVIDIA backend 'gpu' -> 'cuda'.
_PLATFORM_ALIASES: dict[str, tuple[str, ...]] = {
    'metal': ('METAL', 'metal'),
    'cuda':  ('cuda', 'gpu'),
    'cpu':   ('cpu',),
}


def is_apple_silicon() -> bool:
    """True on an arm64 Mac (M1/M2/M3/M4 family)."""
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


def _probe(backend: str) -> jax.Device | None:
    """First device of `backend`, or None when the backend is unavailable.

    `jax.devices` raises RuntimeError when a platform is unknown or its plugin
    fails to initialise; a failed probe leaves the other backends usable.
    """
    for name in _PLATFORM_ALIASES.get(backend, (backend,)):
        try:
            devices = jax.devices(name)
        except RuntimeError:
            continue
        if devices:
            return devices[0]
    return None


def auto_priority() -> tuple[str, ...]:
    """Backends to try, best first, for the current machine."""
    if is_apple_silicon():
        return ('metal', 'cpu')
    return ('cuda', 'cpu')


def select_device(prefer: str | None = None) -> jax.Device:
    """Return the compute device and make it the JAX default.

    `prefer` forces a backend ('metal', 'cuda'/'gpu', 'cpu'); None auto-detects.
    An unavailable forced backend warns and falls back to auto-detection rather
    than aborting the run.
    """
    if prefer:
        requested = 'cuda' if prefer == 'gpu' else prefer
        device = _probe(requested)
        if device is not None:
            return _activate(device)
        print(f"WARNING: backend '{prefer}' is unavailable on this machine; "
              f"falling back to auto-detection.", file=sys.stderr)

    for backend in auto_priority():
        device = _probe(backend)
        if device is not None:
            return _activate(device)

    # jax always ships a CPU backend, so this is unreachable in practice.
    raise RuntimeError('No usable JAX backend found (not even CPU).')


def _activate(device: jax.Device) -> jax.Device:
    """Make `device`'s platform the default so implicit placement matches it."""
    jax.config.update('jax_platform_name', device.platform)
    return device


def backend_name(device: jax.Device) -> str:
    """Normalised backend label: 'cuda', 'metal' or 'cpu'."""
    p = device.platform.lower()
    if p == 'gpu':
        return 'cuda'
    if p == 'metal':
        return 'metal'
    return p


def init_device(device: jax.Device) -> jax.Device:
    """Device on which to evaluate parameter initialisers."""
    if backend_name(device) == 'metal':
        return jax.devices('cpu')[0]
    return device


def describe(device: jax.Device) -> str:
    return f"{backend_name(device)} ({device.device_kind})"
