#!/usr/bin/env python3
"""
Backbone factory for RAPIDS.

RAPIDS is built around the pretrained FAIRChem **UMA omol** MLIP head, but the
paper's cross-backbone diagnostic (see ``part3_results.tex`` §3.6) also runs the
same RAPIDS-committed geometries through two alternative omol-trained backbones,
**MACE-omol** and **ORB-omol**, both as fixed-geometry single points and at
relaxation. This module provides a single entry point,
:func:`get_backbone_calculator`, that returns an ASE-compatible calculator for
any of those three backbones so the core flow (LBFGS relaxation + scoring) can
be driven by a backbone other than UMA without changing the rest of the stack.

Design notes
------------
* **Default is UMA** — existing behavior is unchanged. Selecting ``"uma"``
  reproduces exactly what ``smart_fairchem_flow.py`` did before (same
  ``pretrained_mlip.get_predict_unit`` + ``FAIRChemCalculator`` call, including
  per-composition turbo mode).
* **Guarded imports** — ``mace`` / ``orb-models`` are heavy optional deps and
  are *not* required to run RAPIDS with UMA. They are imported lazily, only when
  the corresponding backbone is actually selected, and a missing package raises
  a clear ``ImportError`` with a ``pip install`` hint rather than breaking module
  import.
* **Return contract** — every branch returns an object implementing the ASE
  ``Calculator`` interface, i.e. it can be assigned to ``atoms.calc`` and supply
  ``get_potential_energy`` / ``get_forces``. The caller is responsible for
  setting ``atoms.info['charge']`` / ``atoms.info['spin']`` where the backbone
  needs them (UMA omol does; this is handled in the core flow already).

This file is intentionally flat-imported (no package), matching the rest of the
RAPIDS source tree.
"""

from __future__ import annotations

from typing import Optional

# Canonical backbone identifier -> human description.
SUPPORTED_BACKBONES = {
    "uma": "FAIRChem UMA omol head (default; pretrained_mlip + FAIRChemCalculator).",
    "mace-omol": "MACE-MH-1 with the omol head (OMol25 / wB97M-VV10) via mace.calculators.mace_mp.",
    "orb-omol": "ORB-v3 conservative omol checkpoint via orb_models.forcefield + ORBCalculator.",
}

# Default checkpoint / model names per backbone. These are overridable via the
# ``model_name`` argument so a user can point at a newer/local checkpoint
# without editing this file. They match the production cross-MLIP drivers
# (RAPIDS_NCI_crossMLIP_L1_L2_L3_supp/raw_L1/{mace_sp.py,orb_sp.py}).
DEFAULT_MODEL_NAMES = {
    "uma": "uma-s-1p2",
    # MACE: the MACE-MH-1 checkpoint loaded via mace_mp(model="mh-1", ...). The
    # omol head (OMol25 / wB97M-VV10 trained) is selected separately via the
    # ``head`` argument (default DEFAULT_MACE_HEAD below); ``mh-1`` is the
    # checkpoint, not the head.
    "mace-omol": "mh-1",
    # ORB: the conservative omol foundation checkpoint, exposed as the loader
    # ``orb_v3_conservative_omol`` in orb_models.forcefield.pretrained.
    "orb-omol": "orb_v3_conservative_omol",
}

# MACE head used for the omol backbone. ``mh-1`` ships multiple heads; the omol
# head is the OMol25 / wB97M-VV10 one used throughout RAPIDS. Overridable via the
# ``head`` argument to :func:`get_backbone_calculator`.
DEFAULT_MACE_HEAD = "omol"


def normalize_backbone(backbone: Optional[str]) -> str:
    """Normalize a user-supplied backbone string to a canonical key.

    Accepts ``None`` (-> ``"uma"``), case-insensitive names, and a few common
    aliases (``"mace"`` -> ``"mace-omol"``, ``"orb"`` -> ``"orb-omol"``,
    underscores treated like hyphens).

    Raises:
        ValueError: if the backbone is not one of ``SUPPORTED_BACKBONES``.
    """
    if backbone is None:
        return "uma"
    key = backbone.strip().lower().replace("_", "-")
    aliases = {
        "mace": "mace-omol",
        "mace-off": "mace-omol",
        "maceomol": "mace-omol",
        "orb": "orb-omol",
        "orbomol": "orb-omol",
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_BACKBONES))}."
        )
    return key


def _make_uma_calculator(device: str, task_name: str, model_name: str,
                         inference_settings: str):
    """Build the default UMA omol calculator (FAIRChem).

    This mirrors the original RAPIDS load path so the UMA default is byte-for-byte
    unchanged in behavior.
    """
    try:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "The 'uma' backbone requires fairchem-core. "
            "Install it with: pip install fairchem-core"
        ) from exc

    predictor = pretrained_mlip.get_predict_unit(
        model_name,
        device=device,
        inference_settings=inference_settings,
    )
    return FAIRChemCalculator(predictor, task_name=task_name)


def _make_mace_calculator(device: str, model_name: str, head: str):
    """Build a MACE-omol ASE calculator (MACE-MH-1, omol head).

    This is the production-tested load path used by the RAPIDS cross-MLIP
    drivers: ``mace.calculators.mace_mp`` with the ``mh-1`` checkpoint and the
    ``omol`` head (OMol25 / wB97M-VV10 trained), float64, and dispersion turned
    off because the omol head already carries the VV10 nonlocal correlation
    (adding D3BJ would double-count dispersion).

    Charge/spin are *not* passed here: the omol head conditions on them through
    ``atoms.info['charge']`` / ``atoms.info['spin']`` at evaluation time, which
    the core flow sets before calling ``get_potential_energy`` (see
    ``smart_fairchem_flow.smart_optimize``).

    Args:
        device: Torch device string ("cuda" / "cpu").
        model_name: MACE checkpoint name (default "mh-1"). An existing local
            checkpoint file path is also accepted and loaded via the generic
            ``MACECalculator``.
        head: MACE head to condition on (default "omol").
    """
    try:
        from mace.calculators import mace_mp, MACECalculator  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'mace-omol' backbone requires the MACE package. "
            "Install it with: pip install mace-torch"
        ) from exc

    # If the caller passed an explicit checkpoint file, use the generic
    # MACECalculator; otherwise use the foundation-model loader by name.
    import os
    if model_name and os.path.exists(model_name):
        return MACECalculator(model_paths=model_name, device=device,
                              default_dtype="float64")
    # mace_mp loads the named MACE foundation checkpoint (e.g. "mh-1") with the
    # requested head. dispersion=False: omol head has VV10 built in (no D3BJ).
    return mace_mp(
        model=model_name,
        head=head,
        dispersion=False,
        default_dtype="float64",
        device=device,
    )


def _make_orb_calculator(device: str, model_name: str):
    """Build an ORB-omol ASE calculator (ORB-v3 conservative omol).

    This is the production-tested load path used by the RAPIDS cross-MLIP
    drivers::

        from orb_models.forcefield import pretrained as orb_pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orb_model = orb_pretrained.orb_v3_conservative_omol(device=device)
        calc = ORBCalculator(orb_model, device=device)

    The default checkpoint is the *conservative* ORB-v3 omol model (energy =>
    energy-gradient forces, i.e. forces are the exact gradient of the energy),
    at the model's default precision (float32-high). Charge/spin are conditioned
    through ``atoms.info['charge']`` / ``atoms.info['spin']`` at evaluation time
    (set by the core flow before ``get_potential_energy``), not here.

    ``pretrained`` exposes one loader function per registered checkpoint, so the
    requested ``model_name`` is resolved to ``getattr(pretrained, name)`` — this
    keeps the default (``orb_v3_conservative_omol``) and any user override (e.g.
    a different ORB-v3 variant) on a single code path.
    """
    try:
        from orb_models.forcefield import pretrained as orb_pretrained
        from orb_models.forcefield.calculator import ORBCalculator
    except ImportError as exc:
        raise ImportError(
            "The 'orb-omol' backbone requires the orb-models package. "
            "Install it with: pip install orb-models"
        ) from exc

    # Registry loader names use underscores; normalize a hyphenated override.
    loader_name = model_name.replace("-", "_")
    if not hasattr(orb_pretrained, loader_name):
        available = [n for n in dir(orb_pretrained) if not n.startswith("_")]
        raise ValueError(
            f"ORB checkpoint '{model_name}' (loader '{loader_name}') not found in "
            f"orb_models.forcefield.pretrained. Available: {', '.join(available)}"
        )
    orb_model = getattr(orb_pretrained, loader_name)(device=device)
    return ORBCalculator(orb_model, device=device)


def get_backbone_calculator(
    backbone: str = "uma",
    device: str = "cuda",
    task_name: str = "omol",
    model_name: Optional[str] = None,
    inference_settings: str = "turbo",
    head: Optional[str] = None,
):
    """Return an ASE-compatible calculator for the requested RAPIDS backbone.

    Args:
        backbone: One of ``{"uma", "mace-omol", "orb-omol"}`` (case-insensitive;
            ``"mace"``/``"orb"`` aliases accepted). Defaults to ``"uma"`` so the
            existing UMA behavior is preserved.
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        task_name: FAIRChem task head for the UMA backbone (default ``"omol"``).
            Ignored by MACE/ORB.
        model_name: Optional checkpoint/model-name override. If ``None``, the
            per-backbone default from :data:`DEFAULT_MODEL_NAMES` is used
            (``uma-s-1p2`` / ``mh-1`` / ``orb_v3_conservative_omol``). For
            MACE/ORB an existing file path is also accepted (loads a local
            checkpoint).
        inference_settings: FAIRChem inference mode for UMA (default ``"turbo"``,
            matching the core flow's per-composition turbo caching). Ignored by
            MACE/ORB.
        head: MACE head to condition on (default :data:`DEFAULT_MACE_HEAD` =
            ``"omol"``). Only used by the ``mace-omol`` backbone; ignored by UMA
            and ORB.

    Returns:
        An object implementing the ASE ``Calculator`` interface
        (``atoms.calc = <returned>``).

    Charge/spin note:
        The omol heads (UMA, MACE-MH-1/omol, ORB-v3 omol) all condition on
        per-structure charge and multiplicity supplied via
        ``atoms.info['charge']`` and ``atoms.info['spin']`` at evaluation time.
        This factory only builds the calculator; the caller (the core flow)
        sets those keys on each ``Atoms`` object before evaluation, so charge
        and spin reach MACE/ORB exactly as they reach UMA.

    Raises:
        ValueError: if ``backbone`` is not supported.
        ImportError: if the selected backbone's package is not installed (the
            message includes the appropriate ``pip install`` hint). UMA-only
            users never trigger the MACE/ORB import paths.
    """
    key = normalize_backbone(backbone)
    resolved_model = model_name or DEFAULT_MODEL_NAMES[key]

    if key == "uma":
        return _make_uma_calculator(
            device=device,
            task_name=task_name,
            model_name=resolved_model,
            inference_settings=inference_settings,
        )
    if key == "mace-omol":
        return _make_mace_calculator(
            device=device,
            model_name=resolved_model,
            head=head or DEFAULT_MACE_HEAD,
        )
    if key == "orb-omol":
        return _make_orb_calculator(device=device, model_name=resolved_model)

    # Unreachable: normalize_backbone already validated the key.
    raise ValueError(f"Unhandled backbone '{key}'")
