"""Video motion capture on SAM 3D Body: video -> animated MHR character GLB.

Per frame the estimator regresses the full MHR state — joint global
rotations (body + hands), camera translation, and 72 face-expression
coefficients (unlocked from the upstream head at image-build time; see
deploy.py's sed step). Temporal smoothing (One-Euro) is applied to the
quaternions, root translation, and expression track, and the result drives
the skinned MHR character exported by gltf_writer.

Self-contained by plugin convention (no imports from other plugins). The MHR
character bundle is baked by ``extract_mhr.py`` into ``/models/mhr/``.
"""

from __future__ import annotations

import math
import os
import tempfile
from functools import lru_cache

import cv2
import numpy as np

from gltf_writer import skinned_character_glb

MOCAP_FPS = float(os.environ.get("MOCAP_FPS", "24"))
MOCAP_MAX_SECONDS = float(os.environ.get("MOCAP_MAX_SECONDS", "60"))
MOCAP_MAX_DIM = int(os.environ.get("MOCAP_MAX_DIM", "1920"))
EXPR_ENABLED = os.environ.get("MOCAP_EXPR", "1") != "0"
ONE_EURO_MIN_CUTOFF = float(os.environ.get("MOCAP_ONE_EURO_MIN_CUTOFF", "1.5"))
ONE_EURO_BETA = float(os.environ.get("MOCAP_ONE_EURO_BETA", "0.3"))

MHR_LOD = int(os.environ.get("MHR_LOD", "3"))
MHR_NPZ = os.environ.get("MHR_NPZ", f"/models/mhr/mhr_lod{MHR_LOD}.npz")
MHR_NPZ_FALLBACK = f"/models/sapiens2/mhr/mhr_lod{MHR_LOD}.npz"
# Momentum works in centimeters; export in meters.
MHR_SCALE = float(os.environ.get("MHR_SCALE", "0.01"))

# SAM 3D Body's mhr_head flips vertices/joint-coords/cam_t into its camera
# convention with y,z *= -1, but joint_global_rots stay in MHR model space
# (y-up — the same space as the exported character). So rotations pass
# through unchanged and positions get un-flipped with this.
_FLIP = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------- quaternions
# Quaternions are (x, y, z, w), matching glTF.


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[:3]


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
                s / 4,
            ]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q / np.linalg.norm(q)


def _mats_to_quats(mats: np.ndarray) -> np.ndarray:
    out = np.zeros((len(mats), 4))
    for j, m in enumerate(mats):
        out[j] = _quat_from_matrix(m)
    return out


# ------------------------------------------------------------------ filters


def one_euro(x: np.ndarray, fps: float) -> np.ndarray:
    """One-Euro filter over axis 0 of (F, ...) data."""
    min_cutoff, beta, d_cutoff = ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, 1.0

    def alpha(cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)

    out = np.empty_like(x)
    out[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    a_d = alpha(d_cutoff)
    for i in range(1, len(x)):
        dx = (x[i] - out[i - 1]) * fps
        dx_prev = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_prev)
        a = 1.0 / (1.0 + (fps / (2 * math.pi * cutoff)))
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


# ---------------------------------------------------------------- extraction


def extract_frames(video_bytes: bytes) -> tuple[list[np.ndarray], float]:
    """Decode to BGR frames at MOCAP_FPS, capped at MOCAP_MAX_SECONDS."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("could not decode input video")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = min(MOCAP_FPS, src_fps)
        step = src_fps / fps
        max_frames = int(MOCAP_MAX_SECONDS * fps)
        frames: list[np.ndarray] = []
        next_pick = 0.0
        i = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= next_pick:
                h, w = frame.shape[:2]
                scale = MOCAP_MAX_DIM / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(
                        frame, (int(w * scale), int(h * scale)), cv2.INTER_AREA
                    )
                frames.append(frame)
                next_pick += step
            i += 1
        cap.release()
        if len(frames) < 2:
            raise RuntimeError("video too short for motion capture (need >= 2 frames)")
        return frames, fps
    finally:
        os.unlink(path)


# ------------------------------------------------------------- MHR character


@lru_cache(maxsize=1)
def load_character() -> dict:
    path = MHR_NPZ if os.path.exists(MHR_NPZ) else MHR_NPZ_FALLBACK
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["joint_names"]]
    return {
        "names": names,
        "by_name": {n: i for i, n in enumerate(names)},
        "parents": d["parents"].astype(int),
        "rest_pos": d["rest_positions"] * MHR_SCALE,
        "rest_rot": d["rest_rotations"],
        "vertices": d["vertices"] * MHR_SCALE,
        "faces": d["faces"],
        "skin_weights": d["skin_weights"],
        "skin_indices": d["skin_indices"],
        "face_shapes": (
            d["face_shape_vectors"] * MHR_SCALE
            if "face_shape_vectors" in d.files
            else None
        ),
    }


def _top4_skin(weights: np.ndarray, indices: np.ndarray):
    order = np.argsort(-weights, axis=1)[:, :4]
    w4 = np.take_along_axis(weights, order, axis=1)
    i4 = np.take_along_axis(indices, order, axis=1)
    s = w4.sum(axis=1, keepdims=True)
    w4 = np.where(s > 1e-8, w4 / np.maximum(s, 1e-8), np.array([1.0, 0, 0, 0]))
    return w4.astype(np.float32), i4.astype(np.uint16)


def _rest_local(char) -> tuple[np.ndarray, np.ndarray]:
    parents = char["parents"]
    rest_pos, rest_rot = char["rest_pos"], char["rest_rot"]
    n = len(parents)
    lp = np.zeros((n, 3), dtype=np.float32)
    lr = np.zeros((n, 4), dtype=np.float32)
    for j in range(n):
        p = parents[j]
        if p < 0:
            lp[j], lr[j] = rest_pos[j], rest_rot[j]
        else:
            inv = _quat_conj(rest_rot[p])
            lp[j] = _quat_rotate(inv, rest_pos[j] - rest_pos[p])
            lr[j] = _quat_mul(inv, rest_rot[j])
    return lp, lr


# ------------------------------------------------------- convention self-check


def _fk_positions(char, glob_rot: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
    parents = char["parents"]
    rest_pos = char["rest_pos"]
    pos = np.zeros_like(rest_pos)
    for j in range(len(parents)):
        p = parents[j]
        if p < 0:
            pos[j] = root_pos
        else:
            offset = rest_pos[j] - rest_pos[p]
            pos[j] = pos[p] + _quat_rotate(glob_rot[p], offset)
    return pos


def _resolve_convention(char, sample: dict) -> bool:
    """True if pred_global_rots are absolute orientations (bind included).

    Verified against the model's own pred_joint_coords: FK the skeleton under
    both interpretations and keep whichever reproduces the coordinates better.
    """
    coords = sample["joint_coords"] @ _FLIP.T  # un-flip back to model space
    root = coords[char["by_name"]["root"]]
    quats_abs = _mats_to_quats(sample["global_rots"])
    quats_delta = np.array(
        [_quat_mul(q, r) for q, r in zip(quats_abs, char["rest_rot"])]
    )

    def err(glob):
        fk = _fk_positions(char, glob, root)
        return float(np.linalg.norm(fk - coords, axis=1).mean())

    e_abs, e_delta = err(quats_abs), err(quats_delta)
    print(f"sam3dbody rotation convention: absolute={e_abs:.4f}m delta={e_delta:.4f}m")
    return e_abs <= e_delta


# ------------------------------------------------------------------- capture


def _pick_person(outputs: list[dict], prev_bbox: np.ndarray | None) -> dict:
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(union, 1e-9)

    if prev_bbox is not None:
        return max(outputs, key=lambda o: iou(prev_bbox, o["bbox"]))
    return max(
        outputs,
        key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]),
    )


def capture(estimator, video_bytes: bytes, bbox_thr: float, progress=None) -> bytes:
    """Full pipeline: video bytes -> animated MHR GLB bytes."""

    def report(msg: str) -> None:
        print(msg)
        if progress is not None:
            progress(msg)

    char = load_character()
    by_name = char["by_name"]
    parents = char["parents"]
    rest_rot = char["rest_rot"]
    rest_pos = char["rest_pos"]
    n_joints = len(char["names"])

    report("mocap: extracting frames")
    frames, fps = extract_frames(video_bytes)
    n_frames = len(frames)

    report(f"mocap: SAM 3D Body across {n_frames} frames")
    quats = np.zeros((n_frames, n_joints, 4))
    cam_t = np.zeros((n_frames, 3))
    expr = np.zeros((n_frames, 72))
    valid = np.zeros(n_frames, dtype=bool)
    convention_absolute: bool | None = None
    prev_bbox = None
    for f, frame in enumerate(frames):
        if f % 24 == 0:
            report(f"mocap: body {f}/{n_frames}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = estimator.process_one_image(rgb, bbox_thr=bbox_thr)
        if not outputs:
            continue
        person = _pick_person(outputs, prev_bbox)
        prev_bbox = np.asarray(person["bbox"], dtype=np.float32)
        rots = np.asarray(person["pred_global_rots"], dtype=np.float64)
        if rots.shape[0] != n_joints:
            raise RuntimeError(
                f"SAM 3D Body joint count {rots.shape[0]} != MHR bundle {n_joints}"
            )
        if convention_absolute is None:
            convention_absolute = _resolve_convention(
                char,
                {
                    "global_rots": rots,
                    "joint_coords": np.asarray(
                        person["pred_joint_coords"], dtype=np.float64
                    ),
                },
            )
        q = _mats_to_quats(rots)
        if not convention_absolute:
            q = np.array([_quat_mul(qq, r) for qq, r in zip(q, rest_rot)])
        quats[f] = q
        cam_t[f] = _FLIP @ np.asarray(person["pred_cam_t"], dtype=np.float64)
        e = np.asarray(person["expr_params"], dtype=np.float64).reshape(-1)
        expr[f, : min(72, len(e))] = e[:72]
        valid[f] = True

    if not valid.any():
        raise RuntimeError("no person detected in the video")

    # Hold the last valid state through detection gaps, then smooth.
    last = None
    for f in range(n_frames):
        if valid[f]:
            last = f
        elif last is not None:
            quats[f], cam_t[f], expr[f] = quats[last], cam_t[last], expr[last]
    first = int(np.argmax(valid))
    quats[:first], cam_t[:first], expr[:first] = (
        quats[first],
        cam_t[first],
        expr[first],
    )

    for f in range(1, n_frames):
        dots = (quats[f] * quats[f - 1]).sum(axis=1)
        quats[f, dots < 0] *= -1
    quats = one_euro(quats, fps)
    quats /= np.linalg.norm(quats, axis=2, keepdims=True).clip(min=1e-9)
    cam_t = one_euro(cam_t, fps)

    # Expression diagnostics: the upstream head's face branch is unlocked at
    # image build; whether the released weights actually trained it shows up
    # here (smooth, ~[0,1], few active dims = trained; wideband noise = not).
    raw_std = float(expr[valid].std(axis=0).mean())
    active = int((expr.std(axis=0) > 0.02).sum())
    report(
        f"mocap: expr stats std={raw_std:.4f} range=[{expr.min():.3f},{expr.max():.3f}] "
        f"active_dims={active}/72"
    )
    expr = np.clip(one_euro(expr, fps), 0.0, 1.5)

    report("mocap: exporting GLB")
    glob = quats
    local_anim: dict[int, np.ndarray] = {}
    for j in range(n_joints):
        p = parents[j]
        local = np.zeros((n_frames, 4))
        for f in range(n_frames):
            local[f] = (
                glob[f, j] if p < 0 else _quat_mul(_quat_conj(glob[f, p]), glob[f, j])
            )
        for f in range(1, n_frames):
            if np.dot(local[f], local[f - 1]) < 0:
                local[f] = -local[f]
        local_anim[j] = local.astype(np.float32)

    root = by_name["root"]
    root_parent = parents[root]
    motion = cam_t - cam_t[0]
    root_trans = np.zeros((n_frames, 3), dtype=np.float32)
    for f in range(n_frames):
        base = rest_pos[root] + motion[f]
        if root_parent >= 0:
            base = _quat_rotate(
                _quat_conj(rest_rot[root_parent]), base - rest_pos[root_parent]
            )
        root_trans[f] = base

    w4, i4 = _top4_skin(char["skin_weights"], char["skin_indices"])
    lp, lr = _rest_local(char)
    times = np.arange(n_frames, dtype=np.float32) / fps

    morph = None
    weights = None
    if EXPR_ENABLED and char.get("face_shapes") is not None:
        morph = char["face_shapes"]
        weights = expr.astype(np.float32)

    return skinned_character_glb(
        vertices=char["vertices"].astype(np.float32),
        faces=char["faces"].astype(np.uint32),
        joints_weights=w4,
        joints_indices=i4,
        joint_names=char["names"],
        parents=parents,
        rest_local_pos=lp,
        rest_local_rot=lr,
        rest_global_pos=rest_pos.astype(np.float32),
        rest_global_rot=rest_rot.astype(np.float32),
        rotation_channels=local_anim,
        translation_channels={root: root_trans},
        times=times,
        morph_targets=morph,
        morph_weights=weights,
    )
