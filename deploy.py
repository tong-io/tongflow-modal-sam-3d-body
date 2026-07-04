"""Modal deploy entry for SAM 3D Body (Meta, single-image human mesh recovery).

Implements the ``image-gen-model`` slot: one input image -> one GLB containing
the recovered 3D mesh of every detected person (MHR parametric body, vertex
colors, people placed by their predicted camera translation).

Pipeline per call: ViTDet-H detects people, MoGe-2 estimates the camera FOV,
SAM 3D Body regresses the full-body MHR mesh per person; the meshes are merged
and exported as one GLB.

The ``facebook/sam-3d-body-*`` checkpoints are gated on Hugging Face: accept
the terms on the repo page, then put that account's ``HF_TOKEN`` in TongFlow
Settings before first use.

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.image_gen_model import ImageGenModelInput, ImageGenModelOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

REPO_URL = "https://github.com/facebookresearch/sam-3d-body.git"
# Pin the upstream revision so redeploys are reproducible (main moves).
REPO_REV = "b5c765a0d89d789985e186d396315e7590887b94"
REPO_DIR = "/app/sam-3d-body"

# Plugin-internal knobs — NOT ABI fields.
MODEL_ID = os.environ.get("SAM_3D_BODY_MODEL", "facebook/sam-3d-body-dinov3")
BBOX_THRESHOLD = float(os.environ.get("SAM_3D_BODY_BBOX_THRESHOLD", 0.5))
MESH_RGBA = (166, 189, 219, 255)

volume = modal.Volume.from_name("models", create_if_missing=True)
# The checkpoint repo is gated: forward the local HF_TOKEN (TongFlow Settings)
# into the container so snapshot downloads authenticate.
secrets = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    # devel base: detectron2 compiles its CUDA ops from source at build time.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
    )
    # clang: Modal's add_python is clang-built, so distutils links extension
    # modules (detectron2's _C) with `clang++`.
    .apt_install("git", "clang", "libgl1", "libglib2.0-0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6;8.9;9.0"})
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # INSTALL.md's dependency line, minus train/dev-only extras
    # (wandb, black, pytest, tensorboard, submitit, ...).
    .pip_install(
        "pytorch-lightning",
        "pyrender",
        "opencv-python-headless",
        "yacs",
        "scikit-image",
        "einops",
        "timm",
        "dill",
        "pandas",
        "rich",
        "hydra-core",
        "hydra-colorlog",
        "pyrootutils",
        "webdataset",
        "networkx==3.2.1",
        "roma",
        "joblib",
        "xtcocotools",
        "loguru",
        "optree",
        "fvcore",
        "pycocotools",
        "huggingface_hub",
        "trimesh",
    )
    # wheel/ninja must predate detectron2: --no-build-isolation reuses the
    # ambient env, and setuptools errors with "invalid command 'bdist_wheel'"
    # without them.
    .pip_install("wheel", "setuptools", "ninja")
    .pip_install(
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        extra_options="--no-build-isolation --no-deps",
    )
    .pip_install("git+https://github.com/microsoft/MoGe.git")
    .pip_install("tongflow==0.2.3")
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout {REPO_REV}",
    )
    # The repo is not a package: import sam_3d_body/tools straight from the
    # clone. Caches (HF snapshot, ViTDet pkl, MoGe) live on the volume.
    .env(
        {
            "PYTHONPATH": REPO_DIR,
            "HF_HOME": "/models/hf",
            "FVCORE_CACHE": "/models/fvcore",
        }
    )
)

with image.imports():
    import io

    import cv2
    import numpy as np
    import trimesh


@deploy
@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
    secrets=[secrets],
    timeout=1800,
    scaledown_window=5,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Load the estimator stack once; reused across calls (warm)."""
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
        from tools.build_detector import HumanDetector
        from tools.build_fov_estimator import FOVEstimator

        model, model_cfg = load_sam_3d_body_hf(MODEL_ID, device="cuda")
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=HumanDetector(name="vitdet", device="cuda"),
            human_segmentor=None,
            fov_estimator=FOVEstimator(name="moge2", device="cuda"),
        )
        # Persist whatever the first boot downloaded (HF snapshots, ViTDet
        # pkl) so later cold starts are download-free.
        volume.commit()

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_MODEL)
    def image_gen_model(self, input: ImageGenModelInput) -> ImageGenModelOutput:
        """One image -> one GLB with every detected person's 3D body mesh.

        text/width/height/seed are part of the image-gen-model contract but
        mesh recovery is deterministic and image-only, so they are ignored.
        """
        try:
            raw = prompt_media_to_bytes(input.image)
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("could not decode input image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            outputs = self.estimator.process_one_image(
                rgb, bbox_thr=BBOX_THRESHOLD
            )
            if not outputs:
                raise RuntimeError("no person detected in the image")

            faces = self.estimator.faces
            meshes = []
            for person in outputs:
                verts = np.asarray(person["pred_vertices"], dtype=np.float64)
                cam_t = np.asarray(person["pred_cam_t"], dtype=np.float64)
                mesh = trimesh.Trimesh(
                    vertices=verts + cam_t,
                    faces=faces.copy(),
                    vertex_colors=np.tile(MESH_RGBA, (verts.shape[0], 1)),
                    process=False,
                )
                meshes.append(mesh)
            merged = trimesh.util.concatenate(meshes)
            # Camera coords are y-down/z-forward; glTF viewers expect y-up.
            merged.apply_transform(np.diag([1.0, -1.0, -1.0, 1.0]))

            buf = io.BytesIO()
            merged.export(buf, file_type="glb")
            data = buf.getvalue()
        except Exception as e:
            return ImageGenModelOutput(success=False, error=str(e))
        return ImageGenModelOutput(
            success=True,
            model=asset(data, mime="model/gltf-binary", filename="bodies.glb"),
        )
