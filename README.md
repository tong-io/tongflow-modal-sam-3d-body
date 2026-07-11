# tongflow-modal-sam-3d-body

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Single-image full-body human mesh recovery with **SAM 3D Body** (Meta, `facebook/sam-3d-body-dinov3`), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Image → 3D** (`image-gen-model`) — recover the full-body 3D mesh (body, hands, feet; MHR parametric rig) of every person in a photo. People are detected automatically (ViTDet-H) and placed in a shared scene by their predicted camera translation; the output is one GLB.
- **Video motion capture** (`video-gen-model`) — monocular video → animated **MHR character** GLB: per-frame full inference (body + hand decoders), One-Euro temporal smoothing, single subject (largest person, IoU-tracked). The upstream head's face-expression and jaw branches are experimentally unlocked and exported as glTF morph-target/joint animation — check the `mocap: expr stats` task log and set `MOCAP_EXPR=0` if they turn out untrained. Requires the MHR character bundle on the shared volume (`modal run extract_mhr.py::extract` once). Plays in the TongFlow model node; imports as a skinned armature in Blender.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | ✅ | The checkpoint is gated: request access on [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3), then use that account's token. |

On first use the plugin deploys to your Modal account automatically and caches the build; weights are cached on a shared Modal volume.

## Tuning (env, optional)

| Env | Default | Notes |
| --- | --- | --- |
| `SAM_3D_BODY_MODEL` | `facebook/sam-3d-body-dinov3` | Or `facebook/sam-3d-body-vith`. |
| `SAM_3D_BODY_BBOX_THRESHOLD` | `0.5` | Person-detection confidence threshold. |
