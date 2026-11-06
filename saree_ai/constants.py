"""Single source of truth for label spaces shared by training, inference and the frontend.

Changing any list here is a breaking change for trained models: bump SCHEMA_VERSION
and retrain/re-export every affected model.
"""

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------- body landmarks
# COCO-17 ordering. Chosen because (a) the pretrained baseline (torchvision Keypoint R-CNN)
# emits exactly this layout and (b) COCO-format annotation tools (CVAT, Label Studio) export it.
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
# Left/right swap pairs used for horizontal-flip augmentation.
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# ---------------------------------------------------------------- saree segmentation
# Index == pixel value in mask PNGs. 255 == ignore (unlabelled / ambiguous).
SEG_CLASSES = [
    "background",      # 0 everything that is not the user
    "person",          # 1 skin, hair, face, hands not covered by fabric
    "other_clothing",  # 2 blouse, petticoat, other garments (NOT saree)
    "saree_body",      # 3 saree fabric not covered by a more specific class
    "waist_wrap",      # 4 the band wrapped around the waist
    "pleats",          # 5 front pleat stack (tucked or in hand)
    "pallu",           # 6 the loose end over / towards the shoulder
]
SEG = {name: i for i, name in enumerate(SEG_CLASSES)}
SEG_IGNORE = 255
SAREE_CLASS_IDS = [SEG["saree_body"], SEG["waist_wrap"], SEG["pleats"], SEG["pallu"]]

# ---------------------------------------------------------------- pleats
PLEAT_QUALITY = ["neat", "uneven_spacing", "misaligned", "loose_or_unformed"]
PLEAT_MAX_COUNT = 12

# ---------------------------------------------------------------- draping stages (Nivi)
DRAPE_STAGES = [
    "not_started",
    "wrapping_started",
    "waist_wrap_complete",
    "pleats_forming",
    "pleats_secured",
    "pallu_in_progress",
    "pallu_complete",
]
STAGE = {name: i for i, name in enumerate(DRAPE_STAGES)}

# ---------------------------------------------------------------- correction labels
# Annotators tag frames with zero or more of these. The correction engine emits the same ids,
# which is what makes correction precision / false-positive rate measurable.
CORRECTION_LABELS = [
    "hem_uneven",
    "waist_wrap_loose",
    "pleats_uneven_spacing",
    "pleats_misaligned",
    "pleats_off_center",
    "pleats_too_few",
    "pleats_too_many",
    "pleats_not_tucked",
    "pallu_wrong_shoulder",
    "pallu_not_on_shoulder",
    "pallu_too_short",
]

SOURCE_TYPES = ["real", "synthetic"]
