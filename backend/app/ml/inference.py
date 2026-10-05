from pathlib import Path
from PIL import Image
import io
import json
import zipfile

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

# Models are loaded only when an image inference endpoint is actually used.
# This keeps Render Free startup memory low while preserving every model file.
_BEE = None
_EFF = None
_YOLO = None
_BEE_DETECTOR = None
_BEE_CLASSES = None
_TRANSFORMS = None
_DEVICE = None

def _torch_compatible_pt(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = [n for n in archive.namelist() if n and not n.endswith("/")]
            top_levels = {n.split("/", 1)[0] for n in names}
            legacy_layout = "version" in names and len(top_levels) > 1
            if not legacy_layout:
                return path
            fixed = path.with_name(path.stem + "_torch_compatible.pt")
            if not fixed.exists() or fixed.stat().st_mtime < path.stat().st_mtime:
                temp = fixed.with_suffix(".tmp.pt")
                root = path.stem
                with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as out:
                    for name in archive.namelist():
                        out.writestr(f"{root}/{name}", archive.read(name))
                temp.replace(fixed)
            return fixed
    except zipfile.BadZipFile:
        return path

def _load_models():
    global _BEE, _EFF, _YOLO, _BEE_DETECTOR, _BEE_CLASSES, _TRANSFORMS, _DEVICE
    if _BEE is not None:
        return

    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from ultralytics import YOLO

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    _DEVICE = torch.device("cpu")

    class BeeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 5, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(5, 10, 3, padding=1)
            self.fc = nn.Linear(25000, 6)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            return self.fc(torch.flatten(x, 1))

    bee = BeeCNN()
    bee_ckpt = torch.load(MODEL_DIR / "bee_health_cnn_best.pth", map_location=_DEVICE, weights_only=False)
    bee.load_state_dict(bee_ckpt["state_dict"])
    bee.to(_DEVICE).eval()

    eff = models.efficientnet_b0(weights=None)
    eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, 2)
    eff_ckpt = torch.load(MODEL_DIR / "efficientnet_b0_varroa_best.pth", map_location=_DEVICE, weights_only=False)
    eff.load_state_dict(eff_ckpt["model_state_dict"])
    eff.to(_DEVICE).eval()

    yolo = YOLO(str(MODEL_DIR / "varroa_yolo11n_best.pt"))
    bee_detector = None
    bee_detector_path = MODEL_DIR / "bee_final.pt"
    if bee_detector_path.exists():
        bee_detector = YOLO(str(_torch_compatible_pt(bee_detector_path)))

    _BEE = bee
    _EFF = eff
    _YOLO = yolo
    _BEE_DETECTOR = bee_detector
    _BEE_CLASSES = bee_ckpt.get("class_mapping", {
        "0":"healthy","1":"few varroa / hive beetles","2":"varroa / small hive beetles",
        "3":"ant problems","4":"hive being robbed","5":"missing queen"
    })
    _TRANSFORMS = (
        transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    )

def _image(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.open(io.BytesIO(x)).convert("RGB")

def predict_image(raw):
    _load_models()
    import torch
    bee, eff, yolo = _BEE, _EFF, _YOLO
    bee_detector = _BEE_DETECTOR
    bee_tf, eff_tf = _TRANSFORMS
    image = _image(raw)
    with torch.no_grad():
        bp = torch.softmax(bee(bee_tf(image).unsqueeze(0).to(_DEVICE)), 1)[0].cpu().tolist()
        ep = torch.softmax(eff(eff_tf(image).unsqueeze(0).to(_DEVICE)), 1)[0].cpu().tolist()

    bee_idx = int(max(range(len(bp)), key=lambda i: bp[i]))
    bee_infestation = float(bp[1] + bp[2])
    eff_infestation = float(ep[1])

    bee_detection_result = bee_detector.predict(source=image, conf=0.20, verbose=False)[0] if bee_detector else None
    bee_detections = []
    if bee_detection_result is not None:
        for b in bee_detection_result.boxes:
            cls_id = int(b.cls[0].cpu()) if b.cls is not None else 0
            bee_detections.append({
                "confidence": float(b.conf[0].cpu()),
                "class": (bee_detector.names.get(cls_id, str(cls_id)) if hasattr(bee_detector, "names") else str(cls_id))
            })

    result = yolo.predict(source=image, conf=0.25, verbose=False)[0]
    boxes = []
    for b in result.boxes:
        xyxy = b.xyxy[0].cpu().tolist()
        boxes.append({"x1":xyxy[0],"y1":xyxy[1],"x2":xyxy[2],"y2":xyxy[3],"confidence":float(b.conf[0].cpu()),"class":"varroa_mite"})
    yolo_score = max((x["confidence"] for x in boxes), default=0.0)

    visual_score = 0.35 * bee_infestation + 0.50 * eff_infestation + 0.15 * yolo_score
    if visual_score >= 0.75: band = "High Concern"
    elif visual_score >= 0.50: band = "Monitor"
    else: band = "Low Concern"

    return {
        "visualScore": round(visual_score, 4),
        "visualStatus": band,
        "beeHealth": {
            "class": _BEE_CLASSES.get(str(bee_idx), str(bee_idx)),
            "confidence": round(float(bp[bee_idx]), 4),
            "infestationEvidence": round(bee_infestation, 4),
            "probabilities": {_BEE_CLASSES.get(str(i), str(i)): round(float(p),4) for i,p in enumerate(bp)}
        },
        "beeDetector": {
            "available": bee_detector is not None,
            "detections": bee_detections,
            "beeCount": sum(1 for x in bee_detections if x["class"] == "bee"),
            "blurredBeeCount": sum(1 for x in bee_detections if x["class"] == "blurred_bee")
        },
        "varroa": {
            "infestedProbability": round(eff_infestation, 4),
            "miteCount": len(boxes),
            "maxDetectionConfidence": round(yolo_score, 4),
            "boxes": boxes
        },
        "fusion": {
            "beeImageWeight": 0.35, "verrdoEfficientNetWeight": 0.50, "verrdoYoloWeight": 0.15,
            "thresholds": {"monitor":0.50,"high":0.75}
        },
        "modelVersion": "HoneyChain visual fusion v1.1 + bee detector",
        "device": str(_DEVICE),
        "imageWidth": image.width, "imageHeight": image.height,
    }
