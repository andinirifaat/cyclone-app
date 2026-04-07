# inference_validd.py

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# ====================================
# CONFIG
# ====================================
MODEL_PATH = "deeplab_mask3_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_CORE_AREA = 300
MIN_IMPACT_AREA = 5000


# ====================================
# LOAD MODEL
# ====================================
def load_model():
    import os

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please ensure deeplab_mask3_best.pth is in the project root directory."
        )

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4,
    )

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {str(e)}")

    model.to(DEVICE)
    model.eval()

    return model


# ====================================
# MAIN INFERENCE FUNCTION
# ====================================
def run_inference(model, image_np):
    """
    image_np: numpy array (H, W, 3) RGB
    return:
        color_mask: (H, W, 3)
        boxes: list of dict {class, box}
        overlay_image: image with bbox
    """

    # =========================
    # PREPROCESS
    # =========================
    orig_rgb = image_np.copy()
    h, w, _ = orig_rgb.shape

    img_input = cv2.resize(orig_rgb, (512, 512))
    tensor = torch.tensor(img_input / 255.0).permute(2,0,1).float().unsqueeze(0).to(DEVICE)

    # =========================
    # PREDICTION
    # =========================
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    pred_mask = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # =========================
    # COLOR MASK
    # =========================
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    color_mask[pred_mask == 1] = [255, 0, 0]     # Red Core
    color_mask[pred_mask == 2] = [255, 0, 255]   # Impact
    color_mask[pred_mask == 3] = [0, 0, 255]     # DCC

    # =========================
    # BBOX EXTRACTION
    # =========================
    boxes = []

    # ---- CLASS 1: RED CORE ----
    binary_core = (pred_mask == 1).astype(np.uint8)
    contours, _ = cv2.findContours(binary_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CORE_AREA:
            continue

        x,y,wc,hc = cv2.boundingRect(cnt)
        boxes.append({"class": "core", "box": [x,y,wc,hc]})

    # ---- CLASS 2: IMPACT ----
    binary_imp = (pred_mask == 2).astype(np.uint8)
    contours, _ = cv2.findContours(binary_imp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    impact_boxes = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_IMPACT_AREA:
            continue

        hull = cv2.convexHull(cnt)
        x,y,wc,hc = cv2.boundingRect(hull)
        impact_boxes.append((x,y,wc,hc))

    # remove nested
    final_boxes = []

    for i, box1 in enumerate(impact_boxes):
        x1,y1,w1,h1 = box1
        nested = False

        for j, box2 in enumerate(impact_boxes):
            if i == j:
                continue

            x2,y2,w2,h2 = box2

            if (x1 >= x2 and y1 >= y2 and
                x1+w1 <= x2+w2 and
                y1+h1 <= y2+h2):
                nested = True
                break

        if not nested:
            final_boxes.append(box1)

    for (x,y,wc,hc) in final_boxes:
        boxes.append({"class": "impact", "box": [x,y,wc,hc]})

    # ---- CLASS 3: DCC ----
    binary_dcc = (pred_mask == 3).astype(np.uint8)
    contours, _ = cv2.findContours(binary_dcc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        boxes.append({"class": "dcc", "box": [x,y,wc,hc]})

    # =========================
    # DRAW OVERLAY
    # =========================
    overlay = orig_rgb.copy()

    for b in boxes:
        x,y,wc,hc = b["box"]

        if b["class"] == "core":
            color = (255,0,0)
        elif b["class"] == "impact":
            color = (255,0,255)
        else:
            color = (0,0,255)

        cv2.rectangle(overlay, (x,y), (x+wc, y+hc), color, 2)

    # =========================
    return color_mask, boxes, overlay