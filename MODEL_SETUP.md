# Model Setup

The cyclone detection app requires the DeepLabV3+ model weights file.

## Required File
- `deeplab_mask3_best.pth` - Place this file in the project root directory

## How to add the model:
1. Train your model locally or download pre-trained weights
2. Save as `deeplab_mask3_best.pth` in the project root
3. Add to `.gitignore` if file is large (>100MB)
4. For Railway deployment, either:
   - Commit the file to git (if < 100MB)
   - Use Railway volumes to store the model file
   - Download from external storage during deployment

## Troubleshooting
If you see "Model file not found" error:
1. Verify `deeplab_mask3_best.pth` exists in project root
2. Check file permissions
3. Ensure model was trained with same architecture (DeepLabV3Plus, resnet34, 4 classes)
