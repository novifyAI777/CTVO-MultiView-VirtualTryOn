import torch

ckpt_path = 'ctvo_core/stage2_cloth_warping/pretrained_weights/unet_wrap.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"Checkpoint type: {type(ckpt)}")
print()

if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print(f"Total keys: {len(keys)}")
    print("\nFirst 30 keys and shapes:")
    for i, k in enumerate(keys[:30]):
        if hasattr(ckpt[k], 'shape'):
            print(f"  {k}: {ckpt[k].shape}")
        else:
            print(f"  {k}: {type(ckpt[k])}")
    
    # Look for extractionA and extractionB structure
    print("\nExtractionA keys:")
    extA_keys = [k for k in keys if 'extractionA' in k]
    for k in extA_keys[:10]:
        if hasattr(ckpt[k], 'shape'):
            print(f"  {k}: {ckpt[k].shape}")
    
    print("\nExtractionB keys:")
    extB_keys = [k for k in keys if 'extractionB' in k]
    for k in extB_keys[:10]:
        if hasattr(ckpt[k], 'shape'):
            print(f"  {k}: {ckpt[k].shape}")
    
    print("\nRegression keys:")
    reg_keys = [k for k in keys if 'regression' in k]
    for k in reg_keys[:15]:
        if hasattr(ckpt[k], 'shape'):
            print(f"  {k}: {ckpt[k].shape}")
else:
    print(f"Checkpoint is not a dict: {ckpt}")
