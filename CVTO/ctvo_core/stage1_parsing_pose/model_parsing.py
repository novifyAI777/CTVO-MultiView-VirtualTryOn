"""
Stage 1: Human Parsing Model

This module handles human parsing using ONNX models (ATR/LIP)
"""

import numpy as np
from PIL import Image
import torchvision.transforms as T
import onnxruntime as ort
from typing import Optional


class HumanParsingModel:
    """Human parsing model using ONNX runtime"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.session = ort.InferenceSession(model_path)
        self.transform = T.Compose([
            T.Resize((473, 473)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def parse_human(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """Run human parsing on input image"""
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).numpy()
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})
        seg = np.argmax(output[0], axis=1)[0].astype(np.uint8)
        
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(seg).save(output_path)
            print(f"Parsing map saved at: {output_path}")
        
        return seg
