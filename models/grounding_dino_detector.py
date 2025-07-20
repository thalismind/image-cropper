"""
Grounding DINO detector for text-guided object detection.
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
import supervision as sv
from groundingdino.util.inference import Model as GroundingDINO
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import os

class GroundingDINODetector:
    """Grounding DINO detector for text-guided object detection."""

    def __init__(self, model_config_path: str = "models/groundingdino/groundingdino_swinb_cogcoor.py",
                 model_checkpoint_path: str = "models/groundingdino/groundingdino_swinb_cogcoor.pth",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Grounding DINO detector.

        Args:
            model_config_path: Path to model configuration file
            model_checkpoint_path: Path to model checkpoint file
            device: Device to run inference on
        """
        self.device = device
        self.model = self._load_model(model_config_path, model_checkpoint_path)

    def _load_model(self, config_path: str, checkpoint_path: str) -> GroundingDINO:
        """Load Grounding DINO model."""
        try:
            # Load configuration
            args = SLConfig.fromfile(config_path)
            args.device = self.device

            # Build model
            model = build_model(args)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            model.eval()
            model.to(self.device)

            print(f"Grounding DINO model loaded successfully on {self.device}")
            return model

        except Exception as e:
            print(f"Error loading Grounding DINO model: {e}")
            print("Please ensure model files are downloaded correctly.")
            raise

    def detect(self, image: np.ndarray, text_prompt: str,
               confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Detect objects in image based on text prompt.

        Args:
            image: Input image as numpy array (BGR format)
            text_prompt: Text description of objects to detect
            confidence_threshold: Minimum confidence for detections

        Returns:
            Tuple of (detections, annotated_image, detection_info)
        """
        try:
                        # Convert BGR to RGB and use load_image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save temporary image and load with load_image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image_rgb)
                from groundingdino.util.inference import load_image, predict
                image_source, image_tensor = load_image(tmp_file.name)
                import os
                os.unlink(tmp_file.name)

            # Run inference using the correct API
            detections = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=confidence_threshold,
                text_threshold=confidence_threshold,
                device=self.device
            )

            # Convert detections to supervision format
            if detections is not None and len(detections[0]) > 0:
                boxes, logits, phrases = detections

                # Convert boxes to xyxy format
                xyxy = boxes.cpu().numpy()
                confidence = logits.cpu().numpy()

                # Create supervision detections
                supervision_detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=np.zeros(len(xyxy), dtype=int)
                )

                # Create annotated image
                annotated_image = image.copy()
                box_annotator = sv.BoxAnnotator()
                annotated_image = box_annotator.annotate(
                    scene=annotated_image,
                    detections=supervision_detections,
                    labels=[f"{text_prompt} {conf:.2f}" for conf in supervision_detections.confidence]
                )

                # Prepare detection info
                detection_info = []
                for i, (xyxy, conf) in enumerate(zip(supervision_detections.xyxy, supervision_detections.confidence)):
                    detection_info.append({
                        'bbox': xyxy.tolist(),
                        'confidence': float(conf),
                        'class_name': text_prompt
                    })

                return supervision_detections, annotated_image, detection_info
            else:
                # No detections found
                return sv.Detections.empty(), image.copy(), []

        except Exception as e:
            print(f"Error during detection: {e}")
            return sv.Detections.empty(), image.copy(), []

    def detect_multiple_classes(self, image: np.ndarray,
                              include_classes: List[str],
                              exclude_classes: List[str],
                              confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect multiple classes in an image.

        Args:
            image: Input image as numpy array
            include_classes: List of classes to include
            exclude_classes: List of classes to exclude
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dictionary with detection results for each class
        """
        results = {
            'include': {},
            'exclude': {},
            'all_detections': sv.Detections.empty()
        }

        # Detect include classes
        for class_name in include_classes:
            detections, annotated_img, detection_info = self.detect(
                image, class_name, confidence_threshold
            )
            results['include'][class_name] = {
                'detections': detections,
                'detection_info': detection_info
            }

        # Detect exclude classes
        for class_name in exclude_classes:
            detections, annotated_img, detection_info = self.detect(
                image, class_name, confidence_threshold
            )
            results['exclude'][class_name] = {
                'detections': detections,
                'detection_info': detection_info
            }

        # Combine all detections
        all_xyxy = []
        all_confidence = []
        all_class_ids = []

        # Add include detections
        for class_name, data in results['include'].items():
            if len(data['detections']) > 0:
                all_xyxy.extend(data['detections'].xyxy)
                all_confidence.extend(data['detections'].confidence)
                all_class_ids.extend([1] * len(data['detections']))  # 1 for include

        # Add exclude detections
        for class_name, data in results['exclude'].items():
            if len(data['detections']) > 0:
                all_xyxy.extend(data['detections'].xyxy)
                all_confidence.extend(data['detections'].confidence)
                all_class_ids.extend([0] * len(data['detections']))  # 0 for exclude

        if all_xyxy:
            results['all_detections'] = sv.Detections(
                xyxy=np.array(all_xyxy),
                confidence=np.array(all_confidence),
                class_id=np.array(all_class_ids)
            )

        return results