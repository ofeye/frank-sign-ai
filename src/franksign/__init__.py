"""Frank Sign AI - Medical image analysis for cardiovascular risk prediction.

This package provides tools for:
- Parsing CVAT annotations for ear images
- Extracting geometric features from Frank Sign
- Training segmentation models (U-Net, Attention U-Net)
- Explainable AI visualizations (Grad-CAM, SHAP)

Example:
    >>> from franksign.data import CVATParser
    >>> parser = CVATParser("annotations.xml")
    >>> annotations = parser.parse()
"""

import logging

# Configure package-wide logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create package logger
logger = logging.getLogger('franksign')

__version__ = "0.1.0"
__author__ = "Frank Sign Research Team"

from franksign.data.cvat_parser import CVATParser
from franksign.data.geometric_features import GeometricFeatureExtractor

__all__ = [
    "CVATParser",
    "GeometricFeatureExtractor",
    "logger",
    "__version__",
]

