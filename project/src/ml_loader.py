from sentence_transformers import CrossEncoder
from .logger_config import setup_logger

logger = setup_logger(__name__)


def load_cross_encoder(model_path: str, device: str = "cpu") -> CrossEncoder:
    """Load Cross-Encoder Model with training params"""

    logger.info(f"Loading model from {model_path} on device {device}")
    model = CrossEncoder(model_path, device=device)
    logger.info("Model was loaded success")
    return model
