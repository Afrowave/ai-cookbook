import logging
import os

from typing import List, Dict
from pydantic import BaseModel, Field
from ollama import chat


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ai_model = 'qwen2.5-coder:14b'
