import pytest
import numpy as np 
from PIL import Image
from backend.predict import process
import torch 
import io


def make_dummy_image_bytes(width=300, height=300):
    """Creates a fake RGB image and returns it as raw bytes (like a real upload)."""
    img = Image.new("RGB", (width, height), color=(120, 180, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_output_shape():
    """TC_UNIT_01: output tensor must be shape (1, 3, 224, 224)"""
    image_bytes = make_dummy_image_bytes(300, 300)
    tensor = process(image_bytes)
    assert tensor.shape == (1, 3, 224, 224), f"Got shape {tensor.shape}"

def test_output_is_tensor():
    """Output must be a PyTorch tensor, not a numpy array or PIL image"""
    image_bytes = make_dummy_image_bytes(300, 300)
    tensor = process(image_bytes)
    assert isinstance(tensor, torch.Tensor)

def test_small_image_still_resizes():
    """A small 64x64 image must still produce (1, 3, 224, 224)"""
    image_bytes = make_dummy_image_bytes(64, 64)
    tensor = process(image_bytes)
    assert tensor.shape == (1, 3, 224, 224)

def test_large_image_still_resizes():
    """A large 1024x1024 image must still produce (1, 3, 224, 224)"""
    image_bytes = make_dummy_image_bytes(1024, 1024)
    tensor = process(image_bytes)
    assert tensor.shape == (1, 3, 224, 224)

def test_output_dtype():
    """Output tensor must be float32 (required by PyTorch models)"""
    image_bytes = make_dummy_image_bytes(300, 300)
    tensor = process(image_bytes)
    assert tensor.dtype == torch.float32