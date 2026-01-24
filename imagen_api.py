"""
Google AI Studio (Gemini API) Integration Module
Handles image generation using Google AI Studio's Imagen API (formerly Vertex AI Imagen).
"""

import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time


def validate_imagen_credentials(api_key: str) -> bool:
    """
    Validate Google AI Studio API key by checking format.

    Args:
        api_key: Google AI Studio API key

    Returns:
        True if valid format, False otherwise
    """
    try:
        # Google AI Studio keys typically start with "AIza"
        if not api_key or len(api_key) < 20:
            return False
        return True
    except Exception:
        return False


def generate_image_with_imagen(
    prompt: str,
    api_key: str,
    number_of_images: int = 1,
    aspect_ratio: str = "1:1",
    safety_filter_level: str = "block_some",
    person_generation: str = "allow_adult"
) -> Dict[str, Any]:
    """
    Generate images using Google AI Studio Imagen API.

    Args:
        prompt: The text prompt for image generation
        api_key: Google AI Studio API key
        number_of_images: Number of images to generate (1-4)
        aspect_ratio: Image aspect ratio ("1:1", "9:16", "16:9", "4:3", "3:4")
        safety_filter_level: Safety filter level ("block_most", "block_some", "block_few", "block_fewest")
        person_generation: Person generation policy ("allow_adult", "allow_all", "dont_allow")

    Returns:
        Dictionary containing generated images (as base64) and metadata
    """
    import requests

    print(f"[DEBUG] Starting image generation for prompt: {prompt[:50]}...")
    print(f"[DEBUG] Aspect ratio: {aspect_ratio}")

    try:
        # Google AI Studio Imagen endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={api_key}"
        print(f"[DEBUG] API endpoint: {url.split('?')[0]}")

        # Map aspect ratio to dimensions
        aspect_ratio_map = {
            "1:1": {"width": 1024, "height": 1024},
            "9:16": {"width": 768, "height": 1344},
            "16:9": {"width": 1344, "height": 768},
            "4:3": {"width": 1152, "height": 896},
            "3:4": {"width": 896, "height": 1152}
        }

        dimensions = aspect_ratio_map.get(aspect_ratio, {"width": 1024, "height": 1024})

        # Request payload for Google AI Studio
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "sampleCount": min(number_of_images, 4),
                "aspectRatio": aspect_ratio,
                "safetyFilterLevel": safety_filter_level,
                "personGeneration": person_generation,
                **dimensions
            }
        }
        print(f"[DEBUG] Request payload: {json.dumps(payload, indent=2)}")

        # Make API request
        headers = {
            'Content-Type': 'application/json'
        }

        print("[DEBUG] Sending request to Google AI Studio Imagen API...")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        print(f"[DEBUG] Response status code: {response.status_code}")

        if response.status_code != 200:
            error_detail = response.json() if response.content else response.text
            print(f"[DEBUG] Error response: {error_detail}")
            raise Exception(f"Google AI Studio Imagen API error ({response.status_code}): {error_detail}")

        result = response.json()
        print(f"[DEBUG] Response keys: {result.keys()}")

        # Extract images from response
        images = []
        if 'predictions' in result:
            print(f"[DEBUG] Found {len(result['predictions'])} predictions")
            for i, prediction in enumerate(result['predictions']):
                print(f"[DEBUG] Prediction {i} keys: {prediction.keys()}")
                # Imagen returns base64-encoded images
                if 'bytesBase64Encoded' in prediction:
                    images.append(prediction['bytesBase64Encoded'])
                    print(f"[DEBUG] Added image {i} (length: {len(prediction['bytesBase64Encoded'])} chars)")
                elif 'image' in prediction:
                    # Alternative field name
                    images.append(prediction['image'])
                    print(f"[DEBUG] Added image {i} from 'image' field")
        else:
            print(f"[DEBUG] No 'predictions' key in response. Response: {json.dumps(result, indent=2)}")

        print(f"[DEBUG] Successfully extracted {len(images)} images")
        return {
            'images': images,
            'prompt': prompt,
            'count': len(images),
            'aspect_ratio': aspect_ratio
        }
    except Exception as e:
        print(f"[DEBUG] Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def save_generated_image(image_base64: str, output_path: str) -> str:
    """
    Save a base64-encoded image to a file.

    Args:
        image_base64: Base64-encoded image data
        output_path: Path where to save the image

    Returns:
        Path to saved image
    """
    import base64
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Decode and save
    image_data = base64.b64decode(image_base64)
    with open(output_path, 'wb') as f:
        f.write(image_data)

    return str(output_path)


def estimate_imagen_cost(number_of_images: int) -> float:
    """
    Estimate the cost of generating images with Google AI Studio Imagen.

    Args:
        number_of_images: Number of images to generate

    Returns:
        Estimated cost in USD
    """
    # Google AI Studio Imagen pricing:
    # Free tier: First 50 images per day free
    # After free tier: $0.04 per image
    # Note: This is significantly cheaper than Vertex AI
    COST_PER_IMAGE = 0.04

    return number_of_images * COST_PER_IMAGE


def test_imagen_connection(api_key: str) -> bool:
    """
    Test Google AI Studio Imagen API connection.

    Args:
        api_key: Google AI Studio API key

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Validate API key format
        if not validate_imagen_credentials(api_key):
            return False

        import requests

        # Test with a simple request to check if API key works
        # We'll use the models list endpoint which is cheaper/free
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

        response = requests.get(url, timeout=10)

        # 200 means success, 403 might mean API not enabled but key is valid
        return response.status_code in [200, 403]

    except Exception:
        return False
