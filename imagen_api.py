"""
Google AI Studio Nano Banana API Integration Module
Handles image generation using Google AI Studio's Nano Banana (Gemini Image) API.
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
    Generate images using Google AI Studio Nano Banana API (Gemini Image).

    Args:
        prompt: The text prompt for image generation
        api_key: Google AI Studio API key
        number_of_images: Number of images to generate (1-4)
        aspect_ratio: Image aspect ratio ("1:1", "9:16", "16:9", "4:3", "3:4")
        safety_filter_level: Safety filter level (kept for compatibility, not used in Gemini API)
        person_generation: Person generation policy (kept for compatibility, not used in Gemini API)

    Returns:
        Dictionary containing generated images (as base64) and metadata
    """
    import requests

    print(f"[DEBUG] Starting Nano Banana image generation for prompt: {prompt[:50]}...")
    print(f"[DEBUG] Aspect ratio: {aspect_ratio}")

    try:
        # Use Gemini 2.5 Flash Image (Nano Banana) for faster generation
        # For higher quality, use "gemini-3-pro-image-preview"
        model = "gemini-2.5-flash-image"

        # Google AI Studio Nano Banana endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        print(f"[DEBUG] API endpoint: {url}")

        # Add aspect ratio guidance to the prompt
        aspect_ratio_prompts = {
            "1:1": "square format, 1:1 aspect ratio",
            "9:16": "vertical format, 9:16 aspect ratio for mobile/stories",
            "16:9": "horizontal widescreen format, 16:9 aspect ratio",
            "4:3": "horizontal format, 4:3 aspect ratio",
            "3:4": "vertical format, 3:4 aspect ratio"
        }

        # Enhance prompt with aspect ratio guidance
        enhanced_prompt = f"{prompt}. Format: {aspect_ratio_prompts.get(aspect_ratio, 'square format')}"

        # Request payload for Google AI Studio Nano Banana
        payload = {
            "contents": [{
                "parts": [
                    {"text": enhanced_prompt}
                ]
            }]
        }
        print(f"[DEBUG] Request payload: {json.dumps(payload, indent=2)}")

        # Make API request with API key in header
        headers = {
            'x-goog-api-key': api_key,
            'Content-Type': 'application/json'
        }

        print("[DEBUG] Sending request to Nano Banana API...")

        # Generate multiple images if requested
        images = []
        for i in range(min(number_of_images, 4)):
            print(f"[DEBUG] Generating image {i+1}/{min(number_of_images, 4)}...")
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            print(f"[DEBUG] Response status code: {response.status_code}")

            if response.status_code != 200:
                error_detail = response.json() if response.content else response.text
                print(f"[DEBUG] Error response: {error_detail}")
                raise Exception(f"Nano Banana API error ({response.status_code}): {error_detail}")

            result = response.json()
            print(f"[DEBUG] Response keys: {result.keys()}")

            # Extract images from response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                print(f"[DEBUG] Candidate keys: {candidate.keys()}")

                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    print(f"[DEBUG] Found {len(parts)} parts in response")

                    for part_idx, part in enumerate(parts):
                        print(f"[DEBUG] Part {part_idx} keys: {part.keys()}")
                        # Look for inline_data with image
                        if 'inline_data' in part or 'inlineData' in part:
                            inline_data = part.get('inline_data') or part.get('inlineData')
                            if 'data' in inline_data:
                                images.append(inline_data['data'])
                                print(f"[DEBUG] Added image {len(images)} (length: {len(inline_data['data'])} chars)")
                            elif 'image' in inline_data:
                                images.append(inline_data['image'])
                                print(f"[DEBUG] Added image {len(images)} from 'image' field")
            else:
                print(f"[DEBUG] No 'candidates' key in response. Response: {json.dumps(result, indent=2)}")

        if not images:
            raise Exception("No images were generated by the Nano Banana API")

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
    Estimate the cost of generating images with Google AI Studio Nano Banana.

    Args:
        number_of_images: Number of images to generate

    Returns:
        Estimated cost in USD
    """
    # Google AI Studio Nano Banana pricing:
    # Free tier: First 50 images per day free
    # After free tier: $0.04 per image for Gemini 2.5 Flash Image
    # Gemini 3 Pro Image Preview: $0.08 per image
    COST_PER_IMAGE = 0.04

    return number_of_images * COST_PER_IMAGE


def test_imagen_connection(api_key: str) -> bool:
    """
    Test Google AI Studio Nano Banana API connection.

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

        # Test with a simple request to the models endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image"

        headers = {
            'x-goog-api-key': api_key
        }

        response = requests.get(url, headers=headers, timeout=10)

        # 200 means success, 403 might mean API not enabled but key is valid
        return response.status_code in [200, 403]

    except Exception:
        return False
