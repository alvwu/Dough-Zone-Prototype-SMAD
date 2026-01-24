"""
Google Vertex AI Imagen Integration Module
Handles image generation using Google Cloud Vertex AI Imagen (reuses Vision API credentials).
"""

import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time


def validate_imagen_credentials(credentials_dict: dict) -> bool:
    """
    Validate Vertex AI credentials by checking required fields.

    Args:
        credentials_dict: Parsed JSON credentials dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        required_fields = ['private_key', 'client_email', 'project_id']
        for field in required_fields:
            if field not in credentials_dict:
                return False
        return True
    except Exception:
        return False


def get_access_token_for_imagen(credentials_dict: dict) -> str:
    """
    Get an access token from service account credentials for Vertex AI.

    Args:
        credentials_dict: The parsed JSON credentials dictionary

    Returns:
        Access token string
    """
    import jwt
    from datetime import datetime, timedelta
    import requests

    # Extract required fields
    private_key = credentials_dict['private_key']
    client_email = credentials_dict['client_email']
    token_uri = credentials_dict.get('token_uri', 'https://oauth2.googleapis.com/token')

    # Create JWT
    now = datetime.utcnow()
    payload = {
        'iss': client_email,
        'sub': client_email,
        'aud': token_uri,
        'iat': now,
        'exp': now + timedelta(hours=1),
        'scope': 'https://www.googleapis.com/auth/cloud-platform'
    }

    # Sign JWT with private key
    signed_jwt = jwt.encode(payload, private_key, algorithm='RS256')

    # Exchange JWT for access token
    response = requests.post(
        token_uri,
        data={
            'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
            'assertion': signed_jwt
        }
    )

    if response.status_code != 200:
        raise Exception(f"Failed to get access token: {response.text}")

    return response.json()['access_token']


def generate_image_with_imagen(
    prompt: str,
    credentials_dict: dict,
    number_of_images: int = 1,
    aspect_ratio: str = "1:1",
    safety_filter_level: str = "block_some",
    person_generation: str = "allow_adult"
) -> Dict[str, Any]:
    """
    Generate images using Google Vertex AI Imagen.

    Args:
        prompt: The text prompt for image generation
        credentials_dict: Google Cloud credentials dictionary (same as Vision API)
        number_of_images: Number of images to generate (1-4)
        aspect_ratio: Image aspect ratio ("1:1", "9:16", "16:9", "4:3", "3:4")
        safety_filter_level: Safety filter level ("block_most", "block_some", "block_few", "block_fewest")
        person_generation: Person generation policy ("allow_adult", "allow_all", "dont_allow")

    Returns:
        Dictionary containing generated images (as base64) and metadata
    """
    import requests

    print(f"[DEBUG] Starting Vertex AI Imagen generation for prompt: {prompt[:50]}...")
    print(f"[DEBUG] Aspect ratio: {aspect_ratio}")

    try:
        # Get access token
        print("[DEBUG] Getting access token...")
        access_token = get_access_token_for_imagen(credentials_dict)
        print("[DEBUG] Access token obtained successfully")

        # Extract project ID
        project_id = credentials_dict['project_id']
        print(f"[DEBUG] Using project ID: {project_id}")

        # Vertex AI Imagen endpoint
        location = "us-central1"  # Imagen is available in us-central1
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/imagegeneration@006:predict"
        print(f"[DEBUG] API endpoint: {url}")

        # Request payload
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "sampleCount": min(number_of_images, 4),  # Max 4 images
                "aspectRatio": aspect_ratio,
                "safetyFilterLevel": safety_filter_level,
                "personGeneration": person_generation
            }
        }
        print(f"[DEBUG] Request payload: {json.dumps(payload, indent=2)}")

        # Make API request
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        print("[DEBUG] Sending request to Vertex AI Imagen API...")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        print(f"[DEBUG] Response status code: {response.status_code}")

        if response.status_code != 200:
            error_detail = response.json() if response.content else response.text
            print(f"[DEBUG] Error response: {error_detail}")
            raise Exception(f"Vertex AI Imagen API error ({response.status_code}): {error_detail}")

        result = response.json()
        print(f"[DEBUG] Response keys: {result.keys()}")

        # Extract images from response
        images = []
        if 'predictions' in result:
            print(f"[DEBUG] Found {len(result['predictions'])} predictions")
            for i, prediction in enumerate(result['predictions']):
                print(f"[DEBUG] Prediction {i} keys: {prediction.keys()}")
                # Imagen returns base64-encoded images in 'bytesBase64Encoded' field
                if 'bytesBase64Encoded' in prediction:
                    images.append(prediction['bytesBase64Encoded'])
                    print(f"[DEBUG] Added image {i} (length: {len(prediction['bytesBase64Encoded'])} chars)")
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
    Estimate the cost of generating images with Vertex AI Imagen.

    Args:
        number_of_images: Number of images to generate

    Returns:
        Estimated cost in USD
    """
    # Vertex AI Imagen pricing:
    # - $0.020 per image for standard resolution (1024x1024)
    COST_PER_IMAGE = 0.020

    return number_of_images * COST_PER_IMAGE


def test_imagen_connection(credentials_dict: dict) -> bool:
    """
    Test Vertex AI Imagen API connection.

    Args:
        credentials_dict: Google Cloud credentials dictionary

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Try to get access token
        access_token = get_access_token_for_imagen(credentials_dict)

        # Verify project exists and we can access Vertex AI
        project_id = credentials_dict['project_id']
        location = "us-central1"

        import requests

        # Simple request to verify access
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/imagegeneration@006"

        headers = {
            'Authorization': f'Bearer {access_token}'
        }

        response = requests.get(url, headers=headers)

        # 200 or 403 both mean credentials work (403 might mean API not enabled)
        return response.status_code in [200, 403]

    except Exception:
        return False
