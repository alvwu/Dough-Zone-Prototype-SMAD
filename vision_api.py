"""
Google Vision API Integration Module
Handles image analysis using Google Cloud Vision API.
"""

import base64
import json
import time
import requests
from pathlib import Path


def get_access_token_from_credentials(credentials_dict: dict) -> str:
    """
    Get an access token from service account credentials using JWT.

    Args:
        credentials_dict: The parsed JSON credentials dictionary

    Returns:
        Access token string
    """
    import jwt
    from datetime import datetime, timedelta

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
        'scope': 'https://www.googleapis.com/auth/cloud-vision'
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


def analyze_image_with_credentials(image_path: str, credentials_dict: dict) -> dict:
    """
    Analyze an image using Google Cloud Vision API with service account credentials.

    Args:
        image_path: Path to the image file
        credentials_dict: Parsed JSON credentials dictionary

    Returns:
        Dictionary containing labels, colors, objects, and text detected
    """
    # Get access token
    access_token = get_access_token_from_credentials(credentials_dict)

    # Read and encode image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, 'rb') as image_file:
        image_content = base64.b64encode(image_file.read()).decode('utf-8')

    # Vision API endpoint (without API key, using OAuth)
    url = "https://vision.googleapis.com/v1/images:annotate"

    # Request payload
    payload = {
        "requests": [{
            "image": {
                "content": image_content
            },
            "features": [
                {"type": "LABEL_DETECTION", "maxResults": 10},
                {"type": "IMAGE_PROPERTIES", "maxResults": 5},
                {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                {"type": "TEXT_DETECTION", "maxResults": 10}
            ]
        }]
    }

    # Make API request with Bearer token
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        error_msg = response.json().get('error', {}).get('message', 'Unknown error')
        raise Exception(f"Vision API error: {error_msg}")

    return parse_vision_response(response.json())


def analyze_image_with_vision_api(image_path: str, api_key: str = None, credentials_dict: dict = None) -> dict:
    """
    Analyze an image using Google Cloud Vision API.

    Args:
        image_path: Path to the image file
        api_key: Google Cloud Vision API key (option 1)
        credentials_dict: Parsed JSON credentials dictionary (option 2)

    Returns:
        Dictionary containing labels, colors, objects, and text detected
    """
    # Use credentials if provided, otherwise use API key
    if credentials_dict:
        return analyze_image_with_credentials(image_path, credentials_dict)

    if not api_key:
        raise ValueError("Either api_key or credentials_dict must be provided")

    # Read and encode image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, 'rb') as image_file:
        image_content = base64.b64encode(image_file.read()).decode('utf-8')

    # Vision API endpoint
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    # Request payload - request multiple feature types
    payload = {
        "requests": [{
            "image": {
                "content": image_content
            },
            "features": [
                {"type": "LABEL_DETECTION", "maxResults": 10},
                {"type": "IMAGE_PROPERTIES", "maxResults": 5},
                {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                {"type": "TEXT_DETECTION", "maxResults": 10}
            ]
        }]
    }

    # Make API request
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        error_msg = response.json().get('error', {}).get('message', 'Unknown error')
        raise Exception(f"Vision API error: {error_msg}")

    return parse_vision_response(response.json())


def parse_vision_response(result: dict) -> dict:
    """Parse the Vision API response and extract relevant information."""
    # Parse response
    if 'responses' not in result or len(result['responses']) == 0:
        raise Exception("No response from Vision API")

    api_response = result['responses'][0]

    # Check for errors in response
    if 'error' in api_response:
        raise Exception(f"Vision API error: {api_response['error'].get('message', 'Unknown error')}")

    # Extract labels
    labels = []
    if 'labelAnnotations' in api_response:
        labels = [label['description'] for label in api_response['labelAnnotations']]

    # Extract dominant colors
    colors = []
    if 'imagePropertiesAnnotation' in api_response:
        color_info = api_response['imagePropertiesAnnotation'].get('dominantColors', {}).get('colors', [])
        for color in color_info[:5]:  # Top 5 colors
            rgb = color.get('color', {})
            r = int(rgb.get('red', 0))
            g = int(rgb.get('green', 0))
            b = int(rgb.get('blue', 0))
            color_name = get_color_name(r, g, b)
            if color_name and color_name not in colors:
                colors.append(color_name)

    # Extract objects
    objects = []
    if 'localizedObjectAnnotations' in api_response:
        objects = list(set([obj['name'] for obj in api_response['localizedObjectAnnotations']]))

    # Extract text
    text = ""
    if 'textAnnotations' in api_response and len(api_response['textAnnotations']) > 0:
        text = api_response['textAnnotations'][0].get('description', '').replace('\n', ' ').strip()

    return {
        'labels': ', '.join(labels) if labels else None,
        'dominant_colors': ', '.join(colors) if colors else None,
        'objects_detected': ', '.join(objects) if objects else None,
        'text_detected': text if text else None
    }


def validate_credentials(credentials_dict: dict) -> bool:
    """
    Validate service account credentials by attempting to get an access token.

    Args:
        credentials_dict: Parsed JSON credentials dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        required_fields = ['private_key', 'client_email', 'project_id']
        for field in required_fields:
            if field not in credentials_dict:
                return False

        # Try to get an access token
        get_access_token_from_credentials(credentials_dict)
        return True
    except Exception:
        return False


def get_color_name(r: int, g: int, b: int) -> str:
    """Convert RGB values to a color name."""
    # Simple color classification based on RGB values
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (139, 69, 19),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'beige': (245, 245, 220),
        'gold': (255, 215, 0),
        'teal': (0, 128, 128),
        'navy': (0, 0, 128),
        'maroon': (128, 0, 0),
        'olive': (128, 128, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
    }

    # Find closest color
    min_distance = float('inf')
    closest_color = 'unknown'

    for color_name, (cr, cg, cb) in colors.items():
        distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color


def validate_api_key(api_key: str) -> bool:
    """
    Validate the Google Vision API key by making a simple request.

    Args:
        api_key: Google Cloud Vision API key

    Returns:
        True if valid, False otherwise
    """
    if not api_key or len(api_key) < 20:
        return False

    # Create a minimal test request with a tiny image
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    # 1x1 transparent PNG
    tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    payload = {
        "requests": [{
            "image": {"content": tiny_image},
            "features": [{"type": "LABEL_DETECTION", "maxResults": 1}]
        }]
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        # API key is valid if we don't get a 400/403 error
        if response.status_code == 200:
            return True
        elif response.status_code in [400, 403]:
            error = response.json().get('error', {})
            # Check if it's an API key error vs other errors
            if 'API key' in str(error) or 'permission' in str(error).lower():
                return False
            # Other errors might just mean the request was malformed but key is valid
            return True
        return False
    except Exception:
        return False
