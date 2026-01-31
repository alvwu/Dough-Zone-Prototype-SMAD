"""
Google Vertex AI Veo Video Generation Module
Handles video generation using Google Cloud Vertex AI Veo (reuses Vision API credentials).
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def validate_video_credentials(credentials_dict: dict) -> bool:
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


def get_access_token_for_video(credentials_dict: dict) -> str:
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


def generate_video_with_veo(
    prompt: str,
    credentials_dict: dict,
    gcs_output_uri: str,
    aspect_ratio: str = "9:16",
    duration_seconds: int = 5,
    number_of_videos: int = 1,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Generate videos using Google Vertex AI Veo.

    Args:
        prompt: The text prompt for video generation
        credentials_dict: Google Cloud credentials dictionary (same as Vision API)
        gcs_output_uri: Cloud Storage URI for output (e.g., "gs://bucket-name/output/")
        aspect_ratio: Video aspect ratio ("9:16", "16:9", "1:1")
        duration_seconds: Video duration in seconds (5-8)
        number_of_videos: Number of videos to generate (1-4)
        progress_callback: Optional callback function for progress updates

    Returns:
        Dictionary containing video URIs and metadata
    """
    import requests

    print(f"[DEBUG] Starting Vertex AI Veo generation for prompt: {prompt[:50]}...")
    print(f"[DEBUG] Aspect ratio: {aspect_ratio}, Duration: {duration_seconds}s")

    try:
        # Get access token
        print("[DEBUG] Getting access token...")
        access_token = get_access_token_for_video(credentials_dict)
        print("[DEBUG] Access token obtained successfully")

        # Extract project ID
        project_id = credentials_dict['project_id']
        print(f"[DEBUG] Using project ID: {project_id}")

        # Vertex AI Veo endpoint (long-running operation)
        location = "us-central1"
        # Using Veo 2.0 model
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/veo-2.0-generate-001:predictLongRunning"
        print(f"[DEBUG] API endpoint: {url}")

        # Request payload for Veo
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "aspectRatio": aspect_ratio,
                "sampleCount": min(number_of_videos, 4),
                "durationSeconds": duration_seconds,
                "outputGcsUri": gcs_output_uri
            }
        }
        print(f"[DEBUG] Request payload: {json.dumps(payload, indent=2)}")

        # Make API request
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        print("[DEBUG] Sending request to Vertex AI Veo API...")
        if progress_callback:
            progress_callback("Sending request to Veo API...")
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"[DEBUG] Response status code: {response.status_code}")

        if response.status_code != 200:
            error_detail = response.json() if response.content else response.text
            print(f"[DEBUG] Error response: {error_detail}")
            raise Exception(f"Vertex AI Veo API error ({response.status_code}): {error_detail}")

        result = response.json()
        print(f"[DEBUG] Response keys: {result.keys()}")

        # Get the operation name for polling
        operation_name = result.get('name')
        if not operation_name:
            raise Exception("No operation name returned from API")

        print(f"[DEBUG] Operation started: {operation_name}")
        if progress_callback:
            progress_callback("Video generation started, waiting for completion...")

        # Poll for operation completion
        videos = poll_video_operation(
            operation_name=operation_name,
            credentials_dict=credentials_dict,
            progress_callback=progress_callback,
            max_wait_seconds=600  # 10 minutes max
        )

        return {
            'videos': videos,
            'prompt': prompt,
            'count': len(videos),
            'aspect_ratio': aspect_ratio,
            'duration': duration_seconds,
            'gcs_uri': gcs_output_uri
        }

    except Exception as e:
        print(f"[DEBUG] Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def poll_video_operation(
    operation_name: str,
    credentials_dict: dict,
    progress_callback=None,
    max_wait_seconds: int = 600,
    poll_interval: int = 10
) -> list:
    """
    Poll a long-running operation until completion.

    Args:
        operation_name: The operation name/ID to poll
        credentials_dict: Google Cloud credentials dictionary
        progress_callback: Optional callback for progress updates
        max_wait_seconds: Maximum time to wait (default 10 minutes)
        poll_interval: Seconds between polls (default 10)

    Returns:
        List of video URIs
    """
    import requests

    access_token = get_access_token_for_video(credentials_dict)
    
    # Build the operation status URL
    # Operation name format: projects/{project}/locations/{location}/operations/{operation_id}
    url = f"https://us-central1-aiplatform.googleapis.com/v1/{operation_name}"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    start_time = time.time()
    poll_count = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            raise Exception(f"Video generation timed out after {max_wait_seconds} seconds")

        poll_count += 1
        if progress_callback:
            progress_callback(f"Generating video... ({int(elapsed)}s elapsed)")

        print(f"[DEBUG] Polling operation (attempt {poll_count}, {int(elapsed)}s elapsed)...")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"[DEBUG] Poll error: {response.status_code} - {response.text}")
            # Refresh token if needed
            if response.status_code == 401:
                access_token = get_access_token_for_video(credentials_dict)
                headers['Authorization'] = f'Bearer {access_token}'
                continue
            raise Exception(f"Failed to poll operation: {response.text}")

        result = response.json()
        
        # Check if operation is done
        if result.get('done'):
            print("[DEBUG] Operation completed!")
            
            # Check for errors
            if 'error' in result:
                error = result['error']
                raise Exception(f"Video generation failed: {error.get('message', str(error))}")
            
            # Extract video URIs from response
            videos = []
            if 'response' in result:
                response_data = result['response']
                print(f"[DEBUG] Response data keys: {response_data.keys()}")
                
                # Videos are typically in 'generatedSamples' or similar field
                if 'generatedSamples' in response_data:
                    for sample in response_data['generatedSamples']:
                        if 'video' in sample:
                            video_info = sample['video']
                            if 'uri' in video_info:
                                videos.append({
                                    'uri': video_info['uri'],
                                    'state': video_info.get('state', 'SUCCEEDED')
                                })
                elif 'predictions' in response_data:
                    for prediction in response_data['predictions']:
                        if 'gcsUri' in prediction:
                            videos.append({
                                'uri': prediction['gcsUri'],
                                'state': 'SUCCEEDED'
                            })
                        elif 'video' in prediction and 'uri' in prediction['video']:
                            videos.append({
                                'uri': prediction['video']['uri'],
                                'state': prediction['video'].get('state', 'SUCCEEDED')
                            })
            
            if not videos:
                print(f"[DEBUG] Full response: {json.dumps(result, indent=2)}")
                raise Exception("No videos found in the response")
            
            return videos

        # Not done yet, wait and poll again
        print(f"[DEBUG] Operation still running, state: {result.get('metadata', {}).get('state', 'UNKNOWN')}")
        time.sleep(poll_interval)


def download_video_from_gcs(
    gcs_uri: str,
    credentials_dict: dict,
    output_path: str
) -> str:
    """
    Download a video from Google Cloud Storage.

    Args:
        gcs_uri: GCS URI (e.g., "gs://bucket/path/video.mp4")
        credentials_dict: Google Cloud credentials dictionary
        output_path: Local path to save the video

    Returns:
        Path to downloaded video
    """
    import requests
    from pathlib import Path
    from urllib.parse import urlparse

    print(f"[DEBUG] Downloading video from: {gcs_uri}")

    # Parse GCS URI
    # Format: gs://bucket-name/path/to/file
    if not gcs_uri.startswith('gs://'):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    uri_parts = gcs_uri[5:].split('/', 1)  # Remove 'gs://' and split
    bucket_name = uri_parts[0]
    object_path = uri_parts[1] if len(uri_parts) > 1 else ''
    
    print(f"[DEBUG] Bucket: {bucket_name}, Object: {object_path}")

    # Get access token
    access_token = get_access_token_for_video(credentials_dict)

    # Use the JSON API to download
    # URL encode the object path
    from urllib.parse import quote
    encoded_object = quote(object_path, safe='')
    
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{encoded_object}?alt=media"
    
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    print(f"[DEBUG] Download URL: {url}")
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        print(f"[DEBUG] Download error: {response.status_code} - {response.text}")
        raise Exception(f"Failed to download video: {response.status_code} - {response.text}")

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[DEBUG] Video saved to: {output_path}")
    return str(output_path)


def create_gcs_bucket_if_needed(
    bucket_name: str,
    credentials_dict: dict,
    location: str = "us-central1"
) -> bool:
    """
    Create a GCS bucket if it doesn't exist.

    Args:
        bucket_name: Name of the bucket to create
        credentials_dict: Google Cloud credentials dictionary
        location: Location for the bucket

    Returns:
        True if bucket exists or was created, False otherwise
    """
    import requests

    access_token = get_access_token_for_video(credentials_dict)
    project_id = credentials_dict['project_id']

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # Check if bucket exists
    check_url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}"
    response = requests.get(check_url, headers=headers)

    if response.status_code == 200:
        print(f"[DEBUG] Bucket {bucket_name} already exists")
        return True

    # Create bucket
    create_url = f"https://storage.googleapis.com/storage/v1/b?project={project_id}"
    payload = {
        "name": bucket_name,
        "location": location,
        "storageClass": "STANDARD"
    }

    response = requests.post(create_url, json=payload, headers=headers)

    if response.status_code in [200, 409]:  # 409 = already exists
        print(f"[DEBUG] Bucket {bucket_name} created or already exists")
        return True
    else:
        print(f"[DEBUG] Failed to create bucket: {response.status_code} - {response.text}")
        return False


def get_default_bucket_name(credentials_dict: dict) -> str:
    """
    Get a default bucket name based on project ID.

    Args:
        credentials_dict: Google Cloud credentials dictionary

    Returns:
        Default bucket name string
    """
    project_id = credentials_dict.get('project_id', 'unknown')
    return f"{project_id}-generated-videos"


def estimate_video_cost(duration_seconds: int, number_of_videos: int = 1) -> float:
    """
    Estimate the cost of generating videos with Vertex AI Veo.

    Args:
        duration_seconds: Duration of each video
        number_of_videos: Number of videos to generate

    Returns:
        Estimated cost in USD
    """
    # Vertex AI Veo pricing (approximate):
    # ~$0.35 per second of video generated
    COST_PER_SECOND = 0.35

    return duration_seconds * number_of_videos * COST_PER_SECOND


def test_video_connection(credentials_dict: dict) -> bool:
    """
    Test Vertex AI Veo API connection.

    Args:
        credentials_dict: Google Cloud credentials dictionary

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Try to get access token
        access_token = get_access_token_for_video(credentials_dict)

        # Verify project exists and we can access Vertex AI
        project_id = credentials_dict['project_id']
        location = "us-central1"

        import requests

        # Simple request to verify access
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/veo-2.0-generate-001"

        headers = {
            'Authorization': f'Bearer {access_token}'
        }

        response = requests.get(url, headers=headers)

        # 200 or 403 both mean credentials work (403 might mean API not enabled)
        return response.status_code in [200, 403, 404]

    except Exception as e:
        print(f"[DEBUG] Connection test failed: {str(e)}")
        return False
