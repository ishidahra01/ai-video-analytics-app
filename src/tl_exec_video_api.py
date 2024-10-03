import requests
import json
import time
import dotenv
import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
import cv2
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# Load the necessary environment variables
def load_environment_variables():
    dotenv.load_dotenv(dotenv_path=".env.video-analytics-vars", override=True)
    return {
        "VIDEO_FILE_URL": os.getenv("VIDEO_FILE_URL"),
        "AZURE_BLOB_OUTPUT_CONTAINER_URL": os.getenv("AZURE_BLOB_OUTPUT_CONTAINER_URL"),
        "VIDEO_DESCRIPTION_TASK_NAME": os.getenv("VIDEO_DESCRIPTION_TASK_NAME"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        "AZURE_OPENAI_COMPLETIONS_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_COMPLETIONS_DEPLOYMENT_NAME"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_AI_SERVICES_ENDPOINT": os.getenv("AZURE_AI_SERVICES_ENDPOINT"),
        "AZURE_AI_SERVICES_API_KEY": os.getenv("AZURE_AI_SERVICES_API_KEY"),
        "BLOB_CONNECTION_STRING": os.getenv("BLOB_CONNECTION_STRING")
    }


# Construct the request URL
def get_video_description_request_url(azure_ai_services_endpoint, task_name):
    return f"{azure_ai_services_endpoint}/computervision/videoanalysis/videodescriptions/{task_name}?api-version=2024-05-01-preview"


# Create the body payload for the video description request
def create_video_description_body_payload(video_file_url, blob_output_url, chat_completions_endpoint, azure_openai_authentication):
    return {
        "input": {
            "kind": "azureBlobFile",
            "url": video_file_url,
            "authentication": {"kind": "managedIdentity"},
        },
        "output": {
            "kind": "azureBlobContainer",
            "url": blob_output_url,
            "authentication": {"kind": "managedIdentity"},
        },
        "resource": {
            "completion": {
                "kind": "gptv",
                "endpoint": chat_completions_endpoint,
                "authentication": azure_openai_authentication,
            }
        },
        "domain": "Default",
        "properties": {
            "taskDescription": {
                "kind": "describe",
                "description": "Describe the video content."
            }
        }
    }


# Create video description task
def create_video_description(task_name, video_file_url, blob_output_url, headers_list, azure_ai_services_endpoint, chat_completions_endpoint, azure_openai_authentication):
    payload = json.dumps(create_video_description_body_payload(video_file_url, blob_output_url, chat_completions_endpoint, azure_openai_authentication))
    request_url = get_video_description_request_url(azure_ai_services_endpoint, task_name)
    response = requests.put(request_url, data=payload, headers=headers_list)
    return response.text


# Get task status
def get_video_description(task_name, azure_ai_services_endpoint, headers_list):
    request_url = get_video_description_request_url(azure_ai_services_endpoint, task_name)
    response = requests.get(request_url, headers=headers_list)
    return response.json()


# Delete a task
def delete_task(task_name, azure_ai_services_endpoint, headers_list):
    request_url = get_video_description_request_url(azure_ai_services_endpoint, task_name)
    response = requests.delete(request_url, headers=headers_list)
    return response.text


# Poll for task completion
def generate_video_description(task_name, video_url, blob_file_url, headers_list, azure_ai_services_endpoint, chat_completions_endpoint, azure_openai_authentication, delta=10):
    create_video_description(task_name, video_url, blob_file_url, headers_list, azure_ai_services_endpoint, chat_completions_endpoint, azure_openai_authentication)
    attempts = 0
    while True:
        response = get_video_description(task_name, azure_ai_services_endpoint, headers_list)
        status = response["status"]
        if status in ["completed", "partiallyCompleted"]:
            print(f"Task completed: {task_name}")
            return response
            break
        if status in ["running", "notStarted"]:
            time.sleep(delta)
        else:
            print(f"Task status: {status}")
            attempts += 1
            if attempts == 3:
                print(f"Task failed {attempts} times.")
                return response
                break
            delete_task(task_name, azure_ai_services_endpoint, headers_list)
            create_video_description(task_name, video_url, blob_file_url, headers_list, azure_ai_services_endpoint, chat_completions_endpoint, azure_openai_authentication)


# Download the blob result
def download_blob_to_file(connection_string, container_name, blob_name, download_file_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(download_file_path, "wb") as download_file:
        download_stream = blob_client.download_blob()
        download_file.write(download_stream.readall())
    return download_file_path


# Extract keyframes from video
def extract_keyframes_from_video(video_path, key_frames):
    cap = cv2.VideoCapture(video_path)
    for key_frame in key_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, key_frame * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame at {key_frame} seconds")
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 4))
        plt.imshow(frame_rgb)
        plt.title(f"Key Frame at {key_frame:.3f} s")
        plt.axis('off')
        plt.show()
    cap.release()


# Extract a segment from a video
def extract_segment_from_video(video_path, offset, duration, output_path):
    ffmpeg_extract_subclip(video_path, offset, offset + duration, targetname=output_path)


if __name__ == "__main__":
    # Load environment variables
    env_vars = load_environment_variables()

    # Set headers for API calls
    headers_list = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": env_vars["AZURE_AI_SERVICES_API_KEY"]
    }

    # Example of generating a video description
    generate_video_description(
        task_name=env_vars["VIDEO_DESCRIPTION_TASK_NAME"],
        video_url=env_vars["VIDEO_FILE_URL"],
        blob_file_url=env_vars["AZURE_BLOB_OUTPUT_CONTAINER_URL"],
        headers_list=headers_list,
        azure_ai_services_endpoint=env_vars["AZURE_AI_SERVICES_ENDPOINT"],
        chat_completions_endpoint=f'{env_vars["AZURE_OPENAI_ENDPOINT"]}/openai/deployments/{env_vars["AZURE_OPENAI_COMPLETIONS_DEPLOYMENT_NAME"]}/chat/completions?api-version={env_vars["AZURE_OPENAI_API_VERSION"]}',
        azure_openai_authentication={"kind": "key", "key": env_vars["AZURE_OPENAI_API_KEY"]}
    )
