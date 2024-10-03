import streamlit as st
import requests
import json
import cv2
from PIL import Image
import os
import os
from openai import AzureOpenAI
import json
import pandas as pd

from tl_exec_video_api import generate_video_description

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.storage.blob import (
    BlobServiceClient,
    ContainerClient,
    BlobClient,
    BlobSasPermissions,
    ContainerSasPermissions,
    UserDelegationKey,
    generate_container_sas,
    generate_blob_sas
)

import datetime
import dotenv

dotenv.load_dotenv(dotenv_path=".env.video-analytics-vars", override=True)

VIDEO_FILE_URL = os.getenv("VIDEO_FILE_URL")
AZURE_BLOB_OUTPUT_CONTAINER_URL = os.getenv("AZURE_BLOB_OUTPUT_CONTAINER_URL")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_COMPLETIONS_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_COMPLETIONS_DEPLOYMENT_NAME"
)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_AI_SERVICES_ENDPOINT = os.getenv("AZURE_AI_SERVICES_ENDPOINT")
AZURE_AI_SERVICES_API_KEY = os.getenv("AZURE_AI_SERVICES_API_KEY")

print(os.getenv("AZURE_TENANT_ID"))
print(os.getenv("AZURE_CLIENT_ID"))
print(os.getenv("AZURE_CLIENT_SECRET"))


# # Setting for Azure Blob Storage
# credential = ClientSecretCredential(os.getenv("AZURE_TENANT_ID"), os.getenv("AZURE_CLIENT_ID"), os.getenv("AZURE_CLIENT_SECRET"))

credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=f"https://stllmopswork533951945278.blob.core.windows.net",
    credential=credential
)

def request_user_delegation_key(blob_service_client: BlobServiceClient):
    # Get a user delegation key that's valid for 1 day
    delegation_key_start_time = datetime.datetime.now(datetime.timezone.utc)
    delegation_key_expiry_time = delegation_key_start_time + datetime.timedelta(days=1)

    user_delegation_key = blob_service_client.get_user_delegation_key(
        key_start_time=delegation_key_start_time,
        key_expiry_time=delegation_key_expiry_time
    )

    return user_delegation_key

def create_user_delegation_sas_blob(blob_client: BlobClient, user_delegation_key: UserDelegationKey):
    # Create a SAS token that's valid for one day, as an example
    start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    expiry_time = start_time + datetime.timedelta(days=4)

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=blob_client.container_name,
        blob_name=blob_client.blob_name,
        user_delegation_key=user_delegation_key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry_time,
        start=start_time
    )

    return sas_token


def upload_to_blob(file_path, blob_name):
    blob_client = blob_service_client.get_blob_client(container="analytics-video", blob=blob_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    blob_url = blob_client.url
    user_delegation_key = request_user_delegation_key(blob_service_client)
    sas_token = create_user_delegation_sas_blob(blob_client, user_delegation_key)
    return blob_url, sas_token
    

# Download the blob result
def download_blob_to_file(container_name, blob_name, download_file_path):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(download_file_path, "wb") as download_file:
        download_stream = blob_client.download_blob()
        download_file.write(download_stream.readall())
    return download_file_path


# Streamlit inputs for GPT prompts
st.sidebar.title("Settings")

# Sidebar input for VIDEO_DESCRIPTION_TASK_NAME
video_description_task_name = st.sidebar.text_input(
    "Video Description Task Name", value="default_task_name"
)

# Streamlit inputs for GPT prompts
st.sidebar.title("Prompt for Caption Generation")

# Sidebar input for User Message
user_message_input_caption = st.sidebar.text_area(
    "User Message for Caption Generation",
)

# Streamlit inputs for GPT prompts
st.sidebar.title("Prompt for Postpocess Generation")

user_message_input_postprocess = st.sidebar.text_area(
    "User Message for postprocess Generation",
)

def create_caption_by_gpt(image_url):
    print("Image URL:", image_url)
    print("Generating caption using GPT-4o model...")
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )

    message_text = [
        {"role": "system", "content": "Perform tasks related to the input image based on the user's prompt instructions."},  # Using user input for system message
        {"role": "user", "content": [
            {
                "type": "text",
                "content": user_message_input_caption  # Using user input for user message
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message_text,
        temperature=0,
        max_tokens=200,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def postprocess_by_gpt(text):
    print(f"Input Text: {text}")
    print("Generating task list using GPT-4o model...")
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    
    system_message = f"""
    # Your role
    You are a excellent AI Assitant. You should complete the task based on the user's request.
    The user's request is to generate a postprocessed output based on the input data.
    
    # Your input data
    {text}
    
    """

    message_text = [
        {"role": "system", "content": f"{system_message}"},  # Using user input for system message
        {"role": "user", "content": [
            {
                "type": "text",
                "content": f"{user_message_input_postprocess}"  # Using user input for user message
            }
        ]}
    ]

    print(f"Message Text: {message_text}")

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message_text,
        # response_format={"type": "json_object"},
        temperature=0,
        max_tokens=4000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

# Add CSS to customize the style
st.markdown("""
    <style>
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .keyframe-button {
        margin: 5px;
    }
    .stExpander {
        margin-bottom: 10px;
    }
    .stExpander > div > div {
        padding: 10px; /* 内側の余白を調整 */
    }
    </style>
    """, unsafe_allow_html=True)

# Function for keyframe extraction and saving function
def extract_keyframe_and_save(video_path, keyframe_time, output_dir):
    # loading video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # convert time to frame number
    time_parts = keyframe_time.strip('PT').split('M')
    minutes = int(time_parts[0]) if len(time_parts) > 1 else 0
    seconds = float(time_parts[1][:-1]) if len(time_parts) > 1 else float(time_parts[0][:-1])
    frame_number = int((minutes * 60 + seconds) * fps)
    
    # set frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # load frame
    success, frame = cap.read()
    cap.release()

    if success:
        # save frame
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_path = os.path.join(output_dir, f"{keyframe_time}.jpg")
        cv2.imwrite(image_path, frame)
        
        image_path = os.path.normpath(image_path).replace('\\', '/')
        return image_path
    else:
        return None

# convert time to string
def format_time(keyframe_time):
    time_parts = keyframe_time.strip('PT').split('M')
    minutes = int(time_parts[0]) if len(time_parts) > 1 else 0
    seconds = float(time_parts[1][:-1]) if len(time_parts) > 1 else float(time_parts[0][:-1])
    return f"{minutes} min {seconds} sec"


# Initialize session state variables if they do not exist
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "frame_caption_list" not in st.session_state:
    st.session_state.frame_caption_list = []
if "postprocessed_outputs" not in st.session_state:
    st.session_state.postprocessed_outputs = {}


# path to the folder where the uploaded videos will be stored
upload_folder = 'uploaded_videos'

# make the folder if it does not exist
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

VIDEO_DESCRIPTION_TASK_NAME = video_description_task_name

# upload video file
st.title('Video Analytics App')
uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])


if uploaded_file is not None:
    
    video_path = os.path.join(upload_folder, uploaded_file.name)
    
    # save the uploaded video file
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    project_folder = "99_streamlit_app/toyota-logistics"
    blob_container_name = "analytics-video"
      
    video_blob_url, video_sas_token = upload_to_blob(video_path, f"{project_folder}/input/{uploaded_file.name}")
    
    # display the uploaded video
    st.video(uploaded_file)

    # show the video URL
    if st.button('Analyze Video'):
        
        frame_caption_list = []
        
        # Send the request to API
        files = {'file': uploaded_file.getvalue()}
        
        headers_list = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": AZURE_AI_SERVICES_API_KEY
        }
        
        print("Generating video description...")
        
        response_api = generate_video_description(
            task_name=VIDEO_DESCRIPTION_TASK_NAME,
            video_url=video_blob_url,
            blob_file_url=f"https://stllmopswork533951945278.blob.core.windows.net/{blob_container_name}/{project_folder}/output",
            headers_list=headers_list,
            azure_ai_services_endpoint=AZURE_AI_SERVICES_ENDPOINT,
            chat_completions_endpoint=f'{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_COMPLETIONS_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}',
            azure_openai_authentication={"kind": "key", "key": AZURE_OPENAI_API_KEY}
        )
        
        print(response_api)
        
        if response_api['status'] == "completed":
            
            file_name = "videoDescription.json"
            blob_name = f"{project_folder}/output/{VIDEO_DESCRIPTION_TASK_NAME}/{file_name}"
            download_dir = "output"
            download_file_path = f"{download_dir}/{VIDEO_DESCRIPTION_TASK_NAME}_{file_name}"

            os.makedirs(download_dir, exist_ok=True)
            
            download_file_path = download_blob_to_file(blob_container_name, blob_name, download_file_path)
            
            with open(download_file_path, 'r', encoding='utf-8-sig') as file:
                analysis_result = json.load(file)
            print(analysis_result)
            
            # show the analysis result
            st.session_state.analysis_result = analysis_result
            st.subheader('analysis result')
            st.write(f"total segments: {analysis_result['metadata']['totalSegments']}")
            
            # generate the list of keyframes
            for segment in analysis_result['videoSegments']:
                output_dir = "keyframes"
                for keyframe in segment['properties']['keyFrames']:
                    image_path = extract_keyframe_and_save(video_path, keyframe, output_dir)
                    print(image_path)
                    blob_url, sas_token = upload_to_blob(image_path, f"99_streamlit_app/toyota-logistics/{image_path}")
                    caption = create_caption_by_gpt(f"{blob_url}?{sas_token}")
                    if image_path:
                        frame_caption = {
                            "image_path": image_path,
                            "segment_id": segment['id'],
                            "frame_time": format_time(keyframe),
                            "caption": caption
                        }
                        frame_caption_list.append(frame_caption)
                    else:
                        st.write(f"Failed to extract keyframe {keyframe}")
            st.session_state.frame_caption_list = frame_caption_list

        else:
            st.error('Failed to analyze video')


# Display frame captions if available in session state
if st.session_state.frame_caption_list:
    
    st.subheader('analysis result')
    st.write(f"total segments: {st.session_state.analysis_result['metadata']['totalSegments']}")
    
    for segment in st.session_state.analysis_result['videoSegments']:
        with st.expander(f"segment ID: {segment['id']} (offset: {segment['offset']}, duration: {segment['duration']})", expanded=False):
            st.write(f"**Description**: {segment['properties']['description']}")
            st.write('**Keyframe**:')
            for item in st.session_state.frame_caption_list:
                if item["segment_id"] == segment['id']:
                    st.image(item["image_path"], caption=f"**{item['frame_time']}**: {item['caption']}")
    
    
    frame_caption_df = pd.DataFrame(st.session_state.frame_caption_list)
    
    # Analyze video frames and tasks
    if st.button('Analyze Video Frames'):
        postprocessed_outputs = {}
        # Group captions by segment_id
        segments = {}
        
        for item in st.session_state.frame_caption_list:
            segment_id = item["segment_id"]
            if segment_id not in segments:
                segments[segment_id] = []
            segments[segment_id].append({"caption": item["caption"], "frame_time": item["frame_time"]})
        
        for segment_id, captions_with_times in segments.items():
            postprocessed_outputs[segment_id] = postprocess_by_gpt(str(captions_with_times))

        st.session_state.postprocessed_outputs = postprocessed_outputs  # Save to session state
        
# Display the task list JSON if available
if st.session_state.postprocessed_outputs:
    for segment_id, postprocessed_output in st.session_state.postprocessed_outputs.items():
        st.subheader(f"Postprocessed for Segment {segment_id}:")
        st.markdown(f"### GPT Output\n{postprocessed_output}")