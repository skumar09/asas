import os
import subprocess
from base64 import b64encode

import requests
from IPython.display import display
from IPython.display import HTML


def display_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    mp4 = open(video_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    video_html = HTML("""
    <video width=1000 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)
    display(video_html)
    return data_url


def download_video(video_url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(video_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        print("Failed to download video from:", video_url)


def convert_video_to_mp4(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    output_directory = os.path.dirname(output_path)
    os.makedirs(output_directory, exist_ok=True)

    command = ['ffmpeg', '-i', input_path, output_path]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video conversion to .mp4 format is successful: {input_path} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        error_details = e.stderr.decode()
        print(f"Error during video conversion: {error_details}")
        return False
