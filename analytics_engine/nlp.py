import os
import cv2
import base64
import requests
import numpy as np
from time import sleep
from IPython.display import Audio, display

class CommentaryGenerator:
    def __init__(self, target_width=768, target_height=768):
        self.target_width = target_width
        self.target_height = target_height

    def resize_base64_images(self, base64_images):
        resized_images = []
        for base64_image in base64_images:
            image_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (self.target_width, self.target_height))
            _, buffer = cv2.imencode('.jpg', resized_img)
            resized_base64_image = base64.b64encode(buffer).decode('utf-8')
            resized_images.append(resized_base64_image)
        return resized_images

    def generate_commentary(self, video_path, output_text_dir, output_audio_dir, api_key, commentary_text, model="gpt-4-vision-preview", audio_model="tts-1-1106", voice="nova", frame_interval=10, max_frames=40):
        print("Opening video file:", video_path)
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print("Error: Could not open video file.")
            return
        
        frame_index = 0
        os.makedirs(output_text_dir, exist_ok=True)
        os.makedirs(output_audio_dir, exist_ok=True)
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                print("Error: Could not read frame.")
                break

            if frame_index % frame_interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                resized_frame = self.resize_base64_images([frame_base64])[0]

                prompt_messages = [{"role": "user", "content": [commentary_text, {"image": resized_frame}]}]
                print("Sending request to OpenAI API for frame index:", frame_index)

                result = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": prompt_messages, "max_tokens": 200}
                ).json()
                
                if 'choices' not in result or not result['choices']:
                    print("Error: Invalid response from OpenAI API:", result)
                    break

                description = result['choices'][0]['message']['content']
                print("Generated description:", description)

                text_path = os.path.join(output_text_dir, f'frame_{frame_index}.txt')
                with open(text_path, 'w') as text_file:
                    text_file.write(description)

                print("Requesting audio generation for description.")
                response = requests.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": audio_model, "input": description, "voice": voice}
                )
                if response.status_code == 200:
                    audio_path = os.path.join(output_audio_dir, f'frame_{frame_index}.wav')
                    with open(audio_path, 'wb') as audio_file:
                        audio_file.write(response.content)
                    display(Audio(audio_path, autoplay=True))
                else:
                    print("Failed to generate audio:", response.text)

            frame_index += frame_interval
            if frame_index >= max_frames:
                break

            sleep(2)

        video.release()
        print("Video processing completed.")
