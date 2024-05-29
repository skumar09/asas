# ASAS
The AI Sports Analytics System (ASAS) integrates YOLOv8’s rapid object detection and Chat GPT-4’s generative capabilities into a cohesive package that extracts meaningful insights from sports videos. ASAS’s underlying computer vision core is a YOLOv8-xl model fine-tuned on a custom dataset of annotated frames from brief highlight videos sourced from 30 NBA games during the 2022-2023 season. The model performs object detection across four object classes: player, basketball, net, and referee. It uses geometric heuristics and analysis to detect scoring instances, plot ball trajectories, identify interesting frames, and calculate ball possession by team jersey color. ASAS also implements a natural language processing component that utilizes Chat GPT-4’s vision-to-text and text-to-speech models to  add color commentary to video highlights. With object detection and generative AI components, ASAS enhances traditional sports analytics by synergizing statistics with readily explainable visual analysis. Its fine-tuned object detection model handily outperforms YOLOv8’s suite of pre-trained models with a 0.962 mean average precision (mAP50) across all four classes. Through this performance, ASAS provides a useful foundation for future work in sports analytics, offering a development scheme for systems that can generate analytics, influence strategic decision-making, and enhance sports viewership.

Follow through this README to run this model in your environment.

# Environment setup
In this README, the steps are given to run ASAS in Google Colab.
## Clone the repo
To run this model in a colab or an IDE environment, one should clone this repo. For eg, to run in Google Colab, use the following commands.

```
!git clone https://github.com/skumar09/asas.git
```
## Install Dependencies
Run the command below.

```
# Navigate to ASAS root and install the requirements
%cd /content/asas
!pip install -r requirements.txt
%cd /content/asas
```

# Running the Model
Once all the necessary packages are installed and the environment is setup, the ASAS analytics engine could be run to analyze the ball possession, Trajectory or Field Goal Detection.

## Ball Possession
Ball possession analysis focuses on determining the duration each team controls the ball during gameplay. This feature provides insights into team performance and ball control by analyzing the possession time for each team and plotting the results. To execute this analysis, you can use the run_analytics method of the ASAS model with analysis_type='BALL_POSSESSION'. The result includes a method to plot the ball possession time by team and provides information used for the ball trajectory module. Below is a code snippet demonstrating how to perform this analysis:

```
asas_analytics = asas_model.run_analytics(video_url, local_save_path, analysis_type='BALL_POSSESSION', display_video=True)
asas_analytics.ball_possession_result.plot_ball_possession_time()
```

## Ball Trajectory
This is the process of analyzing the trajectory of the ball on the court during play. This feature internally uses the results of Ball Possession & FGM modules. To run this, ASAS model's `run_analytics` method is called with `analysis_type='BALL_TRAJECTORY'`. The result of this call has a convenient method to plot the trajectory of the ball. Find the snippet below.

```
asas_analytics = asas_model.run_analytics(video_url, local_save_path, analysis_type='BALL_TRAJECTORY') 
asas_analytics.ball_trajectory_result.plot_ball_and_net_wrto_net_position()
```

## NLP and Commentary Generation
The ASAS model also includes an NLP component for generating live commentary based on the game video. This feature uses a pre-trained language model to create descriptive and engaging commentary for the game. The generated text is then converted into audio commentary.

To execute this functionality, use the generate_commentary method of the CommentaryGenerator class. Below is a code snippet demonstrating how to generate commentary for a basketball game video.

```
from analytics_engine.nlp import CommentaryGenerator

# List of different commentary texts
commentary_texts = [
    "Describe the current play in detail, focusing on the key actions and movements of the players. Avoid using terms like 'frame' or 'snapshot' as this is for live sport commentary. Ensure the description is concise and can be spoken within 12 seconds.",
    "Narrate the ongoing action on the court, highlighting the players' movements and the ball's trajectory. Keep the description short enough for live commentary, not exceeding 12 seconds when spoken.",
    "Provide a play-by-play description of the current action, focusing on player interactions and key moments. Make sure the commentary is brief and suitable for live broadcast, under 12 seconds."
]

# Select a specific commentary text from the list
selected_commentary_text = commentary_texts[1]  # Change the index to select different commentary text

generator = CommentaryGenerator()
generator.generate_commentary(
    video_path='/content/tracked_output_videos/test1.mp4',
    output_text_dir='/content/tracked_output_videos/texts',
    output_audio_dir='/content/tracked_output_videos/audio',
    api_key='YOUR_OPENAI_API_KEY',
    commentary_text=selected_commentary_text
)


```

# Demo
A demo of all the above features are available in the `ASAS_Analytics_Demo.ipynb` file under root.