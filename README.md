# ASAS
The AI Sports Analytics System (ASAS) integrates YOLOv8’s rapid object detection and Chat GPT-4’s generative capabilities into a cohesive package that extracts meaningful insights from sports videos. ASAS’s underlying computer vision core is a YOLOv8-xl model fine-tuned on a custom dataset of annotated frames from brief highlight videos sourced from 30 NBA games during the 2022-2023 season. The model performs object detection across four object classes: player, basketball, net, and referee. It uses geometric heuristics and analysis to detect scoring instances, plot ball trajectories, identify interesting frames, and calculate ball possession by team jersey color. ASAS also implements a natural language processing component that utilizes Chat GPT-4’s vision-to-text and text-to-speech models to  add color commentary to video highlights. With object detection and generative AI components, ASAS enhances traditional sports analytics by synergizing statistics with readily explainable visual analysis. Its fine-tuned object detection model handily outperforms YOLOv8’s suite of pre-trained models with a 0.962 mean average precision (mAP50) across all four classes. Through this performance, ASAS provides a useful foundation for future work in sports analytics, offering a development scheme for systems that can generate analytics, influence strategic decision-making, and enhance sports viewership.

Follow through this README to run this model in your environment.

# Installation

# Env set-up

# Running
Once all the necessary packages are installed and the environment is setup, the ASAS analytics engine could be run to analyze the ball possession, Trajectory or Field Goal Detection.
## Ball Trajectory
This is the process of analyzing the trajectory of the ball on the court during play. This feature internally uses the results of Ball Possession & FGM modules. To run this, ASAS model's `run_analytics` method is called with `analysis_type='BALL_TRAJECTORY'`. The result of this call has a convenient method to plot the trajectory of the ball. Find the snippet below.

```
asas_analytics = asas_model.run_analytics(video_url, local_save_path, analysis_type='BALL_TRAJECTORY') 
asas_analytics.ball_trajectory_result.plot_ball_and_net_wrto_net_position()
```

# Demo
A demo of all the above features are available in the `ASAS_Analytics_Demo.ipynb` file under root.