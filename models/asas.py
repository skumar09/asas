# Imports
import os
import sys

import torch
from ultralytics import YOLO

from analytics_engine.analyzer import BallPossessionAnalyzer, BallTrajectoryAnalyzer
from analytics_engine.results import ASASAnalytics
from config import colors
from utils import media_utils


class ASAS(YOLO):

    def __init__(self, asas_model_path, fps=30, team_a_color='black', team_b_color='white', ball_in_air_threshold=50, device='cpu'):
        """
        Initialize the ASAS model.

        Parameters:
        asas_model_path (str): Path to the trained model file.
        fps (int): Frames per second of the video to be analyzed.
        team_a_color (str): Color representing team A.
        team_b_color (str): Color representing team B.
        :param ball_in_air_threshold:
        """

        # self.model = YOLO(asas_model_path)
        super().__init__(asas_model_path)
        self.to(device)  # Move model to the appropriate device
        #self.to(torch.device(device))
        self.frame_rate = fps
        self.TEAM_A_COLOR = colors.COLOR_MAP[team_a_color]
        self.TEAM_B_COLOR = colors.COLOR_MAP[team_b_color]

        # Initialize dataset attribute
        self.dataset_directory = None

        # Initialize dataset.yaml path attribute
        self.dataset_yaml_path = None

        # Initialize dataset.yaml config attribute
        self.dataset_yaml_config = None

        # Default distance threshold in pixels
        self.ball_in_air_threshold = 150

        self.yolo_tracking_results = None

    
    @staticmethod
    def get_device():
        """Determine the best device to run the model on."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch, 'has_mps') and torch.has_mps:  # Check if the attribute exists and is True
            return 'mps'
        return 'cpu'

    def track_video(self, video_path, tracker_config='bytetrack.yaml', persist=True, save=True, conf=0.5, iou=0.7,
                    show=False):
        print(f"\n----------Started: Running ASAS Tracking ----------\n")
        print(f"Tracking video: {video_path}")
        print(f"Using tracker config: {tracker_config}")
        print(f"Detection confidence threshold: {conf}, IOU threshold: {iou}")
        print(f"Persist results: {'Yes'  if persist else 'No'}, Save output: {'Yes' if save else 'No'}, Show results: {'Yes' if show else 'No'}\n")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            # Run the tracking model on the video
            results = self.track(
                video_path,
                tracker=tracker_config,
                persist=persist,
                save=save,
                conf=conf,
                iou=iou,
                show=show,
                verbose=False  # Hypothetical parameter to control verbosity
            )
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"\n----------Completed: Running ASAS Tracking ----------\n")
        return results

    def welcome_message(self):
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│                                                                 │")
        print("│             Welcome to AI Sports Analytics Systems (ASAS)       │")
        print("│                                                                 │")
        print("│                         By                                      │")
        print("│                         Devansh Vaid                            │")
        print("│                         George Valan                            │")
        print("│                         Kum Gaon                                │")
        print("│                         Santoshkumar Nateekar                   │")
        print("│                                                                 │")
        print("│                 Advisor: Prof. Dr. Sanja Damjanovic             │")
        print("│                                                                 │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print("\n")



    def run_analytics(self, video_url, local_save_path, analysis_type, display_video=False):

        # Initialize ASASAnalytics to hold the result of analytics
        asas_analytics = ASASAnalytics()

        self.welcome_message()

        # Download the input video clip
        media_utils.download_video(video_url, local_save_path)

        # Display original video
        if display_video:
            media_utils.display_video(local_save_path)

        self.yolo_tracking_results = self.yolo_tracking_results if self.yolo_tracking_results else self.track_video(local_save_path)
        asas_analytics.yolo_tracking_results = self.yolo_tracking_results

        # Extract the base filename without extension to construct output paths
        base_filename = os.path.splitext(os.path.basename(local_save_path))[0]
        avi_path = f'runs/detect/track/{base_filename}.avi'
        mp4_output_path = f'/content/tracked_output_videos/{base_filename}.mp4'
        print(f"avi_path: {avi_path}")
        print(f"mp4_output_path: {mp4_output_path}")

        # Display annotated video
        if display_video:
            if not os.path.exists(mp4_output_path):  # Check if MP4 already exists to avoid re-conversion
                media_utils.convert_video_to_mp4(avi_path, mp4_output_path)
            media_utils.display_video(mp4_output_path)

        if analysis_type == 'BALL_POSSESSION':
            # Analyse Ball Possession
            ball_possession_result = self.analyze_ball_possession(self.yolo_tracking_results)
            tpr = ball_possession_result.tpr
            team_possession = ball_possession_result.team_possession
            bp_frame_color = ball_possession_result.bp_frame_color

            print(f'\n----------VIDEO ANALYSIS-------------\n')
            print(f"1. Unique basketball ID: {tpr['basketball_id']}")
            print(f"2. Unique player IDs (filtered): {tpr['player_ids']}")
            print(f"3. Number of unique players (filtered): {len(tpr['player_ids'])}")
            print(f"4. Unique referee IDs (filtered): {tpr['referee_ids']}")
            print(f"5. Number of unique referees (filtered): {len(tpr['referee_ids'])}")
            print(f"6. Unique net IDs (filtered): {tpr['net_ids']}")
            print(f"7. Number of unique nets (filtered): {len(tpr['net_ids'])}")
            print(f"8. Number of Basketball Frames (filtered): {tpr['basketball_frame_count']}")
            print(f"9. Frames with Basketball Detected: {tpr['frames_with_basketball']}")

            print(f'\n----------BALL POSSESSION-------------\n')
            print(f'Team A possession frames: {team_possession["Team_A"]}')
            print(f'Team B possession frames: {team_possession["Team_B"]}')
            print(f'Team A possession time: {team_possession["Team_A"]}')
            print(f'Team B possession time: {team_possession["Team_B"]}')
            print(f'Player with ball: {bp_frame_color}')

            asas_analytics.ball_possession_result = ball_possession_result

        if analysis_type == 'BALL_TRAJECTORY':
            ball_possession_result = ball_possession_result if (ball_possession_result:=asas_analytics.ball_possession_result) else self.analyze_ball_possession(self.yolo_tracking_results)
            ball_trajectory_result = self.analyze_ball_trajectory(self.yolo_tracking_results, ball_possession_result.bp_frame_color)

            asas_analytics.ball_trajectory_result = ball_trajectory_result

        return asas_analytics

    def analyze_ball_possession(self, yolo_tracking_results):
        ball_possession_analyzer = BallPossessionAnalyzer(team_a_color='black', team_b_color='white',
                                                          ball_in_air_threshold=self.ball_in_air_threshold,
                                                          frame_rate=self.frame_rate, confidence=0.5)
        ball_possession_result = ball_possession_analyzer.analyze(yolo_tracking_results)
        return ball_possession_result

    def analyze_ball_trajectory(self, yolo_tracking_results, bp_frame_color):
        ball_trajectory_analyzer = BallTrajectoryAnalyzer(yolo_tracking_results, bp_frame_color, 313)
        ball_trajectory_result = ball_trajectory_analyzer.analyze()
        return ball_trajectory_result
