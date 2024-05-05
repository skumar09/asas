from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from analytics_engine.results import BallPossessionResult, BallTrajectoryResult
from config import colors


class BallPossessionAnalyzer:
    def __init__(self, team_a_color='black', team_b_color='white', ball_in_air_threshold=50, frame_rate=30, confidence=0.5):
        self.TEAM_A_COLOR = colors.COLOR_MAP[team_a_color]
        self.TEAM_B_COLOR = colors.COLOR_MAP[team_b_color]
        self.ball_in_air_threshold = ball_in_air_threshold
        self.frame_rate = frame_rate
        self.confidence = confidence

    # Function to filter based on most frequently appearing IDs
    @staticmethod
    def filter_ids(ids, count):
        id_frequency = Counter(ids)
        most_common_ids = [id for id, freq in id_frequency.most_common(count)]
        return most_common_ids

    def process_tracking_data(self, track_results):
        unique_player_ids = set()
        unique_referee_ids = set()
        unique_net_ids = set()
        basketball_id = None
        basketball_frame_count = 0
        frames_with_basketball = []

        for frame_idx, result in enumerate(track_results):
            basketball_detected = False  # Flag to check if basketball is detected in the current frame
            if result.boxes is not None and hasattr(result.boxes, 'xyxy'):
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
                confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
                class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []

                for box, conf, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                    class_name = result.names.get(int(class_id), 'Unknown')

                    if conf > self.confidence:
                        if class_name == 'basketball':
                            basketball_id = int(track_id)
                            basketball_detected = True # Set the flag to True as basketball is detected in this frame
                        elif class_name == 'player':
                            unique_player_ids.add(int(track_id))
                        elif class_name == 'referee':
                            unique_referee_ids.add(int(track_id))
                        elif class_name == 'net':
                            unique_net_ids.add(int(track_id))

            if basketball_detected:
                basketball_frame_count += 1 # Increment the counter if basketball was detected in the frame
                frames_with_basketball.append(frame_idx) # Add the frame index to the list

        # Filter IDs if needed using some filter_ids function
        # Example function needs to be defined to actually filter IDs
        filtered_player_ids = self.filter_ids(unique_player_ids, 10)  # Assuming up to 10 players
        filtered_referee_ids = self.filter_ids(unique_referee_ids, 3)  # Assuming up to 3 referees
        filtered_net_ids = self.filter_ids(unique_net_ids, 2)  # Assuming up to 2 nets
        return {
            'basketball_id': basketball_id,
            'player_ids': filtered_player_ids,
            'referee_ids': filtered_referee_ids,
            'net_ids': filtered_net_ids,
            'basketball_frame_count': basketball_frame_count,
            'frames_with_basketball': frames_with_basketball
        }

    @staticmethod
    def crop_jersey_from_player(image, bbox):
        # Calculate coordinates to crop the jersey region
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        # Chek if width and height are positive
        if width <= 0 or height <= 0:
            raise ValueError("Invalid bounding box dimensions")

        # Estimate the jersey area within the bounding box, avoiding the head/neck and leg region
        jersey_top = max(y1 + int(height * 0.3), 0) # Start a bit below the top to avoid the player's neck and head
        jersey_bottom = min(y1 + int(height * 0.7), image.shape[0]) # End before the shorts

        # Add a slight margin to the left and right to avoid the arms
        jersey_left = max(x1 - int(width * 0.1), 0)
        jersey_right = min(x2 + int(width * 0.1), image.shape[1])

        # Crop the jersey region
        jersey_img = image[jersey_top:jersey_bottom, jersey_left:jersey_right]

        if jersey_img.size == 0:
            raise ValueError("Cropped jersey image is empty or the dimensions are invalid.")

        return jersey_img

    @staticmethod
    def find_dominant_color(image, mask=None):
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)

        pixels = np.float32(image.reshape(-1, 3))

        if pixels.shape[0] == 0:
            return np.array([0, 0, 0])  # Return a default color if the image is empty

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
        _, labels, palette = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dominant_index = np.argmax(counts)
        dominant_color = palette[dominant_index]

        return dominant_color

    @staticmethod
    def name_dominant_color(dominant_color):
        # convert the RGB to names
        color_name = ""
        min_color_distance = float('inf')

        for color, rgb_values in colors.COLOR_MAP.items():
            color_distance = np.linalg.norm(dominant_color - np.array(rgb_values))
            if color_distance < min_color_distance:
                min_color_distance = color_distance
                color_name = color

        return color_name

    @staticmethod
    def classify_team_by_color(dominant_color, team_a_color=None, team_b_color=None):
        #print(f'dominant color {dominant_color}')
        if isinstance(dominant_color, str):
            dominant_color = colors.COLOR_MAP[dominant_color]
        team_a_dist = np.linalg.norm(np.array(dominant_color) - np.array(team_a_color))
        team_b_dist = np.linalg.norm(np.array(dominant_color) - np.array(team_b_color))

        if team_a_dist < team_b_dist:
            return 'Team_A'
        else:
            return 'Team_B'

    def analyze(self, yolo_tracking_results):
        tpr = self.process_tracking_data(yolo_tracking_results)
        frames_with_basketball = tpr['frames_with_basketball']

        team_possession = {'Team_A': 0, 'Team_B': 0}
        bp_frame_color = {}
        ball_in_air_frames  = []

        # Loop through each frame where basketball is present
        for frame_id in frames_with_basketball:
            result = yolo_tracking_results[frame_id]
            orig_img = result.orig_img  # Accessing the results for the current frame

            # Extract the players and basketball detections
            players = [res for res in result.summary() if res['name'] == 'player']
            basketballs = [res for res in result.summary() if res['name'] == 'basketball']

            # Continue processing only if players and basketball are detected
            if basketballs and players:
                basketball = basketballs[0]
                ball_pos = np.array([(basketball['box']['x1'] + basketball['box']['x2']) / 2,
                                     (basketball['box']['y1'] + basketball['box']['y2']) / 2])
                min_distance = float('inf')
                player_with_ball = None

                # Find the closest player to the ball
                for player in players:
                    player_bbox = player['box']
                    player_pos = np.array([(player_bbox['x1'] + player_bbox['x2']) / 2,
                                           (player_bbox['y1'] + player_bbox['y2']) / 2])
                    distance = np.linalg.norm(ball_pos - player_pos)
                    if distance < min_distance:
                        min_distance = distance
                        player_with_ball = player

                # Determine if the ball is in the air based on the distance threshold
                if min_distance > self.ball_in_air_threshold or not players:
                    if frame_id not in ball_in_air_frames:
                        ball_in_air_frames.append(frame_id)
                        #bp_frame_color[frame_id] = 'air'
                        continue
                #print(f"Minimum distance : {min_distance}")
                # If a player with the ball was found, process their jersey
                if player_with_ball:
                    bbox = player_with_ball['box']
                    player_bbox = [int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])]
                    player_img = orig_img[player_bbox[1]:player_bbox[3], player_bbox[0]:player_bbox[2]]
                    #cv2_imshow(player_img)  # Show the cropped player image
                    #cv2.imwrite(f'/content/original_bb_images/player_bb_img_{frame_id}.jpg', player_img)  # Save the cropped player image
                    jersey_img = self.crop_jersey_from_player(orig_img, player_bbox)  # Assuming this function is defined

                    if jersey_img is not None:
                        dominant_color = self.find_dominant_color(jersey_img)
                        color_name = self.name_dominant_color(dominant_color)
                        bp_frame_color[frame_id] = color_name

                        # Update the team possession count
                        team = self.classify_team_by_color(color_name, self.TEAM_A_COLOR, self.TEAM_B_COLOR)
                        team_possession[team] += 1
                        #print(f"Frame {frame_id}: Team possession: {team}")
                        #cv2_imshow(player_img)  # Show the cropped player image
                        #cv2.imwrite(f'/content/jersey_bb_images/jersey_bb_img_{frame_id}.jpg', jersey_img)  # Save the cropped player image
                    else:
                        print(f"Frame {frame_id}: No jersey found")

                else:
                    print(f"Frame {frame_id}: No player with the ball found")

        team_possession_time = {team: frames / self.frame_rate for team, frames in team_possession.items()}

        return BallPossessionResult(tpr, team_possession, team_possession_time, bp_frame_color, ball_in_air_frames, self.TEAM_A_COLOR, self.TEAM_B_COLOR)


class BallTrajectoryAnalyzer:
    def __init__(self, yolo_tracking_results, bp_frame_color, reference_frame_index=0):
        self.yolo_tracking_results = yolo_tracking_results
        self.bp_frame_color = bp_frame_color
        self.reference_frame_index = reference_frame_index

    def analyze(self):
        ball_centroid_tracker = []
        net_centroid_tracker = []

        orig_h_w_of_image = self.yolo_tracking_results[0].orig_shape
        print('orig_h_w_of_image -> ', orig_h_w_of_image)
        for test_result in self.yolo_tracking_results:
            classes_tensor = test_result.boxes.cls
            ball_index = ball_index.item() if (ball_index := torch.where(classes_tensor == 0)[0]).numel() == 1 else None
            net_index = net_index.item() if (net_index := torch.where(classes_tensor == 1)[0]).numel() == 1 else None

            ball_xywh = test_result.boxes.xywh[ball_index].numpy() if ball_index else np.empty((0,))
            net_xywh = test_result.boxes.xywh[net_index].numpy() if net_index else np.empty((0,))

            # self.mirror_points_from_quad1_to_quad4(ball_xywh, net_xywh, orig_h_w_of_image)

            ball_centroid_tracker.append(ball_xywh)
            net_centroid_tracker.append(net_xywh)

        # # Transform to 4th quadrant to match YOLOv8 format. Works only if the points are normalized coords
        # ball_centroid_tracker, net_centroid_tracker = self.mirror_points_from_quad1_to_quad4_normalized(
        #     ball_centroid_tracker, net_centroid_tracker)

        offset_ball_centroid_tracker, offset_net_centroid_tracker, team_possession = self.offset_centroids_to_reference_net_centroid(ball_centroid_tracker, net_centroid_tracker)
        reference_frame_result = self.yolo_tracking_results[self.reference_frame_index]

        return BallTrajectoryResult(offset_ball_centroid_tracker, offset_net_centroid_tracker, team_possession, reference_frame_result.orig_img, net_centroid_tracker[self.reference_frame_index])

    def offset_centroids_to_reference_net_centroid(self, ball_centroid_tracker, net_centroid_tracker):
        offset_ball_centroid_tracker = []
        offset_net_centroid_tracker = []

        x_center_reference_net = net_centroid_tracker[self.reference_frame_index][0]
        y_center_reference_net = net_centroid_tracker[self.reference_frame_index][1]

        print('self.bp_frame_color->', self.bp_frame_color)

        index = 0
        team_possession = []
        for net_centroid in net_centroid_tracker:
            if len(net_centroid) == 4:
                net_offset_x = net_centroid[0] - x_center_reference_net
                net_offset_y = net_centroid[1] - y_center_reference_net
                offset_net_centroid_tracker.append(
                    ((net_centroid[0] - net_offset_x), (net_centroid[1] - net_offset_y)))
                if len((ball_centroid := ball_centroid_tracker[index])) == 4:
                    offset_ball_centroid_tracker.append(
                        ((ball_centroid[0] - net_offset_x), (ball_centroid[1] - net_offset_y)))
                    team_possession.append(self.bp_frame_color.get(index, 'green'))
            index += 1
        return offset_ball_centroid_tracker, offset_net_centroid_tracker, team_possession