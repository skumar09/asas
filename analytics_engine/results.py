import cv2
from matplotlib import pyplot as plt

from utils.analytics_utils import get_bounding_box_corners


class BallPossessionResult:
    def __init__(self, tpr, team_possession, team_possession_time, bp_frame_color, ball_in_air_frames, team_a_color, team_b_color):
        self.tpr = tpr
        self.team_possession = team_possession
        self.team_possession_time = team_possession_time
        self.bp_frame_color = bp_frame_color
        self.ball_in_air_frames = ball_in_air_frames
        self.TEAM_A_COLOR = team_a_color
        self.TEAM_B_COLOR = team_b_color

    def plot_ball_possession_time(self):
        teams = list(self.team_possession_time.keys())
        times = list(self.team_possession_time.values())
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.set_facecolor('gray')
        team_a_color_normalized = tuple(c / 255 for c in self.TEAM_A_COLOR)
        team_b_color_normalized = tuple(c / 255 for c in self.TEAM_B_COLOR)
        bars = plt.bar(teams, times, color=[team_a_color_normalized , team_b_color_normalized])
        plt.xlabel('Teams')
        plt.ylabel('Possession Time (seconds)')
        plt.title('Ball Possession Time by Team')
        plt.ylim(0, max(times) + 5)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
        plt.show()


class BallTrajectoryResult:
    def __init__(self, offset_ball_centroid_tracker, offset_net_centroid_tracker, team_possession,
                 reference_frame_img, reference_net_xywh):
        self.offset_ball_centroid_tracker = offset_ball_centroid_tracker
        self.offset_net_centroid_tracker = offset_net_centroid_tracker
        # for team in team_possession:
        self.team_possession = team_possession
        self.reference_frame_img = reference_frame_img
        self.reference_net_xywh = reference_net_xywh

    def plot_ball_and_net_wrto_net_position(self):
        plt.figure(figsize=(30, 12))

        image_rgb = cv2.cvtColor(self.reference_frame_img, cv2.COLOR_BGR2RGB)
        # image_rgb = np.resize(image_rgb, (int(width), int(height), image_rgb.shape[2]))
        # print('extent -> ',extent)
        print('image_rgb.shape', image_rgb.shape)
        plt.imshow(image_rgb, alpha=0.35)

        # Plot the bounding box of first net
        first_net_corners_x, first_net_corners_y = get_bounding_box_corners(self.reference_net_xywh[0], self.reference_net_xywh[1], self.reference_net_xywh[2], self.reference_net_xywh[3])
        plt.plot(first_net_corners_x, first_net_corners_y, color='red', linewidth=2)

        # Plot the position of ball on the field
        x_ball = [centroid[0] for centroid in self.offset_ball_centroid_tracker]
        y_ball = [centroid[1] for centroid in self.offset_ball_centroid_tracker]
        plt.scatter(x_ball, y_ball, marker='o', color=self.team_possession)

        # Plot the position of net on the field
        x_net = [centroid[0] for centroid in self.offset_net_centroid_tracker]
        y_net = [centroid[1] for centroid in self.offset_net_centroid_tracker]
        plt.plot(x_net, y_net, marker='v', color="red")

        # Plotting arrows between points
        for i in range(len(x_ball) - 1):
            plt.arrow(x_ball[i], y_ball[i], x_ball[i + 1] - x_ball[i], y_ball[i + 1] - y_ball[i],
                      shape='full', lw=0.5, length_includes_head=True, head_width=7.5, color=self.team_possession[i])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Ball movement during play')

    # def plot_as_is(self, plt):
    #     global x_ball, y_ball, x_net, y_net, last_x_net, last_y_net, i
    #
    #     plt.figure(figsize=(12, 30))
    #     # Plot the bounding box of first net
    #     plt.plot(first_net_corners_x, first_net_corners_y, color='red', linewidth=2)
    #     # Plot the position of ball on the field
    #     x_ball = [centroid[0] for centroid in ball_centroid_tracker if len(centroid) == 4]
    #     y_ball = [centroid[1] for centroid in ball_centroid_tracker if len(centroid) == 4]
    #     plt.scatter(x_ball, y_ball, marker='o', color="orange")
    #     # Plot the position of net on the field
    #     x_net = [centroid[0] for centroid in net_centroid_tracker if len(centroid) == 4]
    #     y_net = [centroid[1] for centroid in net_centroid_tracker if len(centroid) == 4]
    #     plt.plot(x_net, y_net, marker='v', color="red")
    #     last_x_net = x_net[-1]
    #     last_y_net = y_net[-1]
    #     plt.annotate(f'({first_x_center_net}, {first_y_center_net})', (first_x_center_net, first_y_center_net),
    #                  textcoords="offset points", xytext=(0, 10), ha='center')
    #     plt.annotate(f'({last_x_net}, {last_y_net})', (last_x_net, last_y_net), textcoords="offset points", xytext=(0, 10),
    #                  ha='center')
    #     # for (x, y) in zip(x_net, y_net):
    #     #     plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    #     # Plotting arrows between points
    #     for i in range(len(x_ball) - 1):
    #         plt.arrow(x_ball[i], y_ball[i], x_ball[i + 1] - x_ball[i], y_ball[i + 1] - y_ball[i],
    #                   shape='full', lw=0.1, length_includes_head=True, head_width=0.0075, color='green')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.title('Ball & Net position')


class FieldGoalDetectionResult:
    # fgm = Field Goal Made = A successful basketball shot
    def __init__(self, probability_fgm, interesting_frame_indices, field_goal_frame):
        self.probability_fgm = probability_fgm
        self.interesting_frame_indices = interesting_frame_indices
        self.fgm_frame = field_goal_frame


class ASASAnalytics:
    def __init__(self):
        self.yolo_tracking_results = None
        self.ball_possession_result = None
        self.ball_trajectory_result = None
        self.field_goal_detection_result = None
