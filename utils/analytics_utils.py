def get_bounding_box_corners(x_center, y_center, width, height):
    first_net_corners_x = [(x_center - (width / 2)),
                           (x_center + (width / 2)),
                           (x_center + (width / 2)),
                           (x_center - (width / 2)),
                           (x_center - (width / 2))]
    first_net_corners_y = [(y_center + (height / 2)),
                           (y_center + (height / 2)),
                           (y_center - (height / 2)),
                           (y_center - (height / 2)),
                           (y_center + (height / 2))]

    return first_net_corners_x, first_net_corners_y


def mirror_points_from_quad1_to_quad4_normalized(ball_centroid_tracker, net_centroid_tracker):
    ball_centroid_tracker = [(centroid[0], 1 - centroid[1]) if centroid else None for centroid in
                             ball_centroid_tracker]
    net_centroid_tracker = [(centroid[0], 1 - centroid[1]) if centroid else None for centroid in
                            net_centroid_tracker]
    return ball_centroid_tracker, net_centroid_tracker


def mirror_points_from_quad1_to_quad4(ball_xywh, net_xywh, orig_h_w_of_image):
    # Offsetting Y axis to get 1st quadrant as the YOLOv8 detection is in 4th Quadrant by default.
    # Done by Y coordinate if the point by the height of the image.
    if ball_xywh.size > 0:
        ball_xywh[1] = orig_h_w_of_image[0] - ball_xywh[1]
        # ball_xywh[1] = - ball_xywh[1]
    if net_xywh.size > 0:
        net_xywh[1] = orig_h_w_of_image[0] - net_xywh[1]
        # net_xywh[1] = - net_xywh[1]
