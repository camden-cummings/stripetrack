#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:55:21 2025

@author: chamomile
"""
import math

import cv2
import numpy as np
import dearpygui.dearpygui as dpg


def generate_rois(lines, frame_height, frame_width, shift):
    """Creates ROIs from set of lines."""
    formatted_lines = []

    # four corners
    corners = [[0, 0],
               [0, int(frame_height)],
               [int(frame_width), 0],
               [int(frame_width), int(frame_height)]]

    for c1 in corners:
        for c2 in corners:
            # all possible borders of screen
            if c1 != c2 and (c1[0] == c2[0] or c1[1] == c2[1]) and [c2, c1] not in formatted_lines:
                formatted_lines.append([tuple(c1), tuple(c2)])

    for line in lines:
        config_dict = dpg.get_item_configuration(line)
        point1 = [int(i) for i in config_dict["p1"]]
        point2 = [int(i) for i in config_dict["p2"]]
        point1[0] -= shift[0]
        point1[1] -= shift[1]
        point2[0] -= shift[0]
        point2[1] -= shift[1]
        formatted_lines.append([tuple(point1), tuple(point2)])

    img = np.zeros((int(frame_height),
                   int(frame_width), 3), dtype=np.uint8)

    for line in formatted_lines:
        point_1, point_2 = line
        cv2.line(img, point_1, point_2, (255, 255, 255), 3)

    contours, _ = cv2.findContours(
        np.array(img)[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shortened_contours = trim(contours, frame_width, frame_height)

    return shortened_contours


def trim(contours, frame_width, frame_height):
    """Trims size of each contour."""
    shortened_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_height*frame_width*0.995:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

            M = cv2.moments(contour)

            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])

            single = []

            for point in approx:
                exp_cx = point[0][0] + 1 * \
                    (centroid_x - point[0][0]) / \
                    math.dist(point[0], (centroid_x, centroid_y))
                exp_cy = point[0][1] + 1 * \
                    (centroid_y - point[0][1]) / \
                    math.dist(point[0], (centroid_x, centroid_y))
                single.append((exp_cx, exp_cy))

            shortened_contours.append(np.array(single))

    return shortened_contours
