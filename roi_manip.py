#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:44:39 2025

@author: chamomile
"""

import pickle
import os
import math

import numpy as np
import cv2

from roi_selector_dearpygui.roi_selector_dearpygui.interfaces.roipoly import RoiPoly

#TODO add auto trimmer that finds dimensions that can trim the image to to still include all of every contour
#TODO dump all of this nonsense into roiinterface

def convert_to_contours(cell_filename, frame_width, frame_height): 
    if isinstance(cell_filename, str):
        with open(cell_filename, 'rb') as f:
            rois = pickle.load(f)

            if isinstance(rois, RoiPoly):
                rois_dup = []
                for roi in rois:
                    rois_dup.append(roi.lines)
                rois = rois_dup
    else:
        rois = cell_filename
        
    centers = []
    contours = []

    for roi in rois:
        contour = np.array(roi, dtype='int')
        contours.append(contour)
        
        cx, cy = find_centroid_of_contour(contour)
        
        for center in centers:
            if math.dist(center[0], (cx,cy)) == 0.0:
                break
        else:
            centers.append([[cx, cy], contour])
    
    centers.sort(key=lambda tup: tup[0][1])
    
    row = 0
    reorg_centers = []
    curr_row = []
    
    #sorting by row  ------------------------
    for i in range(0, len(centers)):
        if centers[i][0][1] - centers[i-1][0][1] > 20:        
            row += 1
            curr_row.sort(key=lambda tup: tup[0][0])
            reorg_centers.append(curr_row.copy())
            curr_row.clear()
            
        curr_row.append(centers[i])
        
        if i == len(centers)-1:
            curr_row.sort(key=lambda tup: tup[0][0])
            reorg_centers.append(curr_row)

    num_rows = len(reorg_centers)
    # ------------------------
    #TODO if not possible try to group by vertical alignment
    true_value = (num_rows, len(reorg_centers[0]))
    vertical = False
    for row in range(num_rows):
        if true_value != (num_rows, len(reorg_centers[row])):
            vertical = True
            break

    if vertical:
        # sorting by col  ------------------------
        col = 0
        reorg_centers = []
        curr_col = []
        vertical = True

        centers.sort(key=lambda tup: tup[0][0])

        for i in range(0, len(centers)):
            if centers[i][0][0] - centers[i - 1][0][0] > 20:
                col += 1
                curr_col.sort(key=lambda tup: tup[0][1])
                reorg_centers.append(curr_col.copy())
                curr_col.clear()

            curr_col.append(centers[i])

            if i == len(centers) - 1:
                curr_col.sort(key=lambda tup: tup[0][1])
                reorg_centers.append(curr_col)

    shape_of_rows = []

    if vertical:
        num_cols = len(reorg_centers)
        num_rows = len(reorg_centers[0])

        cell_contours = [[] for i in range(len(centers))]
        cell_centers = [[] for j in range(num_rows)]

        for row in range(num_rows):
            shape_of_rows.append(num_cols)
            for col in range(num_cols):
                y_count = row * num_cols + col

                cell_centers[row].append(reorg_centers[col][row][0])
                cell_contours[y_count] = reorg_centers[col][row][1]
    else:
        cell_contours = [[] for i in range(len(centers))]
        cell_centers = [[] for j in range(num_rows)]

        for row in range(num_rows):
            num_cols = len(reorg_centers[row])
            shape_of_rows.append(num_cols)
            for col in range(num_cols):
                y_count = row * num_cols + col

                cell_centers[row].append(reorg_centers[row][col][0])
                cell_contours[y_count] = reorg_centers[row][col][1]

    contour_mask = np.zeros((frame_height, frame_width, 3))
    
    for c in cell_contours:
        if len(c) > 0:
            contour_mask = cv2.drawContours(contour_mask, [c],
                                            -1, (255, 255, 255), thickness=cv2.FILLED)
        
    contour_mask = cv2.cvtColor(
        np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

    print("roimanip",cell_centers)
    return cell_contours, contour_mask, cell_centers, shape_of_rows

def get_cell_bounds(cell_contours):
    bounds = []
    for c in cell_contours:
        x, y, w, h = cv2.boundingRect(c)
        bounds.append([x, y, x+w, y+h])
    return bounds

def find_centroid_of_contour(contour):
    """Given a contour, finds centroid of it."""
    M = cv2.moments(contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

if __name__ == "__main__":
    _, _, centers, _ = convert_to_contours("/home/chamomile/fc2_save_2025-07-24-101645-0000.cells", 1760, 1200)
    im = cv2.imread("/home/chamomile/fc2_save_2025-07-24-101645-0000-mode.png")
    #im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for row in centers:
        for c in row:
            cv2.circle(im, c, radius=5, color=(0, 255, 0), thickness=2)

    cv2.imshow('f', im)
    cv2.waitKey(0)