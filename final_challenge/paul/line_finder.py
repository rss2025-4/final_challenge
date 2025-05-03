#!/usr/bin/env python3
"""
lane_detection.py

Standalone script to detect and draw left/right lane boundaries
on a running video or webcam feed, using OpenCV processing:
  - HSV-based white masking with morphology
  - Canny edge detection
  - Trapezoidal ROI cropping
  - Hough segment extraction
  - Robust line fitting & extrapolation via cv2.fitLine

Usage:
  python3 lane_detection.py [--video path/to/video.mp4]
  (omit --video to use webcam index 0)
"""

import cv2
import numpy as np
import argparse


RIGHT_Y_CUTOFF = 0.9


def threshold_white(img):
    """
    Mask white paint: use HSV (low saturation, high value) + morphological cleanup.
    """
    # image_print(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # L, A ,B  = cv2.split(lab)
    # _, mask_L = cv2.threshold(L, 195, 255, cv2.THRESH_BINARY)
    # image_print(hsv)
    # white: S low (<= 30), V high (>= 200)
    lower = np.array([0, 0, 160], dtype=np.uint8)
    upper = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # vert = cv2.getStructuringElement(cv2.MORPH_RECT,(3,35))  # ← NEW
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vert, 1)   # keep rails
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,vert, 1)   # bridge gaps
    # mask = cv2.bitwise_and(mask, mask_L)
    # return mask
    # image_print(mask)
    # vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vert_kernel)
    # # image_print(mask)
    # # # then your existing cleanup
    # kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern2)
    # remove noise / fill gaps
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # image_print(mask)
    return mask


def region_of_interest(edges):
    """
    Keep only the trapezoid region where lane lines appear.
    Adjust top_y to control how far up the lines are estimated.
    """
    h, w = edges.shape
    mask = np.zeros_like(edges)
    top_y = int(0.4 * h) 
    poly = np.array([[
        (int(0.05*w), h),
        (int(0.05*w), top_y),
        (int(0.95*w), top_y),
        (int(0.95*w), h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    new = cv2.bitwise_and(edges, mask)
    return new

LEFT_SLOPE_MIN  = 0.25  
RIGHT_SLOPE_MIN = 0.25   

MIN_SEG_LEN     = 20 
PARALLEL_TOL    = np.deg2rad(8) 

def fit_and_extrapolate(frame, segments):
    """
    Given a list of (x1,y1,x2,y2) segments for one side, fit a single
    line via cv2.fitLine and extrapolate to full ROI height.
    Returns (x1,y1,x2,y2) or None.
    """
    if not segments:
        return None

    pts = np.vstack([
        [[x1, y1] for (x1,y1,x2,y2) in segments] +
        [[x2, y2] for (x1,y1,x2,y2) in segments]
    ]).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    h, _ = frame.shape[:2]
    y1, y2 = h, int(0.4 * h)
    t1 = (y1 - y0) / vy
    t2 = (y2 - y0) / vy
    x1 = int(x0 + t1 * vx)
    x2 = int(x0 + t2 * vx)

    return (x1, y1, x2, y2)

def _seg_len(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

def _dominant_cluster(segs, nbins=25):
    """Return segs whose slope lies in the bin with the largest *total length*."""
    if not segs:
        return []
    slopes = np.array([(y2-y1)/(x2-x1) for x1,y1,x2,y2 in segs])
    lens   = np.array([_seg_len(x1,y1,x2,y2) for x1,y1,x2,y2 in segs])
    hist, bins = np.histogram(slopes, nbins, weights=lens)
    k = hist.argmax()
    lo, hi = bins[k], bins[k+1]
    return [s for s, m in zip(segs, slopes) if lo <= m < hi]

def cluster_and_fit(frame, lines):
    """
    Split Hough segments into left/right, prune by slope *and* segment length,
    fit one line per side, then make sure the two rails are roughly parallel.
    """
    h, w = frame.shape[:2]
    left_segs, right_segs, total = [], [], 0

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            total += 1
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            length = _seg_len(x1, y1, x2, y2)

            if length < MIN_SEG_LEN:  
                continue
            if abs(slope) < 0.23:    
                continue

            mx = (x1 + x2) / 2
            if slope < 0 and mx < w / 2 and abs(slope) >= LEFT_SLOPE_MIN:
                left_segs.append((x1, y1, x2, y2))
            elif slope > 0 and mx > w / 2 and abs(slope) >= RIGHT_SLOPE_MIN:
                if (y1 / h > RIGHT_Y_CUTOFF) or (y2 / h > RIGHT_Y_CUTOFF):
                    right_segs.append((x1, y1, x2, y2))
        if not left_segs:
            for x1,y1,x2,y2 in lines[:,0]:
                if x2==x1: continue
                m  = (y2-y1)/(x2-x1)
                mx = (x1+x2)/2
                if -0.23 < m < -0.08 and mx < w/2 and _seg_len(x1,y1,x2,y2) > MIN_SEG_LEN:
                    left_segs.append((x1,y1,x2,y2))

        if not right_segs:
            for x1,y1,x2,y2 in lines[:,0]:
                if x2==x1: continue
                m  = (y2-y1)/(x2-x1)
                mx = (x1+x2)/2
                if  0.08 < m <  0.23 and mx > w/2 and _seg_len(x1,y1,x2,y2) > MIN_SEG_LEN:
                    right_segs.append((x1,y1,x2,y2))
    left_segs  = _dominant_cluster(left_segs)  
    right_segs = _dominant_cluster(right_segs) 
    left_line  = fit_and_extrapolate(frame, left_segs)
    right_line = fit_and_extrapolate(frame, right_segs)

    # if left_line is not None and right_line is not None:
    #     x1l, y1l, x2l, y2l = left_line
    #     x1r, y1r, x2r, y2r = right_line
    #     theta_l = np.arctan2(y2l - y1l, x2l - x1l)
    #     theta_r = np.arctan2(y2r - y1r, x2r - x1r)
    #     if abs(abs(theta_l) - abs(theta_r)) > PARALLEL_TOL:
    #         # drop the side with fewer segments
    #         if len(left_segs) < len(right_segs):
    #             left_line = None
    #         else:
    #             right_line = None

    return left_segs, right_segs, left_line, right_line

def _extend_to_y(line, y_target):
    """
    Given endpoints (x1,y1,x2,y2) of a rail segment, return
    a new (xA, y_target, xB, yB) so that the line now reaches y_target.
    Keeps the lower end (usually y = frame height) unchanged.
    """
    x1, y1, x2, y2 = line
    if y2 == y1:        
        return line
    t = (y_target - y1) / (y2 - y1)
    x_target = int(x1 + t * (x2 - x1))
    return (x1, y1, x_target, y_target)

def _intersection(p, q):
    """
    p, q are (x1,y1,x2,y2).  Returns (xi, yi) floats or None if parallel.
    """
    x1,y1,x2,y2 = p
    x3,y3,x4,y4 = q
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None             
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)


def auto_canny(img, sigma=0.33):
    v   = np.median(img)
    lo  = int(max(0,  (1.0-sigma)*v))
    hi  = int(min(255,(1.0+sigma)*v))
    return cv2.Canny(img, lo, hi)

def process_frame(frame):
    """Complete pipeline on a single frame with debug overlays."""
    blur  = cv2.GaussianBlur(frame, (5,5), 0)
    mask  = threshold_white(blur)

    ys, xs = np.where(mask > 0)
    # print(f"White pixels: {len(xs)}; sample (x,y): {list(zip(xs,ys))[:5]}")

    edges = cv2.Canny(mask, 50, 150)
    # k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    # edges  = cv2.dilate(edges, k_vert, iterations=1)
    edges = cv2.dilate(edges,
                   cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), 1)
    roi   = region_of_interest(edges)
    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180, 20,
        minLineLength=20, maxLineGap=200
    )
    # Overlay raw Hough segments in red
    debug_img = frame.copy()
    # if lines is not None:
    #     for x1,y1,x2,y2 in lines[:,0]:
    #         cv2.line(debug_img, (x1,y1), (x2,y2), (0,0,255), 1)

    # Cluster, fit, and get segments
    left_segs, right_segs, left, right = cluster_and_fit(frame, lines)
    ROI_TOP_Y = int(0.40 * frame.shape[0]) 
    left_ext  = _extend_to_y(left,  ROI_TOP_Y)  if left  else None
    right_ext = _extend_to_y(right, ROI_TOP_Y)  if right else None
    for ln in (left_ext, right_ext):
        if ln:
            cv2.line(debug_img, (ln[0],ln[1]), (ln[2],ln[3]), (0,255,255), 2)

    vp = _intersection(left_ext, right_ext) if left_ext and right_ext else None
    if vp:
        cv2.circle(debug_img, vp, 6, (0,0,255), -1) 

    # for ln in (left, right):
    #     if ln is not None:
    #         x1,y1,x2,y2 = ln
    #         cv2.line(debug_img, (x1,y1), (x2,y2), (0,255,0), 5)

    return debug_img, vp


def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
if __name__ == "__main__":
    # Quick test on images 1..67
    # for i in range(1, 60):
    img = cv2.imread(f'/root/racecar_ws/src/final_challenge/final_challenge//racetrack_images/lane_3/image21.png')
    # processed = threshold_white(img)
    processed = process_frame(img)
    image_print(processed)