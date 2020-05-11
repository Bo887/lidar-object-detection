from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon

Coordinate = Tuple[float, float] # Add another float if using 3 dimensions
BoundingBox = List[Coordinate]

def euclidean_distance(c1: Coordinate, c2: Coordinate) -> float:
  dist_squared: float = sum([(d2 - d1) ** 2 for d1, d2 in zip(c1, c2)])
  return sqrt(dist_squared)

class speed_stat:
  def __init__(self, position: BoundingBox, samples_N):
    self.current_position = position
    self.speeds: List[float] = []
    self.average_speed = 0.0
    self.samples_N = samples_N # Max number of samples used to calculate average speed

  def update_position(self, new_position: BoundingBox):
    old_center = np.mean(self.current_position, axis=0)
    new_center = np.mean(new_position, axis=0)
    self.speeds.append(euclidean_distance(new_center, old_center))
    # Remove earliest speed if number of samples is too high
    if len(self.speeds) > self.samples_N: self.speeds.pop(0)
    self.average_speed = np.average(self.speeds)
    self.current_position = new_position

def parse_frame(
  current_objects: List[BoundingBox],
  speed_stat_dict: Dict[int, speed_stat],
  label_start=0,
  samples_N=10) -> Tuple[Dict[int, speed_stat], int]:
  '''
  Inputs:
  current_objects -- List of bounding boxes of objects in the current frame of the scene
  speed_stat_dict -- Labelled dictionary of detected vehicles/objects with coordinates and speed statistics object
  label_start -- Number to begin labelling detected objects with

  Output:
  Returns an updated version of the speed_stat_dict taking the current frame into consideration

  Functionality:
  Tries to match objects seen in frame to existing objects in speed_stat_dict based on positions.
  Updates the speed based on the change in position and returns a new version of speed_stat_dict
  '''

  new_speed_stat_dict: Dict[int, speed_stat] = {}

  for bounding_box in current_objects:
    closest_box_distance = float("inf")
    closest_box_id = None
    for idx, ss_obj in speed_stat_dict.items():
      # Check if bounding boxes overlap
      if Polygon(bounding_box).intersects(Polygon(ss_obj.current_position)):
        if euclidean_distance(np.mean(bounding_box, axis=0), np.mean(ss_obj.current_position, axis=0)) < closest_box_distance:
          closest_box_id = idx
    # Check whether to update existing speed object or create and add a new one
    if closest_box_id == None:
      new_speed_stat_dict[label_start] = speed_stat(bounding_box, samples_N)
      label_start += 1
    else:
      new_speed_stat_dict[closest_box_id] = speed_stat_dict[closest_box_id]
      new_speed_stat_dict[closest_box_id].update_position(bounding_box)
  return new_speed_stat_dict, label_start
