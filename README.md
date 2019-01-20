# Teyered
Machine Learning framework for tiredness detection

### Logic flow
Logic flow of the provided code is as follows (excluding blinks for now):

```Load/Record data -> Eye area calculation from the data -> Save data```

Eye area calculation logic flow is as follows:

```Extract facial points -> Normalize eye points -> Calculate eye area```

### Example code

The following code illustrating the logic flow:

```python
# Setup logging level for the following loggers
logging.basicConfig(level=logging.DEBUG)
 
# Load/Record data
frame = take_photo()
 
# Extract facial points
facial_points_dictionary = extract_facial_points(frame)
 
# Normalize eye points
left_eye_normalized = normalize_eye_points(facial_points_dictionary['left_eye'])
right_eye_normalized = normalize_eye_points(facial_points_dictionary['right_eye'])
 
# Calculate eye area
left_eye_area = calculate_polygon_area(left_eye_normalized)
right_eye_area = calculate_polygon_area(right_eye_normalized)
 
# Save facial points data
write_points_to_file('left_eye_area.csv', left_eye_normalized)
write_points_to_file('right_eye_area.csv', right_eye_normalized)
```

Data validations are needed to make sure that the function was executed successfully. For example, ```extract_facial_points(frame)``` would return ```None``` if no eyes were detected in the provided image.

### Normalization of eye points

Since face is never static in front of the camera, we need to account for head movement, for example when face is closer or further away from the camera. That is, we need some universal measure for every eye such that the area is independent of the head position.

This problem is solved in the "normalize eye points" part of the logic flow. Points of any eye are mapped on a universal size grid that can be then used to compare different eyes in different environments. This does not, however, account for head rotation and tilt (looking up, to the right etc.) - this is a concern for the future.
