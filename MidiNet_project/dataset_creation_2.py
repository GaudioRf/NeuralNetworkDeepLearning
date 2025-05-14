#this follows dataset_creation_1.py. It remove pauses 
#prunes the matrices and segment them

import numpy as np
from functools import reduce
import os
import glob

#this function receives the path and unpack the info and the data
def unpack(path_and_file_index):
  
  path, file_index = path_and_file_index
  
  info_and_data = np.load(path)
  melody = info_and_data['melody']
  info = info_and_data['info']
  
  return [melody, info, file_index]

#this function take the matrix and translates everything in the range [0,MAX]
def prune(out_unpack):
  
  matrix, info, file_index = out_unpack
  maximum, minimum, extension, numerator, denominator, bpm = info.astype(int)
  
  matrix = matrix[minimum : maximum,:]
  
  global MAX 
  global MAX_EXT
  
  pruned_matrix = np.zeros((MAX_EXT, matrix.shape[1]))
  pruned_matrix[0: extension] = matrix
  
  return [pruned_matrix, numerator, denominator, bpm, file_index]

def process_melody_matrix(out_prune):
  
  melody, numerator, denominator, bpm, file_index = out_prune
  
  # step 1: remove leading silent columns
  # find the first column that is not all zeros
  non_zero_col_start = 0
  for col in range(melody.shape[1]):
    if not np.all(melody[:, col] == 0):
      non_zero_col_start = col
      break

  # slice the matrix to remove leading zeros
  melody = melody[:, non_zero_col_start:]

  # step 2: extend notes over silent columns
  for col in range(1, melody.shape[1]):
    if np.all(melody[:, col] == 0):         # if the column is silent (all zeros)
      melody[:, col] = melody[:, col - 1]   # copy the previous column's values
      
  return [melody, numerator, denominator, bpm, file_index]

def segment_piano_roll(out_process_melody_matrix):
  
  piano_roll, numerator, _, bpm, file_index = out_process_melody_matrix

  # calculate number of beats per bar and time steps per beat
  beats_per_bar = numerator
  time_steps_per_beat = 60 / bpm * 100    # Assuming 100 time steps per second
  time_steps_per_bar = beats_per_bar * time_steps_per_beat

  # calculate the number of time steps in 8 bars
  time_steps_per_segment = 8 * time_steps_per_bar
  
  time_steps_per_segment = int(time_steps_per_segment)

  # segment the piano roll
  
  dest_path = "/mnt/segments/"
  os.makedirs(dest_path, exist_ok=True)  # Ensure destination directory exists
    
  # Save each segment directly to disk to avoid memory issues
  for seg_idx, start in enumerate(range(0, piano_roll.shape[1], time_steps_per_segment)):
    end = start + time_steps_per_segment
    segment = piano_roll[:, start:end]
        
    if segment.shape[1] == time_steps_per_segment:
      # Save the segment to disk
      segment_path = f"{dest_path}segment_{file_index}_{seg_idx}.npz"
      np.savez_compressed(segment_path, segment=segment)
      
  return  # No data return, segments saved directly

def composite_function(*func):
    
    def compose(f, g):
        return lambda x : f(g(x))
            
    return reduce(compose, func, lambda x : x)

dataset_path = "/mnt/dataset_first/"

preprocessing_2 = composite_function(segment_piano_roll, process_melody_matrix, prune, unpack)

#loading parameters
params = np.load(dataset_path + "MAX_MIN.npz")
params_array = params["MAX_MIN"]
MAX, MIN, MAX_EXT = params_array.astype(int)  # Cast to int if necessary


# Process each file individually without accumulating data in memory
files_path = glob.glob("/mnt/dataset_first/music_file*.npz")

for i, path in enumerate(files_path):
    preprocessing_2([path, i])
    

