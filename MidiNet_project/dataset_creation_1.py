#this script is 1 of 2
# 
#
# 
#   
import numpy as np
import pretty_midi
from functools import reduce
import glob

#functions to compose

def get_piano_and_info(paths):
    path = paths[0]
    dest_path = paths[1]
    midi_data = pretty_midi.PrettyMIDI(path)
    fs = 100
    piano_roll = midi_data.get_piano_roll(fs = fs)
    _, tempo_changes = midi_data.get_tempo_changes()
    time_signatures = midi_data.time_signature_changes
    tempo = tempo_changes[0]
    time_signature = [time_signatures[0].numerator, time_signatures[0].denominator]
    return [piano_roll, tempo, time_signature, dest_path]

def melody_extractor(out_get_piano_and_info, ignore_velocity=True):
    
  melody = out_get_piano_and_info[0]
  tempo = out_get_piano_and_info[1]
  time_signature = out_get_piano_and_info[2]
  dest_path = out_get_piano_and_info[3]
  
  #M = matrix.copy()
  if ignore_velocity:
    melody[melody != 0] = 1
    
  for col in range(melody.shape[1]):
    # find the index of the first non-zero element in the column
    for row in range(melody.shape[0]-1,-1,-1):
      if melody[row, col] != 0:
        # set all other elements in the column to 0
        melody[:row, col] = 0
        break
  
  return [melody, tempo, time_signature, dest_path]

def drop_short_notes(out_melody_extractor):
    
  modified_roll = out_melody_extractor[0]
  bpm = out_melody_extractor[1]
  time_signature = out_melody_extractor[2]
  dest_path = out_melody_extractor[3]
  
  numerator, denominator = time_signature
  
  # Calculate number of beats per bar and time steps per beat
  beats_per_bar = numerator
  time_steps_per_beat = 60 / bpm * 100  # Assuming 100 time steps per second
  time_steps_per_16th = time_steps_per_beat / 4  # 1/16 is a quarter of a beat

  # Iterate over all notes (rows)
  for pitch in range(128):
    # Find the indices where the note is played (non-zero values)
    note_indices = np.where(modified_roll[pitch] > 0)[0]

    if len(note_indices) == 0:
      continue

    # Determine the start and end of each note
    note_start = note_indices[0]
    for i in range(1, len(note_indices)):
      if note_indices[i] != note_indices[i - 1] + 1:  # If there's a gap
        note_end = note_indices[i - 1]
        note_duration = note_end - note_start + 1  # +1 because end is inclusive

        # Drop short notes
        if note_duration < time_steps_per_16th:
            modified_roll[pitch, note_start:note_end + 1] = 0  # Set to 0 if duration < 1/16
        note_start = note_indices[i]  # Update start for the next note

    # Handle the last note in the sequence
    note_end = note_indices[-1]
    note_duration = note_end - note_start + 1
    if note_duration < time_steps_per_16th:
      modified_roll[pitch, note_start:note_end + 1] = 0
   
  return [modified_roll, bpm, time_signature, dest_path]

#This is the last step I need a function that takes trace of the global maximum and the global minimum,
#and save the modified piano roll and a vector containing (maximum, mimimum, bpm)

def extremes(out_drop_short_notes):
    
    matrix = out_drop_short_notes[0]
    bpm = out_drop_short_notes[1]
    time_signature = out_drop_short_notes[2]
    dest_path = out_drop_short_notes[3]
    
    global MAX
    global MIN
    global MAX_EXT
    
    for i, row in enumerate(matrix):
        if any(elem == 1 for elem in row): 
            minimum = i
            break

    for i in range(len(matrix) - 1, -1, -1):
        if any(elem == 1 for elem in matrix[i]):  # Removed the extra comma
            maximum = i  # Adjusted index to reflect reversed list
            break
    
    extension = maximum - minimum

    if minimum < MIN: 
        MIN = minimum
    if maximum > MAX: 
        MAX = maximum
    if extension > MAX_EXT:
      MAX_EXT = extension
    
    numerator, denominator = time_signature
    info_array = np.array([maximum, minimum, extension, numerator, denominator, bpm])
    
    np.savez_compressed(dest_path, melody = matrix, info = info_array)

    return 

#defining the composition
#1.get_piano_and_info
#2.melody_extractor
#3.drop_short_notes
#4.extremes

def composite_function(*func):
    
    def compose(f, g):
        return lambda x : f(g(x))
            
    return reduce(compose, func, lambda x : x)

preprocessing_1 = composite_function(extremes, drop_short_notes, melody_extractor, get_piano_and_info)


#global variables
MAX = float('-inf')
MIN = float('inf')
MAX_EXT = float('-inf')      #maximum extension

#loading data
files_path = glob.glob("/mnt/all_years_midi/*.midi")
destination_path = "/mnt/dataset_first/" 

i = 0

for path in files_path:
  dest_file = destination_path + "music_file" + str(i)
  preprocessing_1([path, dest_file])

  i += 1

np.savez_compressed(destination_path + "MAX_MIN", MAX_MIN = np.array([MAX,MIN,MAX_EXT]))
print(MAX,'\n')
print(MIN,'\n')