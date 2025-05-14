import numpy as np
import pretty_midi



def get_major_minor_scales():
  MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
  MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

  scales = {}
  for root in range(12):
    scales[f"{pretty_midi.note_number_to_name(root)} Major"] = [(root + i) % 12 for i in MAJOR_SCALE]
    scales[f"{pretty_midi.note_number_to_name(root)} Minor"] = [(root + i) % 12 for i in MINOR_SCALE]
  return scales


def detect_chords(midi_data):
  chords = []

  # define a small time window for grouping notes (e.g., 0.5 seconds)
  time_window = 0.5
  for instrument in midi_data.instruments:
      notes = sorted(instrument.notes, key=lambda x: x.start)
      current_chord = []
      chord_start_time = notes[0].start if notes else 0

      for note in notes:
        # check if the note is within the current time window
        if note.start - chord_start_time < time_window:
          current_chord.append(note.pitch % 12)  # Reduce to pitch class
        else:
            if current_chord:
              chords.append((current_chord, chord_start_time))
            current_chord = [note.pitch % 12]
            chord_start_time = note.start

      # append last chord if any
      if current_chord:
        chords.append((current_chord, chord_start_time))

  return chords


def identify_chord_type(chord):
    if len(chord) < 3:
        return None  # not enough notes to form a triad

    chord.sort()
    intervals = [(chord[i+1] - chord[i]) % 12 for i in range(len(chord)-1)]

    # major triad intervals: 4 semitones + 3 semitones
    if intervals == [4, 3]:
        return "Major"
    # minor triad intervals: 3 semitones + 4 semitones
    elif intervals == [3, 4]:
        return "Minor"
    # diminished triad intervals: 3 semitones + 3 semitones
    elif intervals == [3, 3]:
        return "Diminished"
    else:
        return None

def score_key_with_chords(chords, scales, is_major):
  MAJOR_TRIADS = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii°"}
  MINOR_TRIADS = {0: "i", 2: "ii°", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}

  triads = MAJOR_TRIADS if is_major else MINOR_TRIADS
  score = 0

  for chord, _ in chords:
    root_note = chord[0]  # Take the root of the chord (first note)
    chord_type = identify_chord_type(chord)

      # Check if the chord root matches any triad in the key
    if root_note in triads:
      if (is_major and chord_type == "Major") or (not is_major and chord_type == "Minor"):
        score += 1
      elif chord_type == "Diminished":
        score += 0.5

  return score


def infer_key_signature_with_harmony(midi_data):
  # extract note histogram
  note_histogram = np.zeros(12)
  for instrument in midi_data.instruments:
    for note in instrument.notes:
      pitch_class = note.pitch % 12  # Reduce to pitch class
      note_histogram[pitch_class] += 1

  # normalize the histogram
  note_histogram /= np.sum(note_histogram)

  # get major and minor scales
  scales = get_major_minor_scales()

  # detect chords for harmonic analysis
  chords = detect_chords(midi_data)

  # compare scores for all major and minor scales with harmonic analysis
  best_match = None
  best_score = -np.inf
  for scale_name, scale_pcs in scales.items():
    is_major = "Major" in scale_name

    # score based on note histogram match
    scale_mask = np.zeros(12)
    scale_mask[scale_pcs] = 1
    note_match_score = np.dot(note_histogram, scale_mask)

    # score based on harmonic analysis (chord progression)
    harmonic_score = score_key_with_chords(chords, scale_pcs, is_major)

    # combine both scores (you can adjust weights)
    total_score = note_match_score + harmonic_score

    if total_score > best_score:
      best_score = total_score
      best_match = scale_name

  return best_match

#----------------------------------------------------------------------------------------------------

def midi_info(midi_data, verbose=False):
  time_signatures = midi_data.time_signature_changes
  key_signature = infer_key_signature_with_harmony(midi_data)
  times, tempo_changes = midi_data.get_tempo_changes()

  if verbose:
    print('----------------------------------------------')
    print(f'There are {len(time_signatures)} time signature changes')
    for ts in time_signatures:
      print(f"Time signature: {ts.numerator}/{ts.denominator} at time {ts.time} seconds")
    print(f'\nThere are {len(tempo_changes)} tempo changes')
    for tc in tempo_changes:
      print(f"Tempo change: {tc} bpm at time {times[np.where(tempo_changes == tc)[0][0]]} seconds")
    print(f'\nKey signature: {key_signature}\n')
    print(f'There are {len(midi_data.instruments)} instruments')
    print(f'Instrument has {len(midi_data.instruments[0].notes)} notes')
    print(f'\nDuration: {np.round(midi_data.get_end_time()/60,4)} min')
    print('----------------------------------------------')

  return [(time_signatures[i].numerator,time_signatures[i].denominator) for i in range(len(time_signatures))], \
        [tempo_changes[i] for i in range(len(tempo_changes))], \
        key_signature

#----------------------------------------------------------------------------------------------------

def melody_extractor(matrix, ignore_velocity=True):
  M = matrix.copy()
  if ignore_velocity:
    M[M != 0] = 1

  melody = M.copy()

  for col in range(melody.shape[1]):
    # find the index of the first non-zero element in the column
    for row in range(melody.shape[0]-1,-1,-1):
      if melody[row, col] != 0:
        # set all other elements in the column to 0
        melody[:row, col] = 0
        break
  return melody

#----------------------------------------------------------------------------------------------------

def drop_short_notes(piano_roll, time_signature, bpm):

  numerator, denominator = time_signature

  # Calculate number of beats per bar and time steps per beat
  beats_per_bar = numerator
  time_steps_per_beat = 60 / bpm * 100  # Assuming 100 time steps per second
  time_steps_per_16th = time_steps_per_beat / 4  # 1/16 is a quarter of a beat

  # Create a copy of the piano roll to modify
  modified_roll = piano_roll.copy()

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

  return modified_roll

#----------------------------------------------------------------------------------------------------

def process_melody_matrix(melody):
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

  return melody

#----------------------------------------------------------------------------------------------------

def segment_piano_roll(piano_roll, time_signature, bpm):

  numerator, denominator = time_signature

  # calculate number of beats per bar and time steps per beat
  beats_per_bar = numerator
  time_steps_per_beat = 60 / bpm * 100    # Assuming 100 time steps per second
  time_steps_per_bar = beats_per_bar * time_steps_per_beat

  # calculate the number of time steps in 8 bars
  time_steps_per_segment = 8 * time_steps_per_bar

  # segment the piano roll
  segments = []
  for start in range(0, piano_roll.shape[1], int(time_steps_per_segment)):
    end = start + int(time_steps_per_segment)
    segment = piano_roll[:, start:end]
    if segment.shape[1] == int(time_steps_per_segment):   # ensure full segment
      segments.append(segment)

  return segments
  
#----------------------------------------------------------------------------------------------------

def midi_preprocessing(midi_data, fs=100, ignore_velocity=True):
  piano_roll_data = midi_data.get_piano_roll(fs=fs)

  time_signatures, tempos, key_signature =  midi_info(midi_data, verbose=False)
  time_signature = time_signatures[0]
  bpm = tempos[0]

  melody = melody_extractor(piano_roll_data, ignore_velocity=ignore_velocity)
  melody_w_no_short_notes = drop_short_notes(melody, time_signature, bpm)
  processed_melody = process_melody_matrix(melody_w_no_short_notes)
  segments = segment_piano_roll(processed_melody, time_signature, bpm)

  return segments