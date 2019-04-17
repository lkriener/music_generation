import copy 
import random as r
import numpy as np
import py_midicsv as midi

def load_to_csv(filepath):
    s = midi.midi_to_csv(filepath)
    return s

def split_tracks(csv_string):
    # first number in each line is the track number
    # track 0 always contains some additional info, not the music
    # returns a dictionary with track numbers as keys 
    # and list of csv lines as values
    tracks = {}
    for line in csv_string:
        channel = line[0]
        if channel in tracks.keys():
            tracks[channel].append(line)
        else:
            tracks[channel] = [line]
    return tracks

def get_bpm(csv_string):
    # assume no tempo changes in file
    # necessary value does not have to be included in file
    # in that case use midi-default
    # return value, bool if default was used
    val = -1
    default = True
    for line in csv_string:
        # find line with Time_signature
        split_line = [x.strip() for x in line.split(',')]
        if split_line[2] == 'Tempo':
            val = int(split_line[3])
            default = False
            break
    # midi default tempo: 500000 microseconds per quarter note
    if default:
        val = 500000
    bpm = 1./val * 1e6 * 60
    return bpm, default

def get_ticks_per_quarter(csv_string):
    # how many midi-timesteps are one quarter note
    # can be different for each track
    # therefore this function operates on a single track from the track_dict
    # info does not have to be included in track 
    # in that case use midi-default (1024)
    # return value, bool if default was used
    default = True
    ticks = 1024
    for line in csv_string:
        # find line with Time_signature
        split_line = [x.strip() for x in line.split(',')]
        if split_line[2] == 'Header':
            ticks = int(split_line[5])
            default = False
            break
    return ticks, default

def replace_int_in_line(line, pos, new_val):
    # replace an int-value in a csv line at pos with a new value
    split_line = [x.strip() for x in line.split(',')]
    split_line[pos] = str(new_val)
    new_line = ''
    for elem in split_line[:-1]:
        new_line += elem
        new_line += ', '
    new_line += split_line[-1]
    new_line += '\n'
    return new_line

def reduce_to_single_track(filename, new_filename, tracknr):
    # can be used to make file monophonic if the tracknr of the track 
    # that should be kept is known (has to be determined with other functions)
    # takes filename of polyphonic mide and name of file to which monophonic should be written
    s = load_to_csv(filename)
    track_dict = split_tracks(s)
    new_csv = []
    # must keep first and last line of track '0'
    header_line = track_dict['0'][0]
    # change the number of tracks that is written there to 1
    header_line = replace_int_in_line(header_line, 4, 1)
    new_csv.append(header_line)
    # copy lines of chosen track to new csv
    for elem in track_dict[tracknr]:
        new_csv.append(elem)
    new_csv.append(track_dict['0'][-1])
    # write new csv to disc
    with open(new_filename[:-4]+'.csv', 'w') as writeFile:
        for line in new_csv:
            writeFile.write(line)
    # use provided function to load midi pattern from csv file
    # in order to use built-in function to write midi file itself to disc
    pattern = midi.csv_to_midi(new_filename[:-4]+'.csv')
    with open(new_filename, 'wb') as midiFile:
        writer = midi.FileWriter(midiFile)
        writer.write(pattern)
    return

def check_if_note(line):
    # check if track-line is a note event
    # second return value indicates if on or off event
    split_line = [x.strip() for x in line.split(',')]
    if split_line[2] == 'Note_on_c':
        return True, 1
    elif split_line[2] == 'Note_off_c':
        return True, 0
    else:
        return False, -1

def get_note_info(line):
    # before calling, make sure that line really is note event
    split_line = [x.strip() for x in line.split(',')]
    pitch = int(split_line[4])
    time = int(split_line[1])
    return pitch, time
    
def write_track_line(track_nr, time, command, valuelist):
    # write single line of midi track with given content
    line = str(track_nr) + ', ' + str(time) + ', ' + str(command)
    if len(valuelist) > 0:
        for item in valuelist:
            line += ', ' + str(item)
    line += '\n'
    return line
    
def midi_track_to_numpy(track):
    # convert midi track to numpy array
    # [[pitch0, duration0], [pitch1, duration1],...]
    notes = []
    for i, line in enumerate(track):
        is_note, on_off = check_if_note(line)
        if not is_note:
            continue
        else:
            pitch, time = get_note_info(line)
            notes.append([on_off, time, pitch])
    # split into off- and on-events
    mask = np.array([bool(elem[0]) for elem in notes])
    on_notes = np.array(notes)[mask]
    mask = np.logical_not(mask)
    off_notes = np.array(notes)[mask]
    result = []
    for on_note in on_notes:
        start = on_note[1]
        pitch = on_note[2]
        # find corresponding off-event
        off_idx = np.where(off_notes[:,2] == pitch)
        if len(off_idx[0]) == 0:
            # no corresponding off-event found
            # -> ignore on event
            continue
        off_idx = off_idx[0][0]
        end = off_notes[off_idx][1]
        result.append([pitch, end-start])
        # delete off event (because off-event can only correspond to one on event)
        off_notes = np.delete(off_notes, off_idx, axis=0)
    return np.array(result)

def numpy_to_midi_track(array, track_nr, title, key_signature=(0, 'major'), time_signature=[4, 2, 24, 8]):
    # convert numpy array (as defined in midi_track to numpy) to midi track
    # additional info about the track besides array must be supplied
    track_list = []
    track_list.append(write_track_line(track_nr, 0, 'Start_track', []))
    track_list.append(write_track_line(track_nr, 0, 'Title_t', ['"'+title+'"']))
    track_list.append(write_track_line(track_nr, 0, 'Key_signature', [key_signature[0], '"'+key_signature[1]+'"']))
    track_list.append(write_track_line(track_nr, 0, 'Time_signature', time_signature))
    time = 1024
    for note in array:
        track_list.append(write_track_line(track_nr, time, 'Note_on_c', [0, note[0], 90]))
        time += note[1]
        track_list.append(write_track_line(track_nr, time, 'Note_off_c', [0, note[0], 0]))
    time += 1024
    track_list.append(write_track_line(track_nr, 0, 'End_track', []))
    return track_list
    
def write_to_midi(csv_list, new_filename):
    # write csv_list to disc, convert csv file to midi file
    with open(new_filename[:-4]+'.csv', 'w') as writeFile:
        for line in csv_list:
            writeFile.write(line)
    # use provided function to load midi pattern from csv file
    # in order to use built-in function to write midi file itself to disc
    pattern = midi.csv_to_midi(new_filename[:-4]+'.csv')
    with open(new_filename, 'wb') as midiFile:
        writer = midi.FileWriter(midiFile)
        writer.write(pattern)
    return
    
def track_dict_to_csv(track_dict):
    # convert track_dict to original csv_list
    csv_list = []
    for elem in track_dict['0'][:-1]:
        csv_list.append(elem)
    for key in track_dict.keys():
        if key == '0':
            continue
        else:
            for elem in track_dict[key]:
                csv_list.append(elem)
    csv_list.append(track_dict['0'][-1])
    return csv_list
 
def find_lead_track(track_dict, random=True, largest_range=False, variation=False, rhythm=False):
    # determine the lead track using chosen criterions
    # Remove Header track because it is not really a track which contains music
    new_track_dict = copy.deepcopy(track_dict)
    del new_track_dict['0']
    key_list = list(new_track_dict.keys())
    print(key_list)
    num_tracks = len(key_list)
    # if criterion random is active all others are ignored
    if random:
        nr = r.randint(0, num_tracks-1)
        return key_list[nr]
    scores = [0]*num_tracks
    numpy_tracks = []
    for key in key_list:
        numpy_track = midi_track_to_numpy(new_track_dict[key])
        numpy_tracks.append(numpy_track)
    # largest_range choses the track with highes difference between lowest and highest note
    if largest_range:
        ranges = []
        for n_track in numpy_tracks:
            min_note = np.amin(n_track, axis=0)[0]
            max_note = np.amax(n_track, axis=0)[0]
            ranges.append(max_note - min_note)
        track_idx = np.argmax(ranges)
        scores[track_idx] += 1
        print('Note range:', ranges)
    # find the track with the most different notes
    if variation:
        num_diff_notes = []
        for n_track in numpy_tracks:
            n_diffs = len(np.unique(n_track[:,0]))
            num_diff_notes.append(n_diffs)
        track_idx = np.argmax(num_diff_notes)
        print('Number of different notes:', num_diff_notes)
        scores[track_idx] += 1
    # find the track with the most different note lengths indicating most rhythm
    if rhythm:
        num_diff_lengths = []
        for n_track in numpy_tracks:
            n_diffs = len(np.unique(n_track[:,1]))
            num_diff_lengths.append(n_diffs)
        track_idx = np.argmax(num_diff_lengths)
        print('Number of different lengths:', num_diff_lengths)
        scores[track_idx] += 1
    # find the track with the highest score, if tied -> choose randomly
    winners = np.argwhere(scores == np.amax(scores))
    print('Scores:', scores)
    if len(winners) == 0:
        return key_list[winners[0]]
    else:
        winner = r.randint(0, len(winners)-1)
        return key_list[winner]
    
        