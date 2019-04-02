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

def get_ticks_per_quarter(track):
    # how many midi-timesteps are one quarter note
    # can be different for each track
    # therefore this function operates on a single track from the track_dict
    # info does not have to be included in track 
    # in that case use midi-default (24)
    # return value, bool if default was used
    default = True
    ticks = 24
    for line in track:
        # find line with Time_signature
        split_line = [x.strip() for x in line.split(',')]
        if split_line[2] == 'Time_signature':
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