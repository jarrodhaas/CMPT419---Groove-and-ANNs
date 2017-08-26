# ETM -- Feature extraction script. 2016 - Jarrod Haas.

from mido import MidiFile
import numpy as np
import random as rand
import os as os
import sys as sys

midi_path = '/Library/Application Support/EZDrummer/Midi/000329@AMERICAN_ROCK'

# for debugging
verbose = 0;

#number of midi files processed
num_midi = 0

# number of note values
num_notes = 32

#current cumulative time index in midi file
time_index = 0
#1 feature per 16th note subdivision, so 1 bin per note
num_bins = num_notes
#number of instruments
num_inst = 3

#current bin
curr_bin = 0

# number of instrument mapping failures
map_failures = 0
map_fail_values = np.zeros((1))

# part failure tracking
part_failure = 0

# number of bin collisions
collisions = 0

#class label position in sample vector
class_label = num_bins+num_notes+1

#tempo position in sample vector
tempo_label = num_bins + num_notes

# kick,snare,hats, 32 16th notes, tempo and classification at end
sample = np.zeros((3,num_bins+num_notes+2))
# keep track of filled bins to avoid collisions and properly place notes
bin_flags = np.zeros((3,num_bins+num_notes))

#tempo of file
tempo = 0

# map various possible midi values for kick, snare hats to 0,1, or 2
def mapInstrument(val, mapping):

    inst = 0

    if (mapping == 0):

        # hats
        if (val >= 7 and val <= 26):
            inst = 2
        elif (val == 0 or val == 42 or val == 44 or val == 46):
            inst = 2
        elif (val >= 51 and val <= 65):
            inst = 2
        elif (val >= 84 and val <= 124):
            inst = 2
        #mapping floor tom as hats
        elif (val == 43 or val == 47 or val == 41 or val == 48):
            inst = 2

        # kicks
        elif ((val >= 34 and val<= 36)):
            inst = 0

        #snares
        elif (val == 6 or val == 33):
            inst = 1
        elif (val >= 66 and val<= 71):
            inst = 1
        elif (val >= 125 and val<= 127):
            inst = 1
        elif (val >= 37 and val<= 40):
            inst = 1

        #error!
        else:
            inst = 4

    elif (mapping == 1):

        if (val == 44):
            inst = 2
        elif (val == 40):
            inst = 1
        elif (val == 24):
            inst = 0

        #error!
        else:
            inst = 4

    return inst

def doMessage(message):

    global time_index
    global curr_bin
    global map_failures
    global collisions
    global sample
    global bin_flags
    global num_bins
    global num_notes
    global map_fail_values

    if (message.type == "note_off"):
        print ("Error: note off midi message found")


    time_index += message.time

    # vel = 0 means note ending is triggered by a note_on event, don't need to record its offset
    if (message.type == "note_on" and message.velocity != 0):

      #map instrument value to kick, snare, hats if appropriate
      # second variable is midi mapping: 0 is toontrack library, 1 is logic audio
      instrument = mapInstrument(message.note, 0)

      #if the note is for kick, snare or hats
      if (instrument != 4):

          ibin = time_index / 2400
          curr_bin = ibin

          if (curr_bin > 31):
              print ("Last bin filled. ")
              return 0;

          #calculate deviation (in ticks) from nearest 16th note
          offset = time_index % 2400

          #consider the note late
          if (offset < 1200):
            #if bin is not filled
            if (bin_flags[instrument,ibin] == 0):
              sample[instrument,ibin] = int(offset)
              # indicate this note value is present
              sample[instrument,ibin+num_notes] = 1
            else:
              collisions += 1

          # consider note early, use a negative offset
          elif (offset >= 1200 and offset < 2400 and ibin <= 30):
              offset = 2400 - offset
              offset *= (-1)

              sample[instrument,ibin+1] = int(offset)
              sample[instrument,ibin+1+num_notes] = 1

          if (verbose == 1):
              print ("instrument: ", instrument, " bin: ", ibin, " offset: ", offset, "time_index: ", time_index)
              print (message)
              print("\n")


      else:
          print ("Error: instrument not mapped: ", instrument)
          print (message)
          map_failures += 1

          #record the instrument that was not mapped
          if (np.nonzero(map_fail_values == message.note)[0].size == 0):
            map_fail_values = np.append(map_fail_values, message.note)


    return 1;


def resample(inst):

    global num_bins
    global sample

    # setup first resampling picks for each instrument
    if (inst == 0):
        choice1 = 2
        choice2 = 1
    elif (inst == 1):
        choice1 = 2
        choice2 = 0
    elif (inst == 2):
        choice1 = 1
        choice2 = 0

    for i in range(0,num_bins):

        #only resample if note value wasn't present! don't want to resample if no actual offset
        if (sample[inst, i] == 0 and sample[inst,i+num_notes] == 0):
            if (sample[choice1,i] !=0):
                sample[inst,i] = sample[choice1,i]
            elif (sample[choice2,i] !=0):
                sample[inst,i] = sample[choice2,i]

    return 1;

def resampleOffsets():

    global num_inst

    print("Resampling offset values from other instruments where possible.")

    for i in range(0,num_inst):
        resample(i)

    return 1

def doNullSample():

    global sample

    print ("Creating null sample.")

    null_sample = sample.copy()

    for i in range(0,num_inst):
        null_sample[i,0:32] = 0

    return null_sample;

def doRandomSample(center, noise):

# center is coefficent of offset value (.5 is halfway between null offset and actual offset)
# noise multiplies by offset value to determine +/- range from center

    global sample
    global num_inst
    global num_bins

    # if a note has a 0 offset (perfectly timed) we still want to add some random noise
    zero_scalar = 5

    print ("Creating random sample with center: ", center, " noise: ", noise)

    rand_sample = sample.copy()

    for i in range(0,num_inst):
        for j in range(0, num_bins):

          base = int(center*sample[i,j])
          offset_range = int(abs(sample[i,j]*noise))

          if (offset_range == 0 and sample[i,j+32] == 1):
            #rand_sample[i,j] = rand.randint(base-int(center*zero_scalar), base+int(center*zero_scalar))
            rand_sample[i,j] = rand.randint(base-offset_range, base+offset_range)
          else:
            rand_sample[i,j] = rand.randint(base-offset_range, base+offset_range)

    return rand_sample;


def convertSample(samp, label):

    global num_bins
    global num_notes
    global num_inst
    global tempo

    new_samp = np.zeros((1,(num_bins+num_notes)*num_inst+2))

    for i in range(0,num_inst):
        for j in range(0, num_bins+num_notes):
            new_samp[0,j+(i*(num_bins+num_notes))] = samp[i,j]

    new_samp[0,(num_bins+num_notes)*num_inst] = (60*1000000) / tempo
    new_samp[0,(num_bins+num_notes)*num_inst+1] = label

    return new_samp;

def printStats():

    print ("map_failures: ", map_failures)
    print ("collisions: ", collisions)
    print ("Part failures: ", part_failure)
    #print float(mid.ticks_per_beat)
    print("map fail values: ", map_fail_values)

    print("number of samples created: ", num_midi)


    return 1;

def convertRNNSampleInst(samp):

    global num_bins
    global num_notes

    # 1 by 32 note array (2 bars with 16th note resolution)
    new_samp = np.zeros((1,num_notes))

    # output sample format for RNN
    # each kick/snare/hat combination has a unique id 2^3 combinations, so 8 id's
    #0 = rest
    #1 = kick
    #2 = snare
    #3 = hats
    #4 = kick/snare
    #5 = kick/hats
    #6 = snare/hats
    #7 = kick/snare/hats

    for i in range(num_notes, num_notes+num_bins):

        if(samp[0,i] == 0 and samp[1,i] == 0 and samp[2,i] == 0):
            new_samp[0,i-32] = 0
        elif(samp[0,i] == 1 and samp[1,i] == 0 and samp[2,i] == 0):
            new_samp[0,i-32] = 1
        elif(samp[0,i] == 0 and samp[1,i] == 1 and samp[2,i] == 0):
            new_samp[0,i-32] = 2
        elif(samp[0,i] == 0 and samp[1,i] == 0 and samp[2,i] == 1):
            new_samp[0,i-32] = 3
        elif(samp[0,i] == 1 and samp[1,i] == 1 and samp[2,i] == 0):
            new_samp[0,i-32] = 4
        elif(samp[0,i] == 1 and samp[1,i] == 0 and samp[2,i] == 1):
            new_samp[0,i-32] = 5
        elif(samp[0,i] == 0 and samp[1,i] == 1 and samp[2,i] == 1):
            new_samp[0,i-32] = 6
        elif(samp[0,i] == 1 and samp[1,i] == 1 and samp[2,i] == 1):
            new_samp[0,i-32] = 7

    return new_samp;

def convertRNNSampleTiming(samp):
#this function grabs timing values with layering priorities as below

    global num_bins
    global num_notes

    # 1 by 32 note array (2 bars with 16th note resolution)
    new_samp = np.zeros((1,num_notes))

    # output sample format for RNN
    # each kick/snare/hat combination has a unique id 2^3 combinations, so 8 id's
    #0 = rest
    #1 = kick
    #2 = snare
    #3 = hats
    #4 = kick/snare
    #5 = kick/hats
    #6 = snare/hats
    #7 = kick/snare/hats

    for i in range(0, num_notes):

        if(samp[0,i] == 0 and samp[1,i] == 0 and samp[2,i] == 0):
            new_samp[0,i] = samp[0,i]
        elif(samp[0,i] != 0 and samp[1,i] == 0 and samp[2,i] == 0):
            new_samp[0,i] = samp[0,i]
        elif(samp[0,i] == 0 and samp[1,i] != 0 and samp[2,i] == 0):
            new_samp[0,i] = samp[1,i]
        elif(samp[0,i] == 0 and samp[1,i] == 0 and samp[2,i] != 0):
            new_samp[0,i] = samp[2,i]
        elif(samp[0,i] != 0 and samp[1,i] != 0 and samp[2,i] == 0):
            new_samp[0,i] = samp[2,i]
        elif(samp[0,i] != 0 and samp[1,i] == 0 and samp[2,i] != 0):
            new_samp[0,i] = samp[0,i]
        elif(samp[0,i] == 0 and samp[1,i] != 0 and samp[2,i] != 0):
            new_samp[0,i] = samp[2,i]
        elif(samp[0,i] != 0 and samp[1,i] != 0 and samp[2,i] != 0):
            new_samp[0,i] = samp[2,i]


    return new_samp;

def doSamples():

    global part_failure
    global num_midi

    #check for missing parts, some midi samples are missing hats for e.g.

    for i in range(0,num_inst):
        if (sample[i,32:64].sum() == 0):
            part_failure += 1
            return 0;

    # fill in absent timing values using other instruments in that bin, if possible
    resampleOffsets()

    #
    #create additional sample data that represents 'bad' expressive timing
    #

    # generate samples with random timing variations
    #low_noise_sample = doRandomSample(0.5, .25)
    mid_noise_sample = doRandomSample(1, .25)
    #high_noise_sample = doRandomSample(1.5, .25)

    # create a perfectly quantized sample
    #null_sample = doNullSample()

    #convert sample, class label to input format for algorithm
    et_sample = convertSample(sample, 1)

    #low_sample = convertSample(low_noise_sample, 0)
    mid_sample = convertSample(mid_noise_sample, 0)
    #high_sample = convertSample(high_noise_sample, 0)
    #null_sample = convertSample(null_sample, 0)

    #print csv for debugging
    #np.savetxt("sample.csv", sample, fmt='%d', delimiter=",")
    #np.savetxt("randsample.csv", rand_sample, fmt='%d', delimiter=",")

    # do RNN sample format for instrument output
    rnn_sample_inst = convertRNNSampleInst(sample)

    # do RNN sample format for timing output
    rnn_sample_timing = convertRNNSampleTiming(sample)


    with open('training3.0.csv','a') as f_handle:
        np.savetxt(f_handle,et_sample,fmt='%d', delimiter=",")
        #np.savetxt(f_handle,low_sample,fmt='%d', delimiter=",")
        np.savetxt(f_handle,mid_sample,fmt='%d', delimiter=",")
        #np.savetxt(f_handle,high_sample,fmt='%d', delimiter=",")
        #np.savetxt(f_handle,null_sample,fmt='%d', delimiter=",")

    #output RNN training format (instrument bins)
    with open('RNNtraining_inst.csv','a') as f2_handle:
        np.savetxt(f2_handle,rnn_sample_inst, fmt='%d', delimiter=",")

    #output RNN training format (timing bins)
    with open('RNNtraining_timing.csv','a') as f3_handle:
        np.savetxt(f3_handle,rnn_sample_timing, fmt='%d', delimiter=",")

    num_midi += 1

    return 1;

#
# begin main body of code
#

# go through each subdirectory and open each midi file. extract first 2 bars from each file.

for subdir, dirs, files in os.walk(midi_path):
    for file in files:

        if (file.endswith(".mid") and (subdir.find("FILLS") == -1)):

            print("\n")
            print os.path.join(subdir, file)

            mid = MidiFile(os.path.join(subdir,file))

            #reset values

            sample = np.zeros((3,num_bins+num_notes+2))
            bin_flags = np.zeros((3,num_bins+num_notes))
            curr_bin = 0
            time_index = 0



            for i, track in enumerate(mid.tracks):
                #print('Track {}: {}'.format(i, track.name))

                # check for cases where there is more than 1 track, we don't want this
                if i > 1:
                    print("Error: More than one track found.")
                    break;

                # go through each midi message in the track
                for message in track:
                    if (message.type == 'set_tempo'):
                        tempo = message.tempo
                    if (doMessage(message) == 0):
                        break;


                #process and write sample data
                doSamples()



print("\nProcessing complete. Results: ")
printStats()
