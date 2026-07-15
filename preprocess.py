import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

def get_notes(directory_path):
    """ Get all the notes and chords from the midi files in the specified directory """
    notes = []
    
    # Load all midi files in the given directory
    midi_files = glob.glob(directory_path + "/*.mid")
    if not midi_files:
        print(f"No MIDI files found in {directory_path}")
        return []
        
    print(f"Found {len(midi_files)} MIDI files. Starting extraction...")

    for file in midi_files:
        try:
            midi = converter.parse(file)
            print(f"Parsing {file}")

            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() 
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Failed to parse {file}: {e}")

    return notes

def prepare_sequences(notes, sequence_length=100):
    """ Prepare the sequences used by the Neural Network """
    
    # Get all pitch names
    pitches = sorted(set(item for item in notes))
    n_vocab = len(pitches)
    
    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitches))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input_reshaped = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input
    network_input_reshaped = network_input_reshaped / float(n_vocab)

    return network_input_reshaped, np.array(network_output), note_to_int, n_vocab

if __name__ == '__main__':
    dataset_path = "MIDI MUSIC dataset/midi_dataset/midi_dataset"
    
    print("Extracting notes from MIDI files...")
    notes = get_notes(dataset_path)
    
    if len(notes) > 0:
        print(f"Total notes extracted: {len(notes)}")
        print("Preparing sequences...")
        
        sequence_length = 100
        network_input, network_output, note_to_int, n_vocab = prepare_sequences(notes, sequence_length)
        
        print("Saving preprocessed data...")
        # Save the required objects to disk
        with open('data.pkl', 'wb') as filepath:
            pickle.dump({
                'network_input': network_input,
                'network_output': network_output,
                'note_to_int': note_to_int,
                'n_vocab': n_vocab
            }, filepath)
            
        print("Data preprocessing complete. Saved to data.pkl")
    else:
        print("Error: No notes were extracted. Check your dataset path.")
