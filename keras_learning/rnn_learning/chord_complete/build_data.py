from mido import MidiFile
import numpy as np

def build(separated=True):
    mid = MidiFile('7ths.mid', clip=True)

    all_notes = []
    for track in mid.tracks:
        if track.name == "Velvet 1":
            for msg in track:
                if msg.type == "note_on":
                    all_notes.append(msg.note)

    chord_count = len(all_notes) // 4
    chords = np.array(all_notes).reshape(chord_count, 4)
    chords = np.flip(chords, axis=1)
    if separated:
        ins = chords[:, :-1].reshape(chord_count, 3, 1)
        outs = chords[:, -1:].flatten()
        return ins, outs
    else:
        return chords


if __name__ == "__main__":
    ins, outs = build(separated=True)
    print(ins)
    print(outs)
