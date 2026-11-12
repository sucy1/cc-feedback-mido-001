# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

import io
import struct

from pytest import raises

from mido.midifiles.midifiles import MidiFile

# 25 fps, 40 ticks per frame (division 0xe728).
SMPTE_DIVISION = 0xe728 - 0x10000
PPQN_DIVISION = 480

# delta 0 note_on, delta 480 note_off, delta 0 end_of_track.
# (480 is 0x83 0x60 as a variable length quantity.)
TRACK_EVENTS = bytes.fromhex(
    '00 90 40 40'
    '83 60 80 40 40'
    '00 ff 2f 00'
)

# Same, but with a set_tempo (1000000 us/beat) before the note_off.
# A set_tempo must not affect wall-clock time in a SMPTE file.
TRACK_EVENTS_WITH_TEMPO = bytes.fromhex(
    '00 90 40 40'
    '00 ff 51 03 0f 42 40'
    '83 60 80 40 40'
    '00 ff 2f 00'
)


def make_smf(division, track_data, type=1):
    header = struct.pack('>hhh', type, 1, division)
    return (b'MThd' + struct.pack('>L', len(header)) + header
            + b'MTrk' + struct.pack('>L', len(track_data)) + track_data)


def wrap_riff(smf, extra_chunks=b''):
    data_chunk = b'data' + struct.pack('<L', len(smf)) + smf
    if len(smf) & 1:
        data_chunk += b'\x00'
    body = b'RMID' + extra_chunks + data_chunk
    return b'RIFF' + struct.pack('<L', len(body)) + body


def read(data):
    return MidiFile(file=io.BytesIO(data))


def save_bytes(midifile):
    outfile = io.BytesIO()
    midifile.save(file=outfile)
    return outfile.getvalue()


# --- SMPTE time division ---

def test_smpte_time_conversion():
    midifile = read(make_smf(SMPTE_DIVISION, TRACK_EVENTS))
    # 480 ticks at 25 fps * 40 ticks/frame = 1000 ticks/second.
    times = [msg.time for msg in midifile]
    assert times == [0, 0.48, 0]
    assert midifile.length == 0.48


def test_smpte_tempo_is_ignored():
    midifile = read(make_smf(SMPTE_DIVISION, TRACK_EVENTS_WITH_TEMPO))
    times = [msg.time for msg in midifile]
    assert times == [0, 0, 0.48, 0]
    assert midifile.length == 0.48


def test_smpte_attributes():
    midifile = read(make_smf(SMPTE_DIVISION, TRACK_EVENTS))
    assert midifile.smpte_fps == 25
    assert midifile.smpte_ticks_per_frame == 40
    assert midifile.ticks_per_beat is None


def test_smpte_frame_rates():
    for byte, fps in [(0xe8, 24), (0xe7, 25), (0xe3, 29), (0xe2, 30)]:
        division = (byte << 8 | 40) - 0x10000
        midifile = read(make_smf(division, TRACK_EVENTS))
        assert midifile.smpte_fps == fps
        assert midifile.smpte_ticks_per_frame == 40


def test_smpte_save_keeps_division():
    smf = make_smf(SMPTE_DIVISION, TRACK_EVENTS)
    saved = save_bytes(read(smf))
    # The division bytes in the header must be identical (0xe728).
    assert saved[12:14] == smf[12:14] == b'\xe7\x28'
    # And the saved file reads back as SMPTE.
    reread = read(saved)
    assert reread.smpte_fps == 25
    assert reread.smpte_ticks_per_frame == 40
    assert reread.ticks_per_beat is None
    assert reread.length == 0.48


def test_invalid_smpte_frame_rate():
    # 0xe4 = -28, not a valid SMPTE frame rate.
    division = (0xe4 << 8 | 40) - 0x10000
    with raises(OSError, match='invalid SMPTE frame rate'):
        read(make_smf(division, TRACK_EVENTS))


# --- RIFF RMID (.rmi) files ---

def test_rmi_read():
    smf = make_smf(PPQN_DIVISION, TRACK_EVENTS)
    midifile = read(wrap_riff(smf))
    assert midifile.ticks_per_beat == PPQN_DIVISION
    assert [msg.bytes() for msg in midifile.tracks[0]] == [
        msg.bytes() for msg in read(smf).tracks[0]]


def test_rmi_read_smpte():
    midifile = read(wrap_riff(make_smf(SMPTE_DIVISION, TRACK_EVENTS)))
    assert midifile.smpte_fps == 25
    assert midifile.length == 0.48


def test_rmi_read_skips_unknown_chunks():
    # An odd-sized chunk before the data chunk, padded to even size.
    info = b'INFO' + struct.pack('<L', 3) + b'abc\x00'
    smf = make_smf(PPQN_DIVISION, TRACK_EVENTS)
    midifile = read(wrap_riff(smf, extra_chunks=info))
    assert midifile.ticks_per_beat == PPQN_DIVISION
    assert len(midifile.tracks) == 1


def test_rmi_save_keeps_riff_wrapper():
    smf = make_smf(PPQN_DIVISION, TRACK_EVENTS)
    saved = save_bytes(read(wrap_riff(smf)))
    assert saved[:4] == b'RIFF'
    assert saved[8:12] == b'RMID'
    # The saved file must be readable as an .rmi file again.
    reread = read(saved)
    assert reread.ticks_per_beat == PPQN_DIVISION
    assert [msg.bytes() for msg in reread.tracks[0]] == [
        msg.bytes() for msg in read(smf).tracks[0]]


def test_rmi_not_midi_form():
    smf = make_smf(PPQN_DIVISION, TRACK_EVENTS)
    data = wrap_riff(smf).replace(b'RMID', b'WAVE')
    with raises(OSError, match='not a RIFF MIDI file'):
        read(data)


# --- PPQN behavior is unchanged ---

def test_ppqn_attributes():
    midifile = read(make_smf(PPQN_DIVISION, TRACK_EVENTS))
    assert midifile.ticks_per_beat == PPQN_DIVISION
    assert midifile.smpte_fps is None
    assert midifile.smpte_ticks_per_frame is None


def test_ppqn_timing_uses_tempo():
    # 480 ticks at 480 ticks/beat and the default tempo of
    # 500000 us/beat is 0.5 seconds.
    midifile = read(make_smf(PPQN_DIVISION, TRACK_EVENTS))
    assert [msg.time for msg in midifile] == [0, 0.5, 0]
    assert midifile.length == 0.5

    # A set_tempo of 1000000 us/beat doubles that.
    midifile = read(make_smf(PPQN_DIVISION, TRACK_EVENTS_WITH_TEMPO))
    assert [msg.time for msg in midifile] == [0, 0, 1.0, 0]
    assert midifile.length == 1.0


def test_ppqn_save_keeps_division_and_no_riff():
    smf = make_smf(PPQN_DIVISION, TRACK_EVENTS)
    saved = save_bytes(read(smf))
    assert saved[:4] == b'MThd'
    assert saved == smf


def test_new_file_is_ppqn():
    midifile = MidiFile()
    assert midifile.ticks_per_beat == PPQN_DIVISION
    assert midifile.smpte_fps is None
    assert midifile.smpte_ticks_per_frame is None
