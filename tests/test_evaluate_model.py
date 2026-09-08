from evaluate_model import slice_range


def test_no_offset_no_limit_returns_everything():
    assert slice_range(list(range(10)), 0, None) == list(range(10))


def test_no_offset_with_limit_matches_old_limit_only_behavior():
    assert slice_range(list(range(10)), 0, 3) == [0, 1, 2]


def test_offset_with_limit_covers_a_contiguous_middle_chunk():
    assert slice_range(list(range(10)), 3, 3) == [3, 4, 5]


def test_offset_past_the_end_returns_empty():
    assert slice_range(list(range(10)), 20, 5) == []


def test_offset_with_limit_running_past_the_end_is_clipped_not_an_error():
    # The last of several chunks covering a sequence that doesn't divide
    # evenly needs this: offset 9 with limit 5 on a 10-item sequence
    # should return the 1 remaining item, not raise or wrap around.
    assert slice_range(list(range(10)), 9, 5) == [9]


def test_consecutive_chunks_cover_the_whole_sequence_exactly_once():
    seq = list(range(23))
    chunk_size = 5
    chunks = [slice_range(seq, offset, chunk_size) for offset in range(0, len(seq), chunk_size)]
    reassembled = [x for chunk in chunks for x in chunk]
    assert reassembled == seq
