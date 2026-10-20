import struct

import pandas as pd
import pytest

import sentinel1decoder.constants as cnst
from sentinel1decoder import _field_names as fn
from sentinel1decoder.utilities import (
    range_dec_to_sample_rate,
    read_subcommed_data,
    rename_packet_metadata_columns_to_parsed,
    rename_packet_metadata_columns_to_raw,
)


def test_range_dec_to_sample_rate() -> None:
    base_sample_freq = 4 * cnst.F_REF
    assert range_dec_to_sample_rate(0) == (3 / 4) * base_sample_freq
    assert range_dec_to_sample_rate(1) == (2 / 3) * base_sample_freq
    # 2 is unused
    assert range_dec_to_sample_rate(3) == (5 / 9) * base_sample_freq
    assert range_dec_to_sample_rate(4) == (4 / 9) * base_sample_freq
    assert range_dec_to_sample_rate(5) == (3 / 8) * base_sample_freq
    assert range_dec_to_sample_rate(6) == (1 / 3) * base_sample_freq
    assert range_dec_to_sample_rate(7) == (1 / 6) * base_sample_freq
    assert range_dec_to_sample_rate(8) == (3 / 7) * base_sample_freq
    assert range_dec_to_sample_rate(9) == (5 / 16) * base_sample_freq
    assert range_dec_to_sample_rate(10) == (3 / 26) * base_sample_freq
    assert range_dec_to_sample_rate(11) == (4 / 11) * base_sample_freq

    with pytest.raises(Exception):
        range_dec_to_sample_rate(2)
    with pytest.raises(Exception):
        range_dec_to_sample_rate(12)
    with pytest.raises(Exception):
        range_dec_to_sample_rate(-1)


def test_rename_packet_metadata_columns_to_raw() -> None:
    """Rename parsed columns to raw; no-op when already raw."""
    df_decoded = pd.DataFrame({"ECC Number": [1], "BAQ Mode": [12]})
    df_raw = rename_packet_metadata_columns_to_raw(df_decoded)
    assert "ECC" in df_raw.columns
    assert "BAQMOD" in df_raw.columns
    assert "ECC Number" not in df_raw.columns

    # No-op when already raw
    df_raw2 = rename_packet_metadata_columns_to_raw(df_raw)
    assert df_raw2 is df_raw
    assert list(df_raw2.columns) == ["ECC", "BAQMOD"]


def test_rename_packet_metadata_columns_to_parsed() -> None:
    """Rename raw columns to parsed; no-op when already parsed."""
    df_raw = pd.DataFrame({"ECC": [1], "BAQMOD": [12]})
    df_decoded = rename_packet_metadata_columns_to_parsed(df_raw)
    assert "ECC Number" in df_decoded.columns
    assert "BAQ Mode" in df_decoded.columns
    assert "ECC" not in df_decoded.columns

    # No-op when already decoded
    df_decoded2 = rename_packet_metadata_columns_to_parsed(df_decoded)
    assert df_decoded2 is df_decoded


def _ephemeris_metadata(
    packet_start: int = 0,
    unused_time_byte: int = 0,
    *,
    packet_count_start: int = 0,
    position: tuple[float, float, float] = (7000000.0, 1000.0, -2000.0),
    velocity: tuple[float, float, float] = (1.0, 7500.0, -2.0),
) -> pd.DataFrame:
    """One complete ancillary cycle, using Issue 13 tables 3.2-5 through 3.2-7."""
    words = list(struct.unpack(">12H", struct.pack(">ddd", *position)))
    words.extend(struct.unpack(">6H", struct.pack(">fff", *velocity)))
    words.extend([(unused_time_byte << 8) | 0x12, 0x3456, 0x789A, 0xBCDE])
    words.extend(struct.unpack(">14H", struct.pack(">fffffff", 1.0, 0.0, 0.0, 0.0, 0.5, -0.25, 0.125)))
    words.extend([(unused_time_byte << 8) | 0x23, 0x4567, 0x89AB, 0xCDEF])
    words.extend([0] * 24)
    return pd.DataFrame(
        {
            fn.SUBCOM_ANC_DATA_WORD_INDEX_DECODED: pd.array(range(1, 65), dtype="UInt8"),
            fn.SUBCOM_ANC_DATA_WORD_DECODED: pd.array(words, dtype="UInt16"),
            fn.SPACE_PACKET_COUNT_DECODED: pd.array(
                [(packet_count_start + i) % 2**32 for i in range(64)], dtype="UInt32"
            ),
        },
        index=pd.RangeIndex(packet_start, packet_start + 64, name=fn.PACKET_NUM_DECODED),
    )


@pytest.mark.parametrize("packet_start", [0, 1000])
@pytest.mark.parametrize("use_chunk_index", [False, True])
def test_read_subcommed_data_uses_row_positions(packet_start: int, use_chunk_index: bool) -> None:
    """A complete metadata slice must decode even when its packet labels start above zero."""
    metadata = _ephemeris_metadata(packet_start)
    if use_chunk_index:
        metadata.index = pd.MultiIndex.from_arrays(
            [[7] * len(metadata), metadata.index],
            names=[fn.ACQUISITION_CHUNK_NUM_DECODED, fn.PACKET_NUM_DECODED],
        )

    decoded = read_subcommed_data(metadata)

    assert len(decoded) == 1
    row = decoded.iloc[0]
    assert [row[fn.X_POS_DECODED], row[fn.Y_POS_DECODED], row[fn.Z_POS_DECODED]] == [7000000.0, 1000.0, -2000.0]
    assert [row[fn.X_VEL_DECODED], row[fn.Y_VEL_DECODED], row[fn.Z_VEL_DECODED]] == [1.0, 7500.0, -2.0]
    assert [row[fn.Q0_DECODED], row[fn.Q1_DECODED], row[fn.Q2_DECODED], row[fn.Q3_DECODED]] == [1.0, 0.0, 0.0, 0.0]
    assert [row[fn.X_ANG_RATE_DECODED], row[fn.Y_ANG_RATE_DECODED], row[fn.Z_ANG_RATE_DECODED]] == [0.5, -0.25, 0.125]
    assert row[fn.POD_SOLN_DATA_TIMESTAMP_DECODED] == 0x12345678 + 0x9ABCDE / 2**24
    assert row[fn.ATTITUDE_DATA_TIMESTAMP_DECODED] == 0x23456789 + 0xABCDEF / 2**24


@pytest.mark.parametrize("unused_time_byte", [1, 0xA5, 0xFF])
def test_subcommed_timestamps_ignore_unused_high_byte(unused_time_byte: int) -> None:
    """Reserved bits must not add multiples of 2**32 seconds to either timestamp."""
    metadata = _ephemeris_metadata(unused_time_byte=unused_time_byte)
    decoded = read_subcommed_data(metadata)

    assert decoded[fn.POD_SOLN_DATA_TIMESTAMP_DECODED].iloc[0] == 0x12345678 + 0x9ABCDE / 2**24
    assert decoded[fn.ATTITUDE_DATA_TIMESTAMP_DECODED].iloc[0] == 0x23456789 + 0xABCDEF / 2**24


def test_read_subcommed_data_skips_incomplete_cycle() -> None:
    assert read_subcommed_data(_ephemeris_metadata(1000).iloc[:-1]).empty


@pytest.mark.parametrize("bad_index", [0, pd.NA])
def test_read_subcommed_data_skips_broken_word_sequence(bad_index: object) -> None:
    metadata = _ephemeris_metadata(1000)
    metadata.loc[1032, fn.SUBCOM_ANC_DATA_WORD_INDEX_DECODED] = bad_index

    assert read_subcommed_data(metadata).empty


def test_read_subcommed_data_requires_space_packet_count() -> None:
    metadata = _ephemeris_metadata().drop(columns=fn.SPACE_PACKET_COUNT_DECODED)

    with pytest.raises(ValueError, match="Space Packet Count"):
        read_subcommed_data(metadata)


@pytest.mark.parametrize("packet_count_start", [0, 1000, 2**32 - 32, 2**32 - 1])
def test_read_subcommed_data_accepts_counter_wrap_and_arbitrary_row_labels(packet_count_start: int) -> None:
    metadata = _ephemeris_metadata(packet_count_start=packet_count_start)
    metadata.index = pd.Index([f"row-{3 * i}" for i in range(64)])

    assert len(read_subcommed_data(metadata)) == 1


@pytest.mark.parametrize("bad_count", [pd.NA, -1, 2**32, 1000.5])
def test_read_subcommed_data_skips_invalid_packet_count(bad_count: object) -> None:
    metadata = _ephemeris_metadata(packet_count_start=1000)
    metadata[fn.SPACE_PACKET_COUNT_DECODED] = metadata[fn.SPACE_PACKET_COUNT_DECODED].astype(object)
    metadata.iloc[32, metadata.columns.get_loc(fn.SPACE_PACKET_COUNT_DECODED)] = bad_count

    assert read_subcommed_data(metadata).empty


@pytest.mark.parametrize("change", ["gap", "duplicate", "reorder"])
def test_read_subcommed_data_skips_discontinuous_counter(change: str) -> None:
    metadata = _ephemeris_metadata(packet_count_start=1000)
    counter = metadata.columns.get_loc(fn.SPACE_PACKET_COUNT_DECODED)
    if change == "gap":
        metadata.iloc[32:, counter] += 64
    elif change == "duplicate":
        metadata.iloc[32, counter] = metadata.iloc[31, counter]
    else:
        metadata.iloc[[31, 32], counter] = metadata.iloc[[32, 31], counter].to_numpy()

    assert read_subcommed_data(metadata).empty


def test_read_subcommed_data_accepts_complete_cycles_separated_by_a_gap() -> None:
    first = _ephemeris_metadata(1000, packet_count_start=1000)
    later = _ephemeris_metadata(2000, packet_count_start=2000, position=(7100000.0, 2000.0, -3000.0))

    decoded = read_subcommed_data(pd.concat([first, later]))

    assert decoded[fn.X_POS_DECODED].tolist() == [7000000.0, 7100000.0]


def test_read_subcommed_data_rejects_mixed_cycle_and_keeps_later_complete_cycle() -> None:
    first = _ephemeris_metadata(1000, packet_count_start=1000)
    later = _ephemeris_metadata(
        1064, packet_count_start=1064, position=(7100000.0, 2000.0, -3000.0), velocity=(-10.0, 7400.0, 25.0)
    )
    # Give the later set a timestamp one second later. Losing 64 consecutive
    # packets after word 12 would join first's position to later's velocity/time
    # while its ancillary word indices still appear to be exactly 1 through 64.
    data = later.columns.get_loc(fn.SUBCOM_ANC_DATA_WORD_DECODED)
    later.iloc[20, data] += 0x100
    spliced = pd.concat([first.iloc[:12], later.iloc[12:]])
    assert spliced[fn.SUBCOM_ANC_DATA_WORD_INDEX_DECODED].tolist() == list(range(1, 65))
    assert spliced[fn.SPACE_PACKET_COUNT_DECODED].iloc[11:13].tolist() == [1011, 1076]

    assert read_subcommed_data(spliced).empty

    complete = _ephemeris_metadata(
        1128, packet_count_start=1128, position=(7200000.0, 3000.0, -4000.0), velocity=(-20.0, 7300.0, 30.0)
    )
    decoded = read_subcommed_data(pd.concat([spliced, complete]))
    assert len(decoded) == 1
    row = decoded.iloc[0]
    assert [row[fn.X_POS_DECODED], row[fn.Y_POS_DECODED], row[fn.Z_POS_DECODED]] == [7200000.0, 3000.0, -4000.0]
    assert [row[fn.X_VEL_DECODED], row[fn.Y_VEL_DECODED], row[fn.Z_VEL_DECODED]] == [-20.0, 7300.0, 30.0]
