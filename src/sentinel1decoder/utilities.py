import struct
from typing import Union

import numpy as np
import pandas as pd

from sentinel1decoder import _field_names as fn
from sentinel1decoder.enums import RangeDecimation


def rename_packet_metadata_columns_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename parsed column names to raw (spec-style) names.

    Only renames columns that are present and have a raw counterpart.
    No-op if DataFrame already has raw names; returns input unchanged.
    Does not convert any column data types, only renames the columns.

    Args:
        df: DataFrame with parsed or mixed column names.

    Returns:
        DataFrame with raw column names where applicable.
    """
    rename_map = {old: new for old, new in fn.DECODED_TO_RAW_NAME.items() if old in df.columns}
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def rename_packet_metadata_columns_to_parsed(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw (spec-style) column names to human-readable parsed names.

    Only renames columns that are present and have a parsed counterpart.
    No-op if DataFrame already has parsed names; returns input unchanged.
    Does not convert any column data types, only renames the columns.

    Args:
        df: DataFrame with raw or mixed column names.

    Returns:
        DataFrame with parsed column names where applicable.
    """
    rename_map = {old: new for old, new in fn.RAW_TO_DECODED_NAME.items() if old in df.columns}
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def range_dec_to_sample_rate(rgdec: Union[int, RangeDecimation]) -> float:
    """
    Convert range decimation code to sample rate.

    Args:
        rgdec: Range decimation code (int) or RangeDecimation enum member.

    Returns:
        Sample rate for this range decimation code in Hz.

    """
    if isinstance(rgdec, int):
        rgdec = RangeDecimation(rgdec)
    return rgdec.sample_rate_hz


def read_subcommed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode the sub-commutated satellite ephemeris data present in the file.

    Args:
        df: Pandas dataframe containing parsed packet headers, including ancillary
            word indices, ancillary words and Space Packet Count. Row labels do
            not establish packet continuity.

    Returns:
        A pandas dataframe containing the decoded sub-commutated ancillary data
        words. Incomplete cycles and cycles with invalid or discontinuous packet
        counts are skipped. Counts must advance by one modulo 2**32.

    Raises:
        ValueError: If Space Packet Count is absent, preventing verification of
            cycle continuity.
    """
    if df.index.nlevels > 1:
        df = df.droplevel(fn.ACQUISITION_CHUNK_NUM_DECODED)

    index_col = fn.SUBCOM_ANC_DATA_WORD_INDEX_DECODED
    data_col = fn.SUBCOM_ANC_DATA_WORD_DECODED
    counter_col = fn.SPACE_PACKET_COUNT_DECODED
    if counter_col not in df.columns:
        raise ValueError("Space Packet Count is required to verify ancillary cycle continuity")
    # Packet labels can start above zero when callers pass a metadata slice.
    start_positions = np.flatnonzero(df[index_col].eq(1).fillna(False).to_numpy(dtype=bool))
    output_dict_list = []

    dbl_type = np.dtype(np.float64).newbyteorder(">")
    sgl_type = np.dtype(np.float32).newbyteorder(">")
    for i in start_positions:

        # check our index is followed by a continuous block of 64
        if len(df) - i >= 64:
            block = df.iloc[i : i + 64]
            if block[index_col].eq(list(range(1, 65))).fillna(False).all():
                # Issue 13, section 3.2.4.1: SPCT advances once per generated
                # packet. Ancillary indices alone cannot detect loss of 64 words.
                counter = block[counter_col]
                if counter.isna().any() or not counter.between(0, 2**32 - 1).all():
                    continue
                counts = counter.to_numpy(dtype=np.int64)
                if not counter.eq(counts).all() or np.any(np.diff(counts) % 2**32 != 1):
                    continue
                d = block[data_col].tolist()
                x_bytes = struct.pack(">HHHH", d[0], d[1], d[2], d[3])
                y_bytes = struct.pack(">HHHH", d[4], d[5], d[6], d[7])
                z_bytes = struct.pack(">HHHH", d[8], d[9], d[10], d[11])
                x = np.frombuffer(x_bytes, dtype=dbl_type)[0]
                y = np.frombuffer(y_bytes, dtype=dbl_type)[0]
                z = np.frombuffer(z_bytes, dtype=dbl_type)[0]

                x_vel_bytes = struct.pack(">HH", d[12], d[13])
                y_vel_bytes = struct.pack(">HH", d[14], d[15])
                z_vel_bytes = struct.pack(">HH", d[16], d[17])
                x_vel = np.frombuffer(x_vel_bytes, dtype=sgl_type)[0]
                y_vel = np.frombuffer(y_vel_bytes, dtype=sgl_type)[0]
                z_vel = np.frombuffer(z_vel_bytes, dtype=sgl_type)[0]

                # Table 3.2-7: the high byte of the first time word is unused.
                pvt_t1 = (d[18] & 0xFF) * 2**24
                pvt_t2 = d[19] * 2**8
                pvt_t3 = d[20] * 2**-8
                pvt_t4 = d[21] * 2**-24
                pvt_t = pvt_t1 + pvt_t2 + pvt_t3 + pvt_t4

                output_dictionary = {
                    fn.X_POS_DECODED: x,
                    fn.Y_POS_DECODED: y,
                    fn.Z_POS_DECODED: z,
                    fn.X_VEL_DECODED: x_vel,
                    fn.Y_VEL_DECODED: y_vel,
                    fn.Z_VEL_DECODED: z_vel,
                    fn.POD_SOLN_DATA_TIMESTAMP_DECODED: pvt_t,
                }

                q0_bytes = struct.pack(">HH", d[22], d[23])
                q1_bytes = struct.pack(">HH", d[24], d[25])
                q2_bytes = struct.pack(">HH", d[26], d[27])
                q3_bytes = struct.pack(">HH", d[28], d[29])
                q0 = np.frombuffer(q0_bytes, dtype=sgl_type)[0]
                q1 = np.frombuffer(q1_bytes, dtype=sgl_type)[0]
                q2 = np.frombuffer(q2_bytes, dtype=sgl_type)[0]
                q3 = np.frombuffer(q3_bytes, dtype=sgl_type)[0]

                x_ang_rate_bytes = struct.pack(">HH", d[30], d[31])
                y_ang_rate_bytes = struct.pack(">HH", d[32], d[33])
                z_ang_rate_bytes = struct.pack(">HH", d[34], d[35])
                x_ang_rate = np.frombuffer(x_ang_rate_bytes, dtype=sgl_type)[0]
                y_ang_rate = np.frombuffer(y_ang_rate_bytes, dtype=sgl_type)[0]
                z_ang_rate = np.frombuffer(z_ang_rate_bytes, dtype=sgl_type)[0]

                att_t1 = (d[36] & 0xFF) * 2**24
                att_t2 = d[37] * 2**8
                att_t3 = d[38] * 2**-8
                att_t4 = d[39] * 2**-24
                att_t = att_t1 + att_t2 + att_t3 + att_t4

                output_dictionary.update(
                    {
                        fn.Q0_DECODED: q0,
                        fn.Q1_DECODED: q1,
                        fn.Q2_DECODED: q2,
                        fn.Q3_DECODED: q3,
                        fn.X_ANG_RATE_DECODED: x_ang_rate,
                        fn.Y_ANG_RATE_DECODED: y_ang_rate,
                        fn.Z_ANG_RATE_DECODED: z_ang_rate,
                        fn.ATTITUDE_DATA_TIMESTAMP_DECODED: att_t,
                    }
                )

                output_dict_list.append(output_dictionary)
    out_df = pd.DataFrame(output_dict_list)
    return out_df
