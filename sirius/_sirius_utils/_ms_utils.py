#   Copyright 2019 AUI, Inc. Washington DC, USA
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import xarray as xr
import logging
import numpy as np
import time
import copy

def _calc_optimal_ms_chunk_shape(memory_available_in_bytes,shape,element_size_in_bytes,column_name):
    '''
    Calculates the max number of rows (1st dim in shape) of a variable that can be fit in the memory for a thread.
    '''
    factor = 0.8 #Account for memory used by other objects in thread.
    total_mem = np.prod(shape)*element_size_in_bytes
    single_row_mem = np.prod(shape[1:])*element_size_in_bytes

    try:
        assert single_row_mem < factor*memory_available_in_bytes
    except AssertionError as err:
        logging.exception('Not engough memory in a thread to conatain a row of ' + column_name + '. Need at least ' + str(single_row_mem/factorfactor) + ' bytes.')
        raise err

    rows_chunk_size = int((factor*memory_available_in_bytes)/single_row_mem)

    if rows_chunk_size > shape[0]:
        rows_chunk_size = shape[0]

    logging.debug('Numbers of rows in chunk for ' + column_name + ': ' + str(rows_chunk_size))

    return rows_chunk_size
    

def _sub_table_xds(mxds,
        time_xda,
        chan_xda,
        pol,
        tel_xds,
        phase_center_names,
        phase_center_ra_dec,
        auto_corr,):
    ###############################
    #Create SPECTRAL_WINDOW
    ###############################

    spw_xds = xr.Dataset({
            "FREQ_GROUP":(("row"), np.zeros(shape=1).astype("int")),
            "FLAG_ROW":(("row"), np.zeros(shape=1).astype("bool")),
            "NET_SIDEBAND":(("row"), np.ones(shape=1).astype("int")),
            # if only everything were consistently indexed...
            # maybe it would be better to use chan_xda.spw_name but that might break something downstream
            "FREQ_GROUP_NAME":(
                ("row"),
                np.array([chan_xda.spw_name]),
            ),
            # NB: a naive chan_xda.sum() is high by an order of magnitude!
            "TOTAL_BANDWIDTH":(
                ("row"),
                np.asarray([chan_xda.freq_delta * chan_xda.size]),
            ),
            # "frequency representative of this spw, usually the sky frequency corresponding to the DC edge of the baseband."
            # until "reference" in chan.xda.attrs use 1st channel
            "REF_FREQUENCY":(("row"), np.take(chan_xda.data, [0])),
            # obscure measures tool keyword for Doppler tracking
            "MEAS_FREQ_REF":(("row"), np.ones(shape=1).astype("int")),
            # "Identiﬁcation of the electronic signal path for the case of multiple (simultaneous) IFs.
            # (e.g. VLA: AC=0, BD=1, ATCA: Freq1=0, Freq2=1)"
            "IF_CONV_CHAIN":(("row"), np.zeros(shape=1).astype("int")),
            "NAME":(("row"), np.array([chan_xda.spw_name])),
            "NUM_CHAN":(("row"), np.array([chan_xda.size]).astype("int")),
            # the following share shape (1,chans)
            # "it is more efficient to keep a separate reference to this information"
            "CHAN_WIDTH":(
                ("row", "chan"),
                np.broadcast_to([chan_xda.freq_delta], shape=(1, chan_xda.size)).astype(
                    "float"
                ),
            ),
            # the assumption that input channel frequencies are central will hold for a while
            "CHAN_FREQ":(
                ("row", "chan"),
                np.broadcast_to(
                    np.asarray(chan_xda.data), shape=(1, chan_xda.size)
                ).astype("float"),
            ),
            "RESOLUTION":(
                ("row", "chan"),
                np.broadcast_to(
                    # note that this is not what we call chan.xda.freq_resolution
                    [chan_xda.freq_delta],
                    shape=(1, chan_xda.size),
                ).astype("float"),
            ),
            # we may eventually want to infer this by instrument, e.g., ALMA correlator binning
            # but until "effective_bw" in chan_xda.attrs,
            "EFFECTIVE_BW":(
                ("row", "chan"),
                np.broadcast_to([chan_xda.freq_delta], shape=(1, chan_xda.size)).astype(
                    "float"
                ),
            ),
            }
        )
        
    
    from sirius_data._ms_column_descriptions_dicts import spectral_window_column_description
    spw_xds.attrs['column_descriptions'] = copy.deepcopy(spectral_window_column_description)
    spw_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    spw_xds.attrs['bad_cols'] = []

    mxds.attrs['SPECTRAL_WINDOW'] = spw_xds
    
    ###############################
    #Create POLARIZATION
    ###############################
    # POLARIZATION
    # Surely there is a more elegant way to build this strange index
    pol_index = []
    for pp in pol:
        if pp == 5 or pp == 9:
            pol_index.append([0, 0])
        if pp == 6 or pp == 10:
            pol_index.append([0, 1])
        if pp == 7 or pp == 11:
            pol_index.append([1, 0])
        if pp == 8 or pp == 12:
            pol_index.append([1, 1])
    
    pol_xds = xr.Dataset({
            "NUM_CORR":(("row"), np.asarray([len(pol)], dtype="int")),
            "CORR_TYPE":(("row", "corr"), np.asarray([pol], dtype="int")),
            "FLAG_ROW":(("row"), np.zeros(shape=1).astype("bool")),
            # "Pair of integers for each correlation product, specifying the receptors from which the signal originated."
            "CORR_PRODUCT":(
                ("row", "corr", "corrprod_idx"),
                np.asarray([pol_index], dtype="int"),
            ),
        }
    )
    
    from sirius_data._ms_column_descriptions_dicts import polarization_column_description
    pol_xds.attrs['column_descriptions'] = copy.deepcopy(polarization_column_description)
    pol_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    pol_xds.attrs['bad_cols'] = []
    
    mxds.attrs['POLARIZATION'] = pol_xds
    
    ###############################
    #DATA_DESCRIPTION
    ###############################

    ddi_xds = xr.Dataset(
         {
            # this function operates on a single DDI at once, so this should reduce to length-1 arrays = 0
            # we could also enumerate the ds list if we were reading from existing MS and pass the index
            "SPECTRAL_WINDOW_ID":(("row"), np.zeros(1, dtype="int")),
            "FLAG_ROW":(("row"), np.zeros(1, dtype="bool")),
            "POLARIZATION_ID":(("row"), np.zeros(1, dtype="int")),
        },
    )
    
    from sirius_data._ms_column_descriptions_dicts import data_description_column_description
    ddi_xds.attrs['column_descriptions'] = copy.deepcopy(data_description_column_description)
    ddi_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    ddi_xds.attrs['bad_cols'] = []
    
    mxds.attrs['DATA_DESCRIPTION'] = ddi_xds
    
    
    ###############################
    #ANTENNA
    ###############################
    ant_xds = xr.Dataset(
            {
            "NAME":(("row"),tel_xds.ant_name.data),
            "DISH_DIAMETER":(("row"), tel_xds.DISH_DIAMETER.data),
            "POSITION":(("row", "xyz"), tel_xds.ANT_POS.data),
            # not yet supporting space-based interferometers
            "TYPE":(
                ("row"),
                np.full(tel_xds.ant_name.shape, "GROUND-BASED", dtype="<U12"),
            ),
            "FLAG_ROW":(("row"), np.zeros(tel_xds.ant_name.shape, dtype="bool")),
            # when this input is available from tel.zarr then we can infer it, til then assume alt-az
            "MOUNT":(("row"), np.full(tel_xds.ant_name.shape, "alt-az", dtype="<U6")),
            # likewise, although this seems like it should be pulled from the cfg files
            "STATION":(("row"), np.full(tel_xds.ant_name.shape, "P", dtype="<U1")),
            # until we have some input with OFFSET specified, no conditional
            "OFFSET":(
                ("row", "xyz"),
                np.zeros((tel_xds.dims["ant_name"], 3), dtype=np.float),
            ),
        }
    )
    
    from sirius_data._ms_column_descriptions_dicts import antenna_column_description
    ant_xds.attrs['column_descriptions'] = copy.deepcopy(antenna_column_description)
    ant_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    ant_xds.attrs['bad_cols'] = []
    
    mxds.attrs['ANTENNA'] = ant_xds
    
    ###############################
    # Feed
    ###############################
    if np.all(np.isin(pol, [5, 6, 7, 8])):
        poltype_arr = np.broadcast_to(
            np.asarray(["R", "L"]), (tel_xds.ant_name.size, 2)
            )
    elif np.all(np.isin(pol, [9, 10, 11, 12])):
        # it's clunky to assume linear feeds...
        poltype_arr = np.broadcast_to(
            np.asarray(["X", "Y"]), (tel_xds.ant_name.size, 2)
        )
        
    feed_xds = xr.Dataset(
        data_vars=dict(
            ANTENNA_ID=(("row"), np.arange(0, tel_xds.dims["ant_name"], dtype="int")),
            # -1 fill value indicates that we're not using the optional BEAM subtable
            BEAM_ID=(("row"), np.ones(tel_xds.ant_name.shape, dtype="int") * -1),
            INTERVAL=(
                ("row"),
                np.full(tel_xds.dims["ant_name"], fill_value=1e30, dtype="float"),
            ),
            # we're not supporting offset feeds yet
            POSITION=(
                ("row", "xyz"),
                np.zeros((tel_xds.dims["ant_name"], 3), dtype=np.float),
            ),
            # indexed from FEEDn in the MAIN table
            FEED_ID=(("row"), np.zeros(tel_xds.dims["ant_name"], dtype="int")),
            # "Polarization reference angle. Converts into parallactic angle in the sky domain."
            RECEPTOR_ANGLE=(
                ("row", "receptors"),
                np.zeros((tel_xds.dims["ant_name"], poltype_arr.shape[1])),
            ),
            # "Polarization response at the center of the beam for this feed expressed
            # in a linearly polarized basis (e→x,e→y) using the IEEE convention."
            # practically, broadcast a POLxPOL complex identity matrix along a new N_antenna dim
            POL_RESPONSE=(
                ("row", "receptors", "receptors-2"),
                np.broadcast_to(
                    np.eye(poltype_arr.shape[1], dtype="complex"),
                    (tel_xds.dims["ant_name"], poltype_arr.shape[1], poltype_arr.shape[1]),
                ),
            ),
            # A value of -1 indicates the row is valid for all spectral windows
            SPECTRAL_WINDOW_ID=(
                ("row"),
                np.ones(tel_xds.dims["ant_name"], dtype="int") * -1,
            ),
            NUM_RECEPTORS=(
                ("row"),
                np.full(tel_xds.dims["ant_name"], fill_value=poltype_arr.shape[1], dtype="int"),
            ),
            POLARIZATION_TYPE=(("row", "receptors"), poltype_arr),
            # "the same measure reference used for the TIME column of the MAIN table must be used"
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            TIME=(("row"), np.zeros(tel_xds.dims["ant_name"], dtype="float")),
            # "Beam position oﬀset, as deﬁned on the sky but in the antenna reference frame."
            # the third dimension size could also be taken from phase_center_ra_dec in theory
            BEAM_OFFSET=(
                ("row", "receptors", "radec"),
                np.zeros(shape=(tel_xds.dims["ant_name"], poltype_arr.shape[1], 2), dtype="float"),
            ),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import feed_column_description
    feed_xds.attrs['column_descriptions'] = copy.deepcopy(feed_column_description)
    feed_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    feed_xds.attrs['bad_cols'] = []
    
    mxds.attrs['FEED'] = feed_xds
    
    ###############################
    # Field
    ###############################
    field_xds = xr.Dataset(
        data_vars=dict(
            NAME=(("row"), np.array(phase_center_names)),
            SOURCE_ID=(("row"), np.indices(phase_center_names.shape)[0]),
            # may need to wrap the RA at 180deg to make the MS happy
            REFERENCE_DIR=(
                ("row", "field-poly", "field-dir"),
                # expand_dims was added to dask.array in version 2022.02.0
                np.expand_dims(np.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            PHASE_DIR=(
                ("row", "field-poly", "field-dir"),
                np.expand_dims(np.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            DELAY_DIR=(
                ("row", "field-poly", "field-dir"),
                np.expand_dims(np.array(phase_center_ra_dec, dtype="double"), axis=0),
            ),
            CODE=(
                ("row"),
                np.full(phase_center_names.shape, fill_value="", dtype="<U1"),
            ),
            # "Required to use the same TIME Measure reference as in MAIN."
            # in practice this appears to be 0 since the conversion to casa-expected epoch is done
            TIME=(("row"), np.zeros(phase_center_names.shape, dtype="float")),
            FLAG_ROW=(("row"), np.zeros(phase_center_names.shape, dtype="bool")),
            # Series order for the *_DIR columns
            NUM_POLY=(("row"), np.zeros(phase_center_names.shape, dtype="int")),
        ),
    )
    
    
    from sirius_data._ms_column_descriptions_dicts import field_column_description
    field_xds.attrs['column_descriptions'] = copy.deepcopy(field_column_description)
    field_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    field_xds.attrs['bad_cols'] = []
    
    mxds.attrs['FIELD'] = field_xds
    
    ###############################
    # History
    ###############################
    his_xds = xr.Dataset(
        data_vars=dict(
            MESSAGE=(
                ("row"),
                np.array(["taskname=sirius.dio.write_to_ms"]),
            ),
            APPLICATION=(("row"), np.array(["ms"])),
            # "Required to have the same TIME Measure reference as used in MAIN."
            # but unlike some subtables with ^that^ in the spec, this is actual timestamps
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            TIME=(
                ("row"),
                (np.array([time.time()], dtype="float") / 10**3 + 3506716800.0),
            ),
            PRIORITY=(("row"), np.array(["NORMAL"])),
            ORIGIN=(("row"), np.array(["dask-ms"])),
            OBJECT_ID=(("row"), np.array([0], dtype="int")),
            OBSERVATION_ID=(("row"), np.array([-1], dtype="int")),
            # The MSv2 spec says there is "an adopted project-wide format."
            # which is big if true... appears to have shape expand_dims(MESSAGE)
            APP_PARAMS=(
                ("row", "APP_PARAMS-1"),
                np.array([[""], [""]]).transpose(),
            ),
            CLI_COMMAND=(
                ("row", "CLI_COMMAND-1"),
                np.array([[""], [""]]).transpose(),
            ),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import history_column_description
    his_xds.attrs['column_descriptions'] = copy.deepcopy(history_column_description)
    his_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    his_xds.attrs['bad_cols'] = []
    
    mxds.attrs['HISTORY'] = his_xds
    
    ###############################
    # Observation
    ###############################

    obs_xds = xr.Dataset(
        data_vars=dict(
            TELESCOPE_NAME=(
                ("row"),
                np.array([tel_xds.telescope_name]),
            ),
            RELEASE_DATE=(("row"), np.zeros(1, dtype="float")),
            SCHEDULE_TYPE=(("row"), np.array([""])),
            PROJECT=(("row"), np.array(["SiRIUS simulation"])),
            # first and last value
            TIME_RANGE=(
                ("row", "obs-exts"),
                np.array([np.take(time_xda.data, [0, -1]).astype(np.datetime64).astype("float") / 10**3 + 3506716800.0]),
            ),
            # could try to be clever about this to get uname w/ os or psutil
            OBSERVER=(("row"), np.array(["SiRIUS"])),
            FLAG_ROW=(("row"), np.zeros(1, dtype="bool")),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import observation_column_description
    obs_xds.attrs['column_descriptions'] = copy.deepcopy(observation_column_description)
    obs_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    obs_xds.attrs['bad_cols'] = []
    
    mxds.attrs['OBSERVATION'] = obs_xds

    ###############################
    # Pointing
    ###############################
    ''' IGNORE for NOW
    pnt_xds = xr.Dataset(
        data_vars=dict(
            # is this general enough for the case where phase_center_ra_dec has size > 1 ?
            TARGET=(
                ("row", "point-poly", "radec"),
                np.broadcast_to(
                    np.array(phase_center_ra_dec),
                    shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
                ),
            ),
            # set time origin for polynomial expansions to beginning of the observation
            TIME_ORIGIN=(
                ("row"),
                np.repeat(
                    np.take(times, [0]), repeats=tel_xds.ant_name.size * time_xda.size
                ),
            ),
            INTERVAL=(
                ("row"),
                np.repeat(
                    np.asarray([time_xda.time_delta]),
                    repeats=tel_xds.ant_name.size * time_xda.size,
                ),
            ),
            # True if tracking the nominal pointing position
            TRACKING=(
                ("row"),
                np.ones(shape=tel_xds.ant_name.size * time_xda.size, dtype="bool"),
            ),
            ANTENNA_ID=(
                ("row"),
                np.tile(np.arange(0, tel_xds.ant_name.size), reps=time_xda.size),
            ),
            DIRECTION=(
                ("row", "point-poly", "radec"),
                np.broadcast_to(
                    np.array(phase_center_ra_dec),
                    shape=(tel_xds.ant_name.size * time_xda.size, 1, 2),
                ),
            ),
            # only supporting first order polynomials at present
            NUM_POLY=(
                ("row"),
                np.zeros(shape=tel_xds.ant_name.size * time_xda.size, dtype="int"),
            ),
            # could fill with phase_center_names; the reference implementation is empty
            NAME=(
                ("row"),
                np.full(
                    tel_xds.ant_name.size * time_xda.size, fill_value="", dtype="<U1"
                ),
            ),
            # another different use of this same column name:
            # "Mid-point of the time interval for which the information in this row is valid."
            # NB: difference between Unix origin (1970-01-01) and what CASA expects (1858-11-17) is +/-3506716800 seconds
            TIME=(
                ("row"),
                # must drop from the xr.DataArray to a raw dask.array then make expected shape
                np.repeat(
                    (
                        time_xda.astype(np.datetime64).astype(float) / 10**3
                        + 3506716800.0
                    ).data,
                    repeats=tel_xds.ant_name.size,
                ).rechunk(chunks=tel_xds.ant_name.size * time_xda.size),
            ),
        ),
    )
    
    from sirius_data._ms_column_descriptions_dicts import pointing_column_description
    pnt_xds.attrs['column_descriptions'] = copy.deepcopy(pointing_column_description)
    pnt_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    pnt_xds.attrs['bad_cols'] = []
    
    mxds.attrs['POINTING'] = pnt_xds
    '''
    ###############################
    # State
    ###############################
    
    state_xds = xr.Dataset(
        data_vars=dict(
            FLAG_ROW=(("row"), np.zeros(shape=1).astype("bool")),
            SIG=(("row"), np.ones(shape=1).astype("bool")),
            CAL=(("row"), np.zeros(shape=1).astype("float")),
            # some subset of observing modes e.g., solar will require this
            LOAD=(("row"), np.zeros(shape=1).astype("float")),
            # reference phase if available
            REF=(("row"), np.zeros(shape=1).astype("bool")),
            # relative to SCAN_NUMBER in MAIN, better support TBD
            SUB_SCAN=(("row"), np.zeros(shape=1).astype("int")),
            OBS_MODE=(
                ("row"),
                np.full(
                    shape=1, fill_value="OBSERVE_TARGET.ON_SOURCE", dtype="<U24"
                ),
            ),
        ),
    )

    from sirius_data._ms_column_descriptions_dicts import state_column_description
    state_xds.attrs['column_descriptions'] = copy.deepcopy(state_column_description)
    state_xds.attrs['info'] = {'type': '', 'subType': '', 'readme': ''}
    state_xds.attrs['bad_cols'] = []
    mxds.attrs['STATE'] = state_xds
    
    return mxds
