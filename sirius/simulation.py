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

# Parallel writes https://github.com/ornladios/ADIOS2
# https://github.com/casacore/casacore/issues/432
# https://github.com/casacore/casacore/issues/729


def simulation(
    point_source_flux,
    point_source_ra_dec,
    pointing_ra_dec,
    phase_center_ra_dec,
    beam_parms,
    beam_models,
    beam_model_map,
    uvw_parms,
    tel_xds,
    time_da,
    chan_da,
    pol,
    a_noise_parms=None,
    uvw_precompute=None,
):
    """
    Simulate a interferometric visibilities and uvw coordinates. Dask enabled function

    Parameters
    ----------
    point_source_flux : np.array
    Returns
    -------
    vis : np.array
    uvw : np.array
    """

    from sirius import calc_vis, calc_uvw, calc_a_noise
    from sirius.make_ant_sky_jones import evaluate_beam_models
    from ._sirius_utils._array_utils import _ndim_list
    from ._parm_utils._check_beam_parms import _check_beam_parms
    import numpy as np
    from itertools import cycle
    import itertools
    import copy
    import dask
    import dask.array as da

    _beam_parms = copy.deepcopy(beam_parms)
    _uvw_parms = copy.deepcopy(uvw_parms)
    assert _check_beam_parms(
        _uvw_parms
    ), "######### ERROR: calc_uvw uvw_parms checking failed."
    assert _check_beam_parms(
        _beam_parms
    ), "######### ERROR: make_ant_sky_jones beam_parms checking failed."

    ### TO DO ###
    # Add checks dims n_time, n_chan, n_ant, n_point_sources are consistant or singleton when allowed.

    n_time = len(time_da)
    n_chan = len(chan_da)
    n_ant = tel_xds.dims["ant"]
    print(n_time, n_chan, n_ant)

    # Check all dims are either 1 or n
    f_pc_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
    f_ps_time = n_time if point_source_ra_dec.shape[0] == 1 else 1
    f_sf_time = n_time if point_source_flux.shape[1] == 1 else 1
    f_sf_chan = n_chan if point_source_flux.shape[2] == 1 else 1

    do_pointing = False
    if pointing_ra_dec is not None:
        do_pointing = True
        f_pt_time = n_time if phase_center_ra_dec.shape[0] == 1 else 1
        f_pt_ant = n_ant if point_source_ra_dec.shape[1] == 1 else 1
    else:
        pointing_ra_dec = np.zeros((2, 2, 2))
        f_pt_time = n_time
        f_pt_ant = n_ant

    n_time_chunks = time_da.numblocks[0]
    n_chan_chunks = chan_da.numblocks[0]

    # Iter over time,chan
    iter_chunks_indx = itertools.product(
        np.arange(n_time_chunks), np.arange(n_chan_chunks)
    )

    vis_list = _ndim_list((n_time_chunks, 1, n_chan_chunks, 1))
    uvw_list = _ndim_list((n_time_chunks, 1, 1))

    n_pol = len(pol)

    from ._sirius_utils._array_utils import _calc_n_baseline

    n_baselines = _calc_n_baseline(n_ant, _uvw_parms["auto_corr"])

    # Build graph
    for c_time, c_chan in iter_chunks_indx:

        time_chunk = time_da.partitions[c_time]
        chan_chunk = chan_da.partitions[c_chan]

        # print(time_da.chunks[0][0])
        s_time = c_time * time_da.chunks[0][0]
        e_time = (
            c_time * time_da.chunks[0][0] + time_da.chunks[0][c_time] - 1
        )  # -1 needed for // to work.
        s_chan = c_chan * chan_da.chunks[0][0]
        e_chan = (
            c_chan * chan_da.chunks[0][0] + chan_da.chunks[0][c_chan] - 1
        )  # -1 needed for // to work.

        # print(s_time_indx,e_time_indx + 1)
        # print(s_time_indx//f_sf_time,e_time_indx//f_sf_time + 1)

        # point_source_flux: np.array [n_point_sources,n_time, n_chan, n_pol] (singleton: n_time, n_chan, n_pol)
        point_source_flux_chunk = point_source_flux[
            :,
            s_time // f_sf_time : e_time // f_sf_time + 1,
            s_chan // f_sf_chan : e_chan // f_sf_chan + 1,
            :,
        ]
        point_source_ra_dec_chunk = point_source_ra_dec[
            s_time // f_ps_time : e_time // f_ps_time + 1, :, :
        ]
        phase_center_ra_dec_chunk = phase_center_ra_dec[
            s_time // f_pc_time : e_time // f_pc_time + 1, :
        ]

        if do_pointing:
            pointing_ra_dec_chunk = pointing_ra_dec[
                s_time // f_pt_time : e_time // f_pt_time + 1, :, :
            ]
        else:
            pointing_ra_dec_chunk = None

        ### TO DO ###
        # Subselect channels for each beam_model with channel axis

        # print('time_chunk',time_chunk, chan_chunk)

        print(c_time, c_chan)
        sim_chunk = dask.delayed(simulation_chunk)(
            dask.delayed(point_source_flux_chunk),
            dask.delayed(point_source_ra_dec_chunk),
            dask.delayed(pointing_ra_dec_chunk),
            dask.delayed(phase_center_ra_dec_chunk),
            dask.delayed(beam_parms),
            beam_models,
            dask.delayed(beam_model_map),
            dask.delayed(uvw_parms),
            tel_xds,
            time_chunk,
            chan_chunk,
            dask.delayed(pol),
            dask.delayed(a_noise_parms),
            dask.delayed(None),
        )
        # sim_chunk.compute()

        vis_list[c_time][0][c_chan][0] = da.from_delayed(
            sim_chunk[0],
            (len(time_chunk), n_baselines, len(chan_chunk), n_pol),
            dtype=np.complex,
        )
        uvw_list[c_time][0][0] = da.from_delayed(
            sim_chunk[1], (len(time_chunk), n_baselines, 3), dtype=np.complex
        )

    vis = da.block(vis_list)
    uvw = da.block(uvw_list)

    return vis, uvw


def simulation_chunk(
    point_source_flux,
    point_source_ra_dec,
    pointing_ra_dec,
    phase_center_ra_dec,
    beam_parms,
    beam_models,
    beam_model_map,
    uvw_parms,
    tel_xds,
    time_str,
    freq_chan,
    pol,
    a_noise_parms=None,
    uvw_precompute=None,
):
    """
    Simulate a interferometric visibilities and uvw coordinates.

    Parameters
    ----------
    point_source_flux : np.array
    Returns
    -------
    vis : np.array
    uvw : np.array
    """

    from sirius import calc_vis, calc_uvw, calc_a_noise
    from sirius.make_ant_sky_jones import evaluate_beam_models
    import numpy as np

    # Calculate uvw coordinates
    if uvw_precompute is None:
        uvw, antenna1, antenna2 = calc_uvw(
            tel_xds, time_str, phase_center_ra_dec, _uvw_parms, check_parms=False
        )
    else:
        from ._sirius_utils._array_utils import _calc_baseline_indx_pair

        n_ant = len(ant_pos)
        antenna1, antenna2 = _calc_baseline_indx_pair(n_ant, _uvw_parms["auto_corr"])
        uvw = uvw_precompute

    # Evaluate zpc files
    eval_beam_models, pa = evaluate_beam_models(
        beam_models,
        _beam_parms,
        freq_chan,
        phase_center_ra_dec,
        time_str,
        uvw_parms["site"],
    )

    print(eval_beam_models)
    #
    # Calculate visibilities
    # shape, point_source_flux, point_source_ra_dec, pointing_ra_dec, phase_center_ra_dec, antenna1, antenna2, n_ant, freq_chan, pb_parms = calc_vis_tuple

    vis_data_shape = np.concatenate((uvw.shape[0:2], [len(freq_chan)], [len(pol)]))

    # print('pol',pol)
    vis = calc_vis(
        uvw,
        vis_data_shape,
        point_source_flux,
        point_source_ra_dec,
        pointing_ra_dec,
        phase_center_ra_dec,
        antenna1,
        antenna2,
        freq_chan,
        beam_model_map,
        eval_beam_models,
        pa,
        pol,
        _beam_parms["mueller_selection"],
    )

    if a_noise_parms is not None:
        # calc_a_noise(vis,eval_beam_models,a_noise_parms)
        from sirius import calc_a_noise

        print(calc_a_noise)
        calc_a_noise(vis, eval_beam_models, a_noise_parms)

    return vis, uvw


def make_time_da(
    time_start="2019-10-03T19:00:00.000", time_delta=3600, n_samples=10, n_chunks=4
):
    """
    Create a time series dask array.
    Parameters
    ----------
    -------
    time_da : dask.array
    """
    from astropy import units as u
    from astropy.timeseries import TimeSeries
    import dask.array as da
    import numpy as np

    ts = np.array(
        TimeSeries(
            time_start=time_start, time_delta=time_delta * u.s, n_samples=n_samples
        ).time.value
    )
    chunksize = int(np.ceil(n_samples / n_chunks))
    time_da = da.from_array(ts, chunks=chunksize)
    print("Number of chunks ", len(time_da.chunks[0]))
    return time_da


def make_chan_da(
    freq_start=3 * 10 ** 9,
    freq_delta=0.4 * 10 ** 9,
    freq_resolution=0.01 * 10 ** 9,
    n_channels=3,
    n_chunks=3,
):
    """
    Create a time series dask array.
    Parameters
    ----------
    -------
    time_da : dask.array
    """
    from astropy import units as u
    import dask.array as da
    import numpy as np

    freq_chan = np.arange(0, n_channels) * freq_delta + freq_start
    chunksize = int(np.ceil(n_channels / n_chunks))
    chan_da = da.from_array(freq_chan, chunks=chunksize)
    print("Number of chunks ", len(chan_da.chunks[0]))
    return chan_da
