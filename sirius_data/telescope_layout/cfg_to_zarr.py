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


def cfg_to_zarr(infile, outfile=None, zip=True):
    """
    Convert telescope array layout file (.cfg) to xarray Dataset compatible zarr format.

    This function requires CASA6 casatools module.

    Parameters
    ----------
    infile : str
        Input telescope array file (.cfg)
    outfile : str
        Output zarr filename. If None, will use infile name with .tel.zarr extension
    zip : bool
        Zip tel.zarr file.
    Returns
    -------
    """

    from casatasks.private import simutil

    simu = simutil.simutil()

    try:
        # if True:
        # readantenna returns: earth-centered x,y,z, diameter, pad_name, antenna_name, observatory_name, observatory_measure_dictionary
        (x, y, z, DISH_DIAMETER, PAD_NAME, ANT_NAME, telescope_name, telescope_location) = simu.readantenna(infile)

        if telescope_name == "VLA":
            create_tel_zarr(
                infile, x, y, z, DISH_DIAMETER, PAD_NAME, ANT_NAME, telescope_name, telescope_location, outfile, True
            )
            telescope_name = "EVLA"
            create_tel_zarr(
                infile, x, y, z, DISH_DIAMETER, PAD_NAME, ANT_NAME, telescope_name, telescope_location, outfile, True
            )
        else:
            create_tel_zarr(
                infile, x, y, z, DISH_DIAMETER, PAD_NAME, ANT_NAME, telescope_name, telescope_location, outfile, True
            )
    except Exception as e:
        print("Can not convert", infile)
        print("Error", e)
        print("****************")


def create_tel_zarr(
    infile, x, y, z, DISH_DIAMETER, PAD_NAME, ANT_NAME, telescope_name, telescope_location, outfile=None, zip=True
):
    import os

    import casatools
    import numpy as np
    import xarray as xr

    me = casatools.measures()

    ANT_POS = np.array([x, y, z]).T

    telescope_dict = {}
    coords = {"ant_name": ANT_NAME, "pad_name": ("ant_name", PAD_NAME), "pos_coord": np.arange(3)}
    telescope_dict["ANT_POS"] = xr.DataArray(ANT_POS, dims=["ant_name", "pos_coord"])
    telescope_dict["DISH_DIAMETER"] = xr.DataArray(DISH_DIAMETER, dims=["ant_name"])
    telescope_xds = xr.Dataset(telescope_dict, coords=coords)
    telescope_xds.attrs["telescope_name"] = telescope_name

    site_pos = me.measure(me.observatory(telescope_name), "ITRF")
    assert (site_pos["refer"] == "ITRF") and (site_pos["m0"]["unit"] == "rad") and (site_pos["m1"]["unit"] == "rad")

    convert_latlong_to_xyz(site_pos)
    telescope_xds.attrs["site_pos"] = [site_pos]

    if outfile is None:
        if telescope_name == "EVLA":
            outfile = infile[: infile.rfind("/") + 1] + "evla" + infile[infile.find(".") : -3] + "tel.zarr"
        else:
            outfile = infile[:-3] + "tel.zarr"

    os.system("rm -fr " + outfile)
    os.system("mkdir " + outfile)

    xr.Dataset.to_zarr(telescope_xds, store=outfile, mode="w")

    if zip:
        try:
            shutil.make_archive(outfile, "zip", outfile)
        except Exception as ex:
            print(ex)
            print("Cant compress", outfile)

    return telescope_xds


def convert_latlong_to_xyz(site_pos):
    import numpy as np

    x = site_pos["m2"]["value"] * np.cos(site_pos["m1"]["value"]) * np.cos(site_pos["m0"]["value"])
    y = site_pos["m2"]["value"] * np.cos(site_pos["m1"]["value"]) * np.sin(site_pos["m0"]["value"])
    z = site_pos["m2"]["value"] * np.sin(site_pos["m1"]["value"])

    site_pos["m0"]["unit"] = "m"
    site_pos["m0"]["value"] = x
    site_pos["m1"]["unit"] = "m"
    site_pos["m1"]["value"] = y
    site_pos["m2"]["value"] = z


if __name__ == "__main__":
    import os
    import shutil

    # Note that Meerkat converion does not work
    # because simu.readantenna (anaconda3/envs/zinc/lib/python3.8/site-packages/casatasks/private/simutil.py)
    # makes the telescope name upper case but me.obslist() returns MeerKAT.
    # To fix add found = True above (line 1673)
    #    if found:
    #        posobs=me.measure(me.observatory(self.telescopename),'WGS84')
    directory = os.fsencode("data")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".cfg"):
            # print(filename)
            cfg_to_zarr("data/" + filename, zip=True)
