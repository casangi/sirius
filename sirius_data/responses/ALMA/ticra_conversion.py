#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to convert existing beam response models (see https://github.com/casangi/casaconfig/tree/master/data/alma/responses) from native CASA image format to zarr.
"""


def convert_ticra():
    """
    Interpretation of the legacy format is aided by the 2007 TICRA report, which can be found at the following URL:
    https://safe.nrao.edu/wiki/pub/Main/ALMAFEInfo/2008-01-31-optical-analysis-final-report.pdf

    From the image headers, -7.639437398726E-04 (at 125GHz) and -1.581363541536E-04 (at 720GHz) are the CDELT values for spatial coordinate which seem to be in radians, since those reflect ranges of ~6 and ~1 degrees respectively across the entire image.

    The middle number in the image names seems to be the "representative" frequency, which can be used to assign coordinate values to the chan dimension in the combined output.
    """

    import os, dask, xarray
    from casa_formats_io.casa_dask import image_to_dask

    # This path presumes the availability of CV site distribution of CASA config data
    # Could be configured to run for any installation (or pull straight from the repo)
    ticra_root = "/home/casa/data/distro/alma/responses/"

    test_images = [image for image in os.listdir(ticra_root) if ".im" in image]
    dv_images = [vertex for vertex in test_images if "DV" in vertex]
    da_images = [alcatel for alcatel in test_images if "DA" in alcatel]
    pm_images = [mitsubishi for mitsubishi in test_images if "PM" in mitsubishi]

    # outer loop - separate datasets desired for each antenna type
    for subset in [pm_images, dv_images, da_images]:
        xds = xarray.Dataset()
        for test_im in subset:
            da = image_to_dask(os.path.join(ticra_root, test_im))
            # coordinate defintions

            # TODO: add position angle coordinate, even if it's inferred and singleton
            xda = xarray.DataArray(da, dims=["chan", "pol", "lat", "lon"])
            xda = xda.assign_coords(
                {
                    "chan": [float(test_im.split("GHz")[0].split("_")[-3])],
                    "pol": [9, 10, 11, 12],
                    "lat": xda.lat,
                    "lon": xda.lon,
                }
            )
            xds = xarray.merge([xds, xda])

            # TODO: beam data should be represented by a single Variable (J)
            # condition required to handle conversion to J_norm from *.im.square.normalized

        xds.to_zarr(store=test_im.split("_")[2] + ".zarr", consolidated=True)


if __name__ == "__main__":
    convert_ticra()
