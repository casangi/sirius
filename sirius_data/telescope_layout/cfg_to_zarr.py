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

def cfg_to_zarr(infile, outfile=None):
    """
    Convert telescope array layout file (.cfg) to xarray Dataset compatible zarr format.

    This function requires CASA6 casatools module.

    Parameters
    ----------
    infile : str
        Input telescope array file (.cfg)
    outfile : str
        Output zarr filename. If None, will use infile name with .tel.zarr extension
        
    Returns
    -------
    """

    import os
    import numpy as np
    import xarray as xr
    from casatasks.private import simutil
    import casatools
    simu = simutil.simutil()
    me = casatools.measures()
    
    try:
        (x,y,z,DISH_DIAMETER,ANT_NAME,_,telescope_name, telescope_location) = simu.readantenna(infile)
        ANT_POS = np.array([x,y,z]).T
            
        telescope_dict = {}
        coords = {'ant_name': ANT_NAME, 'pos_coord': np.arange(3)}
        telescope_dict['ANT_POS'] = xr.DataArray(ANT_POS, dims=['ant_name','pos_coord'])
        #telescope_dict['ANT_NAME'] = xr.DataArray(ANT_NAME, dims=['ant'])
        telescope_dict['DISH_DIAMETER'] = xr.DataArray(DISH_DIAMETER, dims=['ant_name'])
        telescope_xds = xr.Dataset(telescope_dict, coords=coords)
        #telescope_xds = telescope_xds.assign_coords({"ant_name":("ant",ANT_NAME)})
        telescope_xds.attrs['telescope_name'] = telescope_name
        #telescope_xds.attrs['telescope_long'] = telescope_location['m0']['value']
        #telescope_xds.attrs['telescope_lat'] = telescope_location['m1']['value']
        #telescope_xds.attrs['telescope_elevation'] = telescope_location['m2']['value']
        #telescope_xds.attrs['long_units'] = telescope_location['m0']['unit']
        #telescope_xds.attrs['lat_units'] = telescope_location['m1']['unit']
        #telescope_xds.attrs['elevation_units'] = telescope_location['m2']['unit']
        #telescope_xds.attrs['coordinate_system'] = telescope_location['refer']
        #print(telescope_location)
        
        site_pos=me.measure(me.observatory(telescope_name),'ITRF')
        assert (site_pos['refer'] == 'ITRF') and (site_pos['m0']['unit'] == 'rad') and (site_pos['m1']['unit'] == 'rad')
        
        convert_latlong_to_xyz(site_pos)
        telescope_xds.attrs['site_pos'] = [site_pos]
        
        #print(telescope_name,site_pos)

        if outfile == None:
            outfile = infile[:-3] + 'tel.zarr'
            
        tmp = os.system("rm -fr " + outfile)
        tmp = os.system("mkdir " + outfile)
            
        xr.Dataset.to_zarr(telescope_xds, store=outfile, mode='w')
        return telescope_xds
    except Exception:
        print('Can not convert' , infile)
        
def convert_latlong_to_xyz(site_pos):
    import numpy as np
    x = site_pos['m2']['value']*np.cos(site_pos['m1']['value'])*np.cos(site_pos['m0']['value'])
    y = site_pos['m2']['value']*np.cos(site_pos['m1']['value'])*np.sin(site_pos['m0']['value'])
    z = site_pos['m2']['value']*np.sin(site_pos['m1']['value'])
    
    site_pos['m0']['unit'] = 'm'
    site_pos['m0']['value'] = x
    site_pos['m1']['unit'] = 'm'
    site_pos['m1']['value'] = y
    site_pos['m2']['value'] = z

if __name__ == '__main__':
    from itertools import chain
    import os
    import shutil

    directory = os.fsencode('data')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.cfg'):
            print(filename)
            cfg_to_zarr('data/'+filename)
            try:
                shutil.make_archive('data/'+filename[:-4]+'.tel.zarr', 'zip', 'data/'+filename[:-4]+'.tel.zarr')
            except:
                print('Cant compress','data/'+filename[:-4]+'.tel.zarr')
            

