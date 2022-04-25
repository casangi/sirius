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

import numpy as np

def _get_time_ha_aligned_time_casa(tel_xds,start_date,phase_center_ra_dec):
    from casatools import simulator, measures
    from sirius._sirius_utils._cngi_io import read_ms
    me = measures()
    import os
    import numpy as np

    sm = simulator()
    ant_pos = tel_xds.ANT_POS.values
    os.system("rm -rf temp_42.ms")
    sm.open(ms="temp_42.ms")

    ## Set the antenna configuration
    sm.setconfig(
        telescopename=tel_xds.telescope_name,
        x=ant_pos[:, 0],
        y=ant_pos[:, 1],
        z=ant_pos[:, 2],
        dishdiameter=tel_xds.DISH_DIAMETER.values,
        mount=["alt-az"],
        antname=list(
            tel_xds.ant_name.values
        ),  # CASA can't handle an array of antenna names.
        coordsystem="global",
        referencelocation=tel_xds.site_pos[0],
    )

    sm.setfeed(mode="perfect X Y", pol=[""])


    sm.setspwindow(
        spwname="temp",
        freq="1GHz",
        deltafreq="0.5GHz",
        freqresolution="0.5GHz",
        nchannels=1,
        refcode="LSRK",
    )


    sm.setauto(autocorrwt=0.0)

    sm.settimes(
        integrationtime='60s',
        usehourangle=True,
        referencetime=me.epoch('utc', start_date),
    )
    
    
    dir_dict = {"m0": {"unit": "rad", "value": phase_center_ra_dec[0]},
                "m1": {"unit": "rad", "value": phase_center_ra_dec[1]},
                "refer": "J2000",
                "type": "direction",}
    
    sm.setfield(sourcename="temp_field", sourcedirection=dir_dict)


    sm.observe(sourcename="temp_field",
                spwname="temp",
                starttime="0h",
                stoptime="1h",)

    sm.close()
    
    mxds = read_ms("temp_42.ms")
    start_time = mxds.xds0.TIME.values[0]
    
    return start_time
    
    
    
    
    
def _get_time_ha_aligned_time(earth_location,initial_start_time,phase_center_ra_dec):
    import casatools
    qa = casatools.quanta()
    qq = qa.quantity

    ref = 'LAST'
    me = casatools.measures()
    me.doframe(earth_location)
    last = me.measure(me.epoch('utc', initial_start_time), ref) #Convert utc to Local Apparent Sidereal Time
    new_last = np.round(last['m0']['value']) + phase_center_ra_dec[0]/(2*np.pi) # LAST = HA + RA
    new_last = {'m0': {'value': new_last, 'unit': 'd'},'refer': ref,'type': 'epoch'}
    new_start_time = me.measure(new_last, 'UTC')
    new_start_time = qa.time({'value': new_start_time['m0']['value'], 'unit': 'd'},form="fits")[0]
    
    return(new_start_time)
