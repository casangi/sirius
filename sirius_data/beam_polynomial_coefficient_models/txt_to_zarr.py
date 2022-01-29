
#http://www.aoc.nrao.edu/evla/geninfo/memoseries/evlamemo113.pdf
#https://library.nrao.edu/public/memos/evla/EVLAM_195.pdf
#https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casatools/src/code/synthesis/TransformMachines/PBMath1DEVLA.cc
'''
    //Float mag = 1.21;  // half-squint magnitude in arcmin at 1 GHz)
    feedConf_p["L"] = (-185.9)*C::pi/180.0;    // squint orientation, rads, North of +AZ axis
    feedConf_p["S"] = (-11.61)*C::pi/180.0;
    feedConf_p["C"]=(-104.8)*C::pi/180.0;
    feedConf_p["X"]=(-113.7)*C::pi/180.0;
    feedConf_p["U"]=(42.4)*C::pi/180.0;
    feedConf_p["K"]=(64.4)*C::pi/180.0;
    feedConf_p["A"]=(106.9)*C::pi/180.0;
    feedConf_p["Q"]=(85.5)*C::pi/180.0;
'''
def txt_to_zarr(filename,freq_to_hertz,dish_diam,max_coef):
    #from cngi.dio import write_zarr
    import xarray as xr
    import numpy as np
    import dask.array as da
    import csv
    from datetime import date
    import matplotlib.pyplot as plt
    
    band_mode = False
    coef_mode = False
    band = None
    
    band_name = []
    band_name_axis = []
    band_start = []
    band_end = []
    
    pol_coef = []
    
    freq = []
    
    
    txt_list = []
    with open(filename, newline='') as txtfile:
        lines = txtfile.readlines()
        for i,line in enumerate(lines):
            #print(i,line)
            if "# Band definitions" in line:
                band_mode = True
                coef_mode = False
            if "# Coefficients" in line:
                coef_mode = True
                band_mode = False
            
            if band_mode and "# Band definitions" not in line:
                split_line = line.split(' ')
                band_name.append(split_line[0])
                band_start.append(float(split_line[1]))
                band_end.append(float(split_line[2][:-1]))
             
            if coef_mode and "# Coefficients" not in line:
                if line[0] in band_name:
                    band = line[0]
                
                if line[0] not in band_name:
                    band_name_axis.append(band)
                    start_indx = line.index('{')+1
                    end_indx =  line.index('}')
                    split_line_coef = line[start_indx:end_indx].split(',')
                    start_indx = line.index('[')+1
                    end_indx =  line.index(']')
                    split_line_freq = line[start_indx:end_indx]
                    #print(len(split_line))
                    
                    coef_temp = np.zeros((max_coef,))
                    freq.append(float(split_line_freq)*10**6)
                    
                    assert len(split_line_coef) <= max_coef, 'Number of coefficients exceed max_coef'

                    for j,coef in enumerate(split_line_coef):
                        coef_temp[j] = float(coef)
                        
                    pol_coef.append(coef_temp)
    
    pol_coef = np.array(pol_coef)
    freq = np.array(freq)
    band = np.array(band_name_axis)
    #print(pol_coef.shape)
    #print(freq.shape)
                    
    pc_dataset = xr.Dataset()
    coords = {'pol':[5],'chan':freq,'coef_indx':np.arange(max_coef),'band':('chan',band)}
    pc_dataset = pc_dataset.assign_coords(coords)
    pc_dataset['BPC'] = xr.DataArray(pol_coef[:,None,:], dims=['chan','pol','coef_indx'])
    pc_dataset['ETA'] = xr.DataArray(np.zeros(pc_dataset['BPC'].shape), dims=['chan','pol','coef_indx'])
    pc_dataset.attrs['pc_file_name'] = filename.partition('/')[2]
    pc_dataset.attrs['telescope_name'] = filename.partition('/')[2].partition('_')[0]
    pc_dataset.attrs['conversion_date'] = str(date.today())
    pc_dataset.attrs['dish_diam'] = dish_diam
    
    if pc_dataset.attrs['telescope_name'].lower() == 'evla':
        pc_dataset.attrs['max_rad_1GHz'] = 0.8564*np.pi/180
    elif pc_dataset.attrs['telescope_name'].lower() == 'ngvla':
        pc_dataset.attrs['max_rad_1GHz'] = 1.5*np.pi/180
    elif pc_dataset.attrs['telescope_name'].lower() == 'alma':
        pc_dataset.attrs['max_rad_1GHz'] = 1.784*np.pi/180
    elif pc_dataset.attrs['telescope_name'].lower() == 'aca':
        pc_dataset.attrs['max_rad_1GHz'] = 3.568*np.pi/180
        
    #write_zarr(pc_dataset,filename.split('.')[0]+'.bpc.zarr')
    xr.Dataset.to_zarr(pc_dataset,filename.split('.')[0]+'.bpc.zarr',mode='w')
    
    print(pc_dataset)
    
if __name__ == '__main__':
    import shutil
    #Remove all . in name except for last (before .csv)
    filenames = ['EVLA_.txt']
    dish_diams = [25]
    freq_to_hertz = 10**6
    max_coefs =[5]
    for filename,dish_diam,max_coef in zip(filenames,dish_diams,max_coefs):
        print(filename)
        txt_to_zarr('data/'+filename,freq_to_hertz,dish_diam,max_coef)
        
        try:
            shutil.make_archive('data/'+filename[:-4]+'.bpc.zarr', 'zip', 'data/'+filename[:-4]+'.bpc.zarr')
        except:
            print('Cant compress','data/'+filename[:-4]+'.bpc.zarr')
