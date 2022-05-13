# Sekhar et al 2020 in Preperation explains model.
# The column headings are assumed to be stokes,freq,ind,real,imag
def csv_to_zarr(filename, freq_to_hertz, dish_diam):
    import csv
    from datetime import date

    import numpy as np
    import xarray as xr

    csv_list = []
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        next(csvreader)  # Heading ['#stokes,freq,ind,real,imag']
        for row in csvreader:
            csv_list.append(np.array(row, float))

    csv_array = np.array(csv_list)

    pol_codes = np.unique(csv_array[:, 0]).astype(int)
    chan_freq = np.unique(csv_array[:, 1])
    num_coef = int(np.max(csv_array[:, 2])) + 1

    zc_array = np.zeros((len(chan_freq), len(pol_codes), num_coef), dtype=np.complex)
    eta_array = np.zeros((len(chan_freq), len(pol_codes), num_coef), dtype=np.double)

    for i_csv in range(len(csv_array)):
        i_pol = np.where(pol_codes == csv_array[i_csv, 0])[0][0]
        i_chan = np.where(chan_freq == csv_array[i_csv, 1])[0][0]
        i_coef = int(csv_array[i_csv, 2])
        zc_array[i_chan, i_pol, i_coef] = csv_array[i_csv, 3] + 1j * csv_array[i_csv, 4]
        if len(csv_array[i_csv, :]) > 5:
            eta_array[i_chan, i_pol, i_coef] = csv_array[i_csv, 5]
        else:
            eta_array[i_chan, i_pol, i_coef] = 1

    zc_dataset = xr.Dataset()
    coords = {"pol": pol_codes, "chan": chan_freq * freq_to_hertz, "coef_indx": np.arange(num_coef)}
    zc_dataset = zc_dataset.assign_coords(coords)
    zc_dataset["ZPC"] = xr.DataArray(zc_array, dims=["chan", "pol", "coef_indx"])
    zc_dataset["ETA"] = xr.DataArray(eta_array, dims=["chan", "pol", "coef_indx"])
    zc_dataset.attrs["apc_file_name"] = filename.partition("/")[2]
    zc_dataset.attrs["telescope_name"] = filename.partition("/")[2].partition("_")[0]
    zc_dataset.attrs["conversion_date"] = str(date.today())
    zc_dataset.attrs["dish_diam"] = dish_diam

    if zc_dataset.attrs["telescope_name"].lower() == "evla":
        zc_dataset.attrs["max_rad_1GHz"] = 0.8564 * np.pi / 180
    elif zc_dataset.attrs["telescope_name"].lower() == "ngvla":
        zc_dataset.attrs["max_rad_1GHz"] = 1.5 * np.pi / 180
    elif zc_dataset.attrs["telescope_name"].lower() == "alma":
        zc_dataset.attrs["max_rad_1GHz"] = 1.784 * np.pi / 180
    elif zc_dataset.attrs["telescope_name"].lower() == "aca":
        zc_dataset.attrs["max_rad_1GHz"] = 3.568 * np.pi / 180

    xr.Dataset.to_zarr(zc_dataset, filename.split(".")[0] + ".apc.zarr", mode="w")
    print(zc_dataset)


if __name__ == "__main__":
    import shutil

    # Remove all . in name except for last (before .csv)
    filenames = ["evla_Lband.csv", "evla_Sband.csv", "meerkat_Lband.csv"]
    dish_diams = [25, 25, 25, 13.5]
    freq_to_hertz = 10**6
    for filename, dish_diam in zip(filenames, dish_diams):
        print(filename)
        csv_to_zarr("data/" + filename, freq_to_hertz, dish_diam)
        try:
            shutil.make_archive("data/" + filename[:-4] + ".apc.zarr", "zip", "data/" + filename[:-4] + ".apc.zarr")
        except Exception as ex:
            print("Cant compress", "data/" + filename[:-4] + ".apc.zarr")
            print(ex)
