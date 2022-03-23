import pytest

from sirius_data.beam_1d_func_models.airy_disk import aca, alma, vla


@pytest.fixture()
def vla_beam_model():
    airy_disk_params = vla
    return airy_disk_params


@pytest.fixture()
def aca_beam_model():
    airy_disk_params = aca
    return airy_disk_params


@pytest.fixture()
def alma_beam_model():
    airy_disk_params = alma
    return airy_disk_params


@pytest.mark.parametrize(
    "key,value",
    [
        ("func", "casa_airy"),
        ("dish_diam", 24.5),
        ("blockage_diam", 0.0),
        ("max_rad_1GHz", 0.014946999714079439),
    ],
)
def test_model_vla(vla_beam_model, key, value):
    assert vla_beam_model[key] == value


@pytest.mark.parametrize(
    "key,value",
    [
        ("func", "casa_airy"),
        ("dish_diam", 6.25),
        ("blockage_diam", 0.75),
        ("max_rad_1GHz", 0.06227334771115768),
    ],
)
def test_model_aca(aca_beam_model, key, value):
    assert aca_beam_model[key] == value


@pytest.mark.parametrize(
    "key,value",
    [
        ("func", "casa_airy"),
        ("dish_diam", 10.7),
        ("blockage_diam", 0.75),
        ("max_rad_1GHz", 0.03113667385557884),
    ],
)
def test_alma_model_numerics(alma_beam_model, key, value):
    assert key in alma_beam_model.keys()
