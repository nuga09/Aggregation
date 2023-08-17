from trep import TREP
import time

def test_Ilm_Kreis():
    glr = TREP("Ilm-Kreis", identifier="name")
    glr.add_tech("Wind")
    glr.techs["Wind"].estimate_potential()
    assert 13.8 > glr.techs["Wind"].ec.percentAvailable > 13.6, "Unexpected area outcome"
    assert len(glr.techs["Wind"].predicted_items) == 1246, "Unexpected turbine outcome"


def test_Ilmenau():
    glr = TREP(level="MUN", region="Ilmenau", identifier="name")
    glr.add_tech("Wind")
    glr.techs["Wind"].estimate_potential()
    assert 5.2 > glr.techs["Wind"].ec.percentAvailable > 4.8, "Unexpected area outcome"
    assert len(glr.techs["Wind"].predicted_items) == 39, "Unexpected turbine outcome"


def test_other():
    glr = TREP(level="MUN", region="Reußenköge", identifier="name")
    glr.Wind.get_existing_plants(glr.Wind.ec, mastr=True)
    glr.Wind.exclude_existing(how="ellipse")
    glr.Wind.estimate_potential()
    assert 7.8 > glr.Wind.ec.percentAvailable > 7.4, "Unexpected area outcome"
    assert len(glr.Wind.predicted_items) == 48, "Unexpected turbine outcome"


def test_Juelich():
    glr = TREP("Jülich", level="MUN", case="tests", identifier="name")
    glr.Wind.get_existing_plants(glr.Wind.ec, mastr=True)
    glr.Wind.estimate_potential()
    assert 7.8 > glr.techs["Wind"].ec.percentAvailable > 7.4, "Unexpected area outcome"
    assert len(glr.techs["Wind"].predicted_items) == 92, "Unexpected turbine outcome"
