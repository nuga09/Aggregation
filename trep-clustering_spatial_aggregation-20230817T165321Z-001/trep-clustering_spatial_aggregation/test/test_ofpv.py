from trep import TREP


def test_Ilmenau():
    glr = TREP(level="MUN", region="Ilmenau", identifier="name")
    glr.add_tech("OpenfieldPVRoads")
    glr.techs["OpenfieldPVRoads"].estimate_potential()
    assert (1 < glr.OpenfieldPVRoads.ec.percentAvailable < 1.1, "Unexpected area for roads")
    # TODO Test existing and capacity

    glr.add_tech("OpenfieldPV")
    glr.techs["OpenfieldPV"].estimate_potential()
    assert (11.5 < glr.OpenfieldPVRoads.ec.percentAvailable < 12, "Unexpected area for ofpv")
