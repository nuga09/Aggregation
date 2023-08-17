from trep import TREP


def test_rtpv_Ilmenau_ryberg():
    glr = TREP(level="MUN", region="Ilmenau", identifier="name")
    glr.add_tech("RooftopPV")
    P_pv = glr.techs["RooftopPV"].estimate_roof_pv_potential()
    assert (58 < P_pv < 60), "Unexpected PV capacity"
    glr.techs["RooftopPV"].get_roof_pv_items_ryberg(P_pv)
    modules = glr.techs["RooftopPV"].predicted_items
    assert len(modules) == 16471, "Unexpected module outcome"
    assert (0.98 < modules["prob"].sum() < 1.01), "Probability doesn't add up to 1"
    assert (58 < modules["capacity"].sum() / 1e3 < 61), "Final capacity doesn't add up to calculated."

# TODO: Add test for rtpv with 3d data
