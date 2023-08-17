from trep import TREP

def test_1percent():
    glr = TREP(level="MUN", region="Ilmenau", identifier="name")
    exclusion_dict = {
                "water_still": None,
                "water_river": None,
                "water_stream": None,
                "motorway": None,
                "primary": None,
                "trunk": None,
                "medium_roads": None,
                "railways": None,
                "power_lines": None,
                "agriculture": None,
                "forests": None,
                "urban": 2000,
                "urban_osm": None,
                "industrial": None,
                "airports": None,
                "mineral_extraction": None,
                "dump_sites": None,
                "construction": None,
                "protected": None,
                "birds": None,
                "nature_protection": None,
                "nationalpark": None,
                "habitats": None,
                "landscape": None,
                "region_edge": None,
                "existing": None,
                "prior_wind_100m": None,
                "wind_100m": None,
                "wind_100m_power": None,
                "slope": None,
                "state": "Thüringen",
                "soft_exclusion": True
            }
    glr.add_tech("Wind")
    glr.techs["Wind"].run_exclusion(exclusion_dict=exclusion_dict)
    glr.techs["Wind"].restrict_area(share=0.01, tolerance=0.0001, step=0.0001)
    assert (0.009 * 100 < glr.techs["Wind"].ec.percentAvailable < 0.011 * 100), "Bigger deviation than tolerance"
