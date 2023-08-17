from abc import ABC, abstractmethod
import glaes as gl
import os
import json

from numpy.lib.function_base import place
from trep import utils
import reskit as rk
import pandas as pd
import datetime as dt
from MATES.core.InputGenerator import IG_utils
import warnings
import numpy as np
import geokit as gk
import time
from matplotlib.sankey import Sankey
from typing import Dict, Type, Text
import plotly.graph_objects as go
from FINE.spagat.RE_representation import represent_RE_technology

class Technology(ABC):
    def __init__(self,
                 parent):
        self.parent = parent
        self.ec = parent.new_ec()
        self.predicted_items = None
        self.ts_predicted_items = None
        self.existing_items = None
        self.report_dict = None
        self.ts_existing_items = None

    def _run_exclusion(self, exclusion_dict, ec=None, plot_sankey=True, use_net_flows=True):
        """Run exclusion with glaes based on exclusion dict.

        Workflow wrapper for GLAES. Some additional functionalities and files
        for regional exclusions in certain German states.

        Parameters
        ----------
        exclusion_dict : dict
            Dictionary containing the information for the exclusion
        ec : optional
            exclusion calculator, by default None
        plot_sankey : bool, optional
            by default False
        """
        print("Start exclusion!")
        # TODO reomve unused keys and add new keys
        keys_social = [
            "health_treatment_buildings", "buildings", "mixed_buildings", "inner_areas", "outer_areas", "residential",
            "mixed_usage", "5Houses", "cemetery", "10Houses", "industrial_commercial", "military", "recreational",
            "camping", "historical", "mineral_extraction", "dump_sites", "construction", "buildings_all",
            "buildings_commercial"
        ]
        keys_infrastructre = [
            "airports", "airfields", "motorway", "primary_roads", "secondary_roads",
            "regional_roads", "railways", "power_lines", "dvor", "vor", "seismic_station",
        ]
        keys_physical = [
            "water_still", "water_river", "water_stream", "farmland", "grassland", "forests", "forests_outside_FRA",
            "forests_in_FRA_without_coniferous_forests", "trees"
        ]
        keys_eco_tech = [
            "wind_100m", "wind_100m_era", "wind_100m_power", "elevation", "slope",
        ]
        keys_protected = [
            "birds", "nature_protection", "nationalpark", "habitats", "landscape", "protected",
            "biospheres_core", "biospheres_develop", "biospheres_maintain"
        ]
        implemented_keys = [
            "airports", "airfields", "health_treatment_buildings", "buildings", "mixed_buildings",
            "water_still", "water_river", "water_stream", "motorway", "primary_roads", "secondary_roads",
            "regional_roads", "railways", "power_lines", "farmland", "grassland", "forests",
            "inner_areas", "outer_areas", "residential", "mixed_usage", "5Houses", "cemetery",
            "10Houses", "industrial_commercial", "dvor", "vor", "military", "recreational", "available_side_stripes",
            "camping", "historical", "mineral_extraction", "dump_sites", "construction", "buildings_all",
            "wind_100m", "wind_100m_era", "wind_100m_power", "elevation", "slope", "seismic_station",
            "birds", "nature_protection", "nationalpark", "habitats", "landscape", "trees",
            "protected", "region_edge", "existing", "soft_exclusion", "state", "auxiliary", "forests_outside_FRA",
            "forests_in_FRA_without_coniferous_forests", "buildings_commercial", "border", "biospheres_core",
            "biospheres_develop", "biospheres_maintain"
        ]
        # get the conditions of each key
        exclusion_dict = self._get_exclusion_condition(exclusion_dict)
        for key in exclusion_dict.keys():
            if key not in implemented_keys:
                warnings.warn(
                    f"Key {key} in exclusion dict is not implemented and"
                    + "therefore not excluded. Possible exclusions are"
                    + f" {implemented_keys}", UserWarning)
        if ec is None:
            _ec = self.ec
        else:
            _ec = ec
        _datasources_path = self.parent.datasource_path
        # _osm_path = os.path.join(self.parent.datasource_path, utils.get_osm_path(self.parent._state))
        _osm_path = os.path.join(self.parent.datasource_path, os.path.join("osm", "merged"))
        _state_path = os.path.join(
            _datasources_path, "state_" + self.parent._state)
        _road_shape_path = os.path.join(_osm_path, "gis_osm_roads_free_1.shp")
        _intermediate_path = self.parent.intermediate_path
        _wdpa_path = [
            os.path.join(
                _datasources_path, "wdpa", "WDPA_DE.shp")]

        # get the initial available area
        init_available_areas = _ec.areaAvailable
        remaining_area = init_available_areas
        excluded_areas = []

        excluded_areas_social = []
        labels_social = []
        excluded_areas_infrastructure = []
        labels_infrastructure = []
        excluded_areas_physical = []
        labels_physical = []
        excluded_areas_eco_tech = []
        labels_eco_tech = []
        excluded_areas_protected = []
        labels_protected = []
        # TODO another category only for auxiliary exclusion?
        excluded_areas_others = []
        labels_others = []

        start = time.time()
        if exclusion_dict.get("border") is not None:
            # Urban fabric
            self._exclude_features(exclusion_dict["border"], _ec,
                                   intermediate=os.path.join(_intermediate_path,
                                                             "border_"
                                                             f"{exclusion_dict['border']['source']}_"
                                                             f"{exclusion_dict['border']['buffer']}_"
                                                             f"{self.parent._id}.tif"
                                                             ),
                                   plot_sankey=plot_sankey
                                   )
            print("Excluded border with " +
                  f"{exclusion_dict['border']['source']} " +
                  f"{exclusion_dict['border']['buffer']} " +
                  f"after {(time.time() - start) / 60} minutes", flush=True)
        else:
            print("not excluding border", flush=True)
        if exclusion_dict.get("motorway") is not None:
            if isinstance(exclusion_dict['motorway']['path'], (tuple, list)):
                excluded_area = 0
                for n in range(len(exclusion_dict['motorway']['path'])):
                    temp_dict = exclusion_dict["motorway"].copy()
                    temp_dict["path"] = exclusion_dict["motorway"]["path"][n]
                    temp_dict["where_text"] = exclusion_dict["motorway"]["where_text"][n]
                    excluded_area += self._exclude_features(temp_dict, _ec,
                                                            intermediate=os.path.join(_intermediate_path,
                                                                                      f"motorway_{n}_"
                                                                                      f"{exclusion_dict['motorway']['source']}_"
                                                                                      f"{exclusion_dict['motorway']['buffer']}_"
                                                                                      f"{self.parent._id}.tif"
                                                                                      ),
                                                            plot_sankey=plot_sankey
                                                            )
            else:
                excluded_area = self._exclude_features(exclusion_dict["motorway"], _ec,
                                                       intermediate=os.path.join(_intermediate_path,
                                                                                 "motorway_"
                                                                                 f"{exclusion_dict['motorway']['source']}_"
                                                                                 f"{exclusion_dict['motorway']['buffer']}_"
                                                                                 f"{self.parent._id}.tif"
                                                                                 ),
                                                       plot_sankey=plot_sankey
                                                       )
            print("Excluded motorway with " +
                  f"{exclusion_dict['motorway']['source']} " +
                  f"{exclusion_dict['motorway']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Motorways")
        else:
            print("not excluding motorway", flush=True)
        if exclusion_dict.get("primary_roads") is not None:
            excluded_area = self._exclude_features(exclusion_dict["primary_roads"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "primary_roads_"
                                                                             f"{exclusion_dict['primary_roads']['source']}_"
                                                                             f"{exclusion_dict['primary_roads']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded primary_roads with " +
                  f"{exclusion_dict['primary_roads']['source']} " +
                  f"{exclusion_dict['primary_roads']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Primary roads")
        else:
            print("not excluding primary_roads", flush=True)
        if exclusion_dict.get("secondary_roads") is not None:
            excluded_area = self._exclude_features(exclusion_dict["secondary_roads"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "secondary_roads_"
                                                                             f"{exclusion_dict['secondary_roads']['source']}_"
                                                                             f"{exclusion_dict['secondary_roads']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded secondary_roads with " +
                  f"{exclusion_dict['secondary_roads']['source']} " +
                  f"{exclusion_dict['secondary_roads']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Secondary roads")
        else:
            print("not excluding secondary_roads", flush=True)
        if exclusion_dict.get("railways") is not None:
            if isinstance(exclusion_dict['railways']['path'], (tuple, list)):
                excluded_area = 0
                for n in range(len(exclusion_dict['railways']['path'])):
                    temp_dict = exclusion_dict["railways"].copy()
                    temp_dict["path"] = exclusion_dict["railways"]["path"][n]
                    temp_dict["where_text"] = exclusion_dict["railways"]["where_text"][n]
                    excluded_area += self._exclude_features(temp_dict, _ec,
                                                            intermediate=os.path.join(_intermediate_path,
                                                                                      f"railways_{n}_"
                                                                                      f"{exclusion_dict['railways']['source']}_"
                                                                                      f"{exclusion_dict['railways']['buffer']}_"
                                                                                      f"{self.parent._id}.tif"
                                                                                      ),
                                                            plot_sankey=plot_sankey
                                                            )
            else:
                excluded_area = self._exclude_features(exclusion_dict["railways"], _ec,
                                                       intermediate=os.path.join(_intermediate_path,
                                                                                 "railways_"
                                                                                 f"{exclusion_dict['railways']['source']}_"
                                                                                 f"{exclusion_dict['railways']['buffer']}_"
                                                                                 f"{self.parent._id}.tif"
                                                                                 ),
                                                       plot_sankey=plot_sankey
                                                       )
            print("Excluded railways with " +
                  f"{exclusion_dict['railways']['source']} " +
                  f"{exclusion_dict['railways']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Railways")
        else:
            print("not excluding railways", flush=True)
        if exclusion_dict.get("power_lines") is not None:
            excluded_area = self._exclude_features(exclusion_dict["power_lines"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "power_lines_"
                                                                             f"{exclusion_dict['power_lines']['source']}_"
                                                                             f"{exclusion_dict['power_lines']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded power line with " +
                  f"{exclusion_dict['power_lines']['source']} " +
                  f"{exclusion_dict['power_lines']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Power lines")
        else:
            print("not excluding power line", flush=True)

        if exclusion_dict.get("inner_areas") is not None:
            excluded_area = self._exclude_features(exclusion_dict["inner_areas"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "inner_areas_"
                                                                             f"{exclusion_dict['inner_areas']['source']}_"
                                                                             f"{exclusion_dict['inner_areas']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded inner areas with " +
                  f"{exclusion_dict['inner_areas']['source']} " +
                  f"{exclusion_dict['inner_areas']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Inner areas")
        else:
            print("not excluding inner areas", flush=True)

        if exclusion_dict.get("5Houses") is not None:
            excluded_area = self._exclude_features(exclusion_dict["5Houses"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "5Houses_"
                                                                             f"{exclusion_dict['5Houses']['source']}_"
                                                                             f"{exclusion_dict['5Houses']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded 5Houses with " +
                  f"{exclusion_dict['5Houses']['source']} " +
                  f"{exclusion_dict['5Houses']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("5-Houses")
        else:
            print("not excluding 5 Houses", flush=True)

        if exclusion_dict.get("10Houses") is not None:
            excluded_area = self._exclude_features(exclusion_dict["10Houses"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "10Houses_"
                                                                             f"{exclusion_dict['10Houses']['source']}_"
                                                                             f"{exclusion_dict['10Houses']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded 10Houses with " +
                  f"{exclusion_dict['10Houses']['source']} " +
                  f"{exclusion_dict['10Houses']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("10-Houses")
        else:
            print("not excluding 10Houses", flush=True)

        if exclusion_dict.get("outer_areas") is not None:
            excluded_area = self._exclude_features(exclusion_dict["outer_areas"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "outer_areas_"
                                                                             f"{exclusion_dict['outer_areas']['source']}_"
                                                                             f"{exclusion_dict['outer_areas']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            # Urban fabric
            print("Excluded residential outer areas with " +
                  f"{exclusion_dict['outer_areas']['source']} " +
                  f"{exclusion_dict['outer_areas']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Outer areas")
        else:
            print("not excluding outer areas", flush=True)

        if exclusion_dict.get("residential") is not None:
            excluded_area = self._exclude_features(exclusion_dict["residential"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "residential_"
                                                                             f"{exclusion_dict['residential']['source']}_"
                                                                             f"{exclusion_dict['residential']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            # Urban fabric
            print("Excluded residential areas with " +
                  f"{exclusion_dict['residential']['source']} " +
                  f"{exclusion_dict['residential']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Residential areas")
        else:
            print("not excluding residential areas", flush=True)

        if exclusion_dict.get("mixed_usage") is not None:
            excluded_area = self._exclude_features(exclusion_dict["mixed_usage"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "mixed_usage_"
                                                                             f"{exclusion_dict['mixed_usage']['source']}_"
                                                                             f"{exclusion_dict['mixed_usage']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            # Urban fabric
            print("Excluded residential mixed used areas with " +
                  f"{exclusion_dict['mixed_usage']['source']} " +
                  f"{exclusion_dict['mixed_usage']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Mixed-use areas")
        else:
            print("not excluding residential mixed used areas", flush=True)

        if exclusion_dict.get("industrial_commercial") is not None:
            excluded_area = self._exclude_features(exclusion_dict["industrial_commercial"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "industrial_commercial_"
                                                                             f"{exclusion_dict['industrial_commercial']['source']}_"
                                                                             f"{exclusion_dict['industrial_commercial']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded industrial or commercial areas with " +
                  f"{exclusion_dict['industrial_commercial']['source']} " +
                  f"{exclusion_dict['industrial_commercial']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Indu & Commer")
        else:
            print("not excluding industrial or commercial areas", flush=True)
        if exclusion_dict.get("secondary_roads") is not None:
            excluded_area = self._exclude_features(exclusion_dict["secondary_roads"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "secondary_roads_"
                                                                             f"{exclusion_dict['secondary_roads']['source']}_"
                                                                             f"{exclusion_dict['secondary_roads']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded secondary_roads with " +
                  f"{exclusion_dict['secondary_roads']['source']} " +
                  f"{exclusion_dict['secondary_roads']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Secondary roads")
        else:
            print("not excluding secondary_roads", flush=True)
        if exclusion_dict.get("regional_roads") is not None:
            excluded_area = self._exclude_features(exclusion_dict["regional_roads"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "regional_roads_"
                                                                             f"{exclusion_dict['regional_roads']['source']}_"
                                                                             f"{exclusion_dict['regional_roads']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded regional_roads with " +
                  f"{exclusion_dict['regional_roads']['source']} " +
                  f"{exclusion_dict['regional_roads']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Regional roads")
        else:
            print("not excluding regional_roads", flush=True)
        if exclusion_dict.get("buildings") is not None:
            # residential buildings
            excluded_area = self._exclude_features(exclusion_dict["buildings"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "buildings_"
                                                                             f"{exclusion_dict['buildings']['source']}_"
                                                                             f"{exclusion_dict['buildings']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded buildings with "
                  f"{exclusion_dict['buildings']['source']} " +
                  f"{exclusion_dict['buildings']['buffer']} " +
                  f"after {(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            remaining_area = _ec.areaAvailable
            excluded_areas_social.append(excluded_area)
            labels_social.append("Residential buildings")
        else:
            print("not excluding buildings", flush=True)
        if exclusion_dict.get("buildings_commercial") is not None:
            # commercial buildings
            excluded_area = self._exclude_features(exclusion_dict["buildings_commercial"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "buildings_commercial_"
                                                                             f"{exclusion_dict['buildings_commercial']['source']}_"
                                                                             f"{exclusion_dict['buildings_commercial']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded buildings_commercial with "
                  f"{exclusion_dict['buildings_commercial']['source']} " +
                  f"{exclusion_dict['buildings_commercial']['buffer']} " +
                  f"after {(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            remaining_area = _ec.areaAvailable
            excluded_areas_social.append(excluded_area)
            labels_social.append("Commercial buildings")
        else:
            print("not excluding buildings_commercial", flush=True)
        if exclusion_dict.get("buildings_all") is not None:
            # all buildings
            excluded_area = self._exclude_features(exclusion_dict["buildings_all"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "buildings_all_"
                                                                             f"{exclusion_dict['buildings_all']['source']}_"
                                                                             f"{exclusion_dict['buildings_all']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded buildings_all with "
                  f"{exclusion_dict['buildings_all']['source']} " +
                  f"{exclusion_dict['buildings_all']['buffer']} " +
                  f"after {(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            remaining_area = _ec.areaAvailable
            excluded_areas_social.append(excluded_area)
            labels_social.append("Buildings")
        else:
            print("not excluding buildings_all", flush=True)
        if exclusion_dict.get("mixed_buildings") is not None:
            # Urban fabric
            excluded_area = self._exclude_features(exclusion_dict["mixed_buildings"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "mixed_buildings_"
                                                                             f"{exclusion_dict['mixed_buildings']['source']}_"
                                                                             f"{exclusion_dict['mixed_buildings']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded mixed_buildings with " +
                  f"{exclusion_dict['mixed_buildings']['source']} " +
                  f"{exclusion_dict['mixed_buildings']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Mixed-use buildings")
        else:
            print("not excluding mixed buildings", flush=True)
        if exclusion_dict.get("health_treatment_buildings") is not None:
            # Urban fabric
            excluded_area = self._exclude_features(exclusion_dict["health_treatment_buildings"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "health_treatment_buildings_"
                                                                             f"{exclusion_dict['health_treatment_buildings']['source']}_"
                                                                             f"{exclusion_dict['health_treatment_buildings']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded health_treatment_buildings with " +
                  f"{exclusion_dict['health_treatment_buildings']['source']} " +
                  f"{exclusion_dict['health_treatment_buildings']['buffer']} " +
                  f"after {(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            remaining_area = _ec.areaAvailable
            excluded_areas_social.append(excluded_area)
            labels_social.append("Health treatment buildings")
        else:
            print("not excluding treatment_buildings buildings", flush=True)
        if exclusion_dict.get("water_still") is not None:
            excluded_area = self._exclude_features(exclusion_dict["water_still"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "water_still_"
                                                                             f"{exclusion_dict['water_still']['source']}_"
                                                                             f"{exclusion_dict['water_still']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded water_still with " +
                  f"{exclusion_dict['water_still']['source']} " +
                  f"{exclusion_dict['water_still']['buffer']}" +
                  f" after {(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Lakes")
        else:
            print("not excluding still water", flush=True)
        if exclusion_dict.get("water_river") is not None:
            excluded_area = self._exclude_features(exclusion_dict["water_river"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "water_river_"
                                                                             f"{exclusion_dict['water_river']['source']}_"
                                                                             f"{exclusion_dict['water_river']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded water_river with " +
                  f"{exclusion_dict['water_river']['source']} " +
                  f"{exclusion_dict['water_river']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Rivers")
        else:
            print("not excluding rivers", flush=True)
        if exclusion_dict.get("water_stream") is not None:
            excluded_area = self._exclude_features(exclusion_dict["water_stream"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "water_stream_"
                                                                             f"{exclusion_dict['water_stream']['source']}_"
                                                                             f"{exclusion_dict['water_stream']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded streams with " +
                  f"{exclusion_dict['water_stream']['source']} " +
                  f"{exclusion_dict['water_stream']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Streams")
        else:
            print("not excluding streams", flush=True)
        if exclusion_dict.get("farmland") is not None:
            excluded_area = self._exclude_features(exclusion_dict["farmland"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "farmland_"
                                                                             f"{exclusion_dict['farmland']['source']}_"
                                                                             f"{exclusion_dict['farmland']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded farmland with " +
                  f"{exclusion_dict['farmland']['source']} " +
                  f"{exclusion_dict['farmland']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Farmlands")
        else:
            print("not excluding farmland", flush=True)
        if exclusion_dict.get("grassland") is not None:
            excluded_area = self._exclude_features(exclusion_dict["grassland"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "grassland_"
                                                                             f"{exclusion_dict['grassland']['source']}_"
                                                                             f"{exclusion_dict['grassland']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded grassland with " +
                  f"{exclusion_dict['grassland']['source']} " +
                  f"{exclusion_dict['grassland']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Grassland")
        else:
            print("not excluding grassland", flush=True)
        if exclusion_dict.get("forests") is not None:
            excluded_area = self._exclude_features(exclusion_dict["forests"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "forests_"
                                                                             f"{exclusion_dict['forests']['source']}_"
                                                                             f"{exclusion_dict['forests']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded forests with " +
                  f"{exclusion_dict['forests']['source']} " +
                  f"{exclusion_dict['forests']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Forests")
        else:
            print("not excluding forests", flush=True)
        if exclusion_dict.get("forests_outside_FRA") is not None:
            excluded_area = self._exclude_features(exclusion_dict["forests_outside_FRA"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "forests_outside_FRA_"
                                                                             f"{exclusion_dict['forests_outside_FRA']['source']}_"
                                                                             f"{exclusion_dict['forests_outside_FRA']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded forests_outside_FRA with " +
                  f"{exclusion_dict['forests_outside_FRA']['source']} " +
                  f"{exclusion_dict['forests_outside_FRA']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Forests outside FRA")
        else:
            print("not excluding forests_outside_FRA", flush=True)
        if exclusion_dict.get("forests_in_FRA_without_coniferous_forests") is not None:
            excluded_area = self._exclude_features(exclusion_dict["forests_in_FRA_without_coniferous_forests"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "forests_in_FRA_without_coniferous_forests_"
                                                                             f"{exclusion_dict['forests_in_FRA_without_coniferous_forests']['source']}_"
                                                                             f"{exclusion_dict['forests_in_FRA_without_coniferous_forests']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded forests_in_FRA_without_coniferous_forests with " +
                  f"{exclusion_dict['forests_in_FRA_without_coniferous_forests']['source']} " +
                  f"{exclusion_dict['forests_in_FRA_without_coniferous_forests']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("NC forests in FRA")
        else:
            print("not excluding forests_in_FRA_without_coniferous_forests", flush=True)
        if exclusion_dict.get("trees") is not None:
            excluded_area = self._exclude_features(exclusion_dict["trees"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "trees_"
                                                                             f"{exclusion_dict['trees']['source']}_"
                                                                             f"{exclusion_dict['trees']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded trees with " +
                  f"{exclusion_dict['trees']['source']} " +
                  f"{exclusion_dict['trees']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_physical.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_physical.append("Trees")
        else:
            print("not excluding trees", flush=True)

        if exclusion_dict.get("airports") is not None:
            excluded_area = self._exclude_features(exclusion_dict["airports"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "airports_"
                                                                             f"{exclusion_dict['airports']['source']}_"
                                                                             f"{exclusion_dict['airports']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded airports with " +
                  f"{exclusion_dict['airports']['source']} " +
                  f"{exclusion_dict['airports']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Airports")
        else:
            print("not excluding airports", flush=True)

        if exclusion_dict.get("airfields") is not None:
            excluded_area = self._exclude_features(exclusion_dict["airfields"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "airfields_"
                                                                             f"{exclusion_dict['airfields']['source']}_"
                                                                             f"{exclusion_dict['airfields']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded airfields with " +
                  f"{exclusion_dict['airfields']['source']} " +
                  f"{exclusion_dict['airfields']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Airfields")
        else:
            print("not excluding airfields", flush=True)

        if exclusion_dict.get("dvor"):
            excluded_area = self._exclude_features(exclusion_dict["dvor"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "dvor_"
                                                                             f"{exclusion_dict['dvor']['source']}_"
                                                                             f"{exclusion_dict['dvor']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded radio navigator D-VOR with " +
                  f"{exclusion_dict['dvor']['source']} " +
                  f"{exclusion_dict['dvor']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("D-VOR")
        else:
            print("not excluding radio navigator D-VOR", flush=True)

        if exclusion_dict.get("vor"):
            excluded_area = self._exclude_features(exclusion_dict["vor"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "vor_"
                                                                             f"{exclusion_dict['vor']['source']}_"
                                                                             f"{exclusion_dict['vor']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded radio navigator VOR with " +
                  f"{exclusion_dict['vor']['source']} " +
                  f"{exclusion_dict['vor']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("VOR")
        else:
            print("not excluding radio navigator VOR", flush=True)

        if exclusion_dict.get("seismic_station"):
            excluded_area = self._exclude_features(exclusion_dict["seismic_station"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "seismic_station_"
                                                                             f"{exclusion_dict['seismic_station']['source']}_"
                                                                             f"{exclusion_dict['seismic_station']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded radio navigator seismic station with " +
                  f"{exclusion_dict['seismic_station']['source']} " +
                  f"{exclusion_dict['seismic_station']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_infrastructure.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_infrastructure.append("Seismic station")
        else:
            print("not excluding seismic station", flush=True)

        if exclusion_dict.get("military") is not None:
            excluded_area = self._exclude_features(exclusion_dict["military"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "military_"
                                                                             f"{exclusion_dict['military']['source']}_"
                                                                             f"{exclusion_dict['military']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded military landuse with " +
                  f"{exclusion_dict['military']['source']} " +
                  f"{exclusion_dict['military']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Military")
        else:
            print("not excluding military", flush=True)

        if exclusion_dict.get("cemetery") is not None:
            excluded_area = self._exclude_features(exclusion_dict["cemetery"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "cemetery_"
                                                                             f"{exclusion_dict['cemetery']['source']}_"
                                                                             f"{exclusion_dict['cemetery']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded cemetery areas with " +
                  f"{exclusion_dict['cemetery']['source']} " +
                  f"{exclusion_dict['cemetery']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Cemetery")
        else:
            print("not excluding cemetery", flush=True)

        if exclusion_dict.get("recreational") is not None:
            excluded_area = self._exclude_features(exclusion_dict["recreational"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "recreational_"
                                                                             f"{exclusion_dict['recreational']['source']}_"
                                                                             f"{exclusion_dict['recreational']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded recreational areas with " +
                  f"{exclusion_dict['recreational']['source']} " +
                  f"{exclusion_dict['recreational']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Recreational areas")
        else:
            print("not excluding recreational", flush=True)

        if exclusion_dict.get("camping") is not None:
            excluded_area = self._exclude_features(exclusion_dict["camping"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "camping_"
                                                                             f"{exclusion_dict['camping']['source']}_"
                                                                             f"{exclusion_dict['camping']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded camping sites with " +
                  f"{exclusion_dict['camping']['source']} " +
                  f"{exclusion_dict['camping']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Camping sites")
        else:
            print("not excluding camping", flush=True)

        if exclusion_dict.get("historical") is not None:
            excluded_area = self._exclude_features(exclusion_dict["historical"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "historical_"
                                                                             f"{exclusion_dict['historical']['source']}_"
                                                                             f"{exclusion_dict['historical']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded historical sites with " +
                  f"{exclusion_dict['historical']['source']} " +
                  f"{exclusion_dict['historical']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Historical sites")
        else:
            print("not excluding historical sites", flush=True)

        if exclusion_dict.get("mineral_extraction") is not None:
            excluded_area = self._exclude_features(exclusion_dict["mineral_extraction"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "mineral_extraction_"
                                                                             f"{exclusion_dict['mineral_extraction']['source']}_"
                                                                             f"{exclusion_dict['mineral_extraction']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded mineral extraction with " +
                  f"{exclusion_dict['mineral_extraction']['source']} " +
                  f"{exclusion_dict['mineral_extraction']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Mineral extraction sites")
        else:
            print(f"Didn't exclude mineral extraction", flush=True)

        if exclusion_dict.get("dump_sites") is not None:
            excluded_area = self._exclude_features(exclusion_dict["dump_sites"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "dump_sites_"
                                                                             f"{exclusion_dict['dump_sites']['source']}_"
                                                                             f"{exclusion_dict['dump_sites']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded dump sites with " +
                  f"{exclusion_dict['dump_sites']['source']} " +
                  f"{exclusion_dict['dump_sites']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Dump sites")
        else:
            print("not excluding dump sites", flush=True)

        if exclusion_dict.get("construction") is not None:
            excluded_area = self._exclude_features(exclusion_dict["construction"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "construction_"
                                                                             f"{exclusion_dict['construction']['source']}_"
                                                                             f"{exclusion_dict['construction']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded construction sites with " +
                  f"{exclusion_dict['construction']['source']} " +
                  f"{exclusion_dict['construction']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_social.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_social.append("Construction sites")
        else:
            print("not excluding construction sites", flush=True)

        if exclusion_dict.get("wind_100m") is not None:
            excluded_area = self._exclude_features(exclusion_dict["wind_100m"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "wind_100m_"
                                                                             f"{exclusion_dict['wind_100m']['source']}_"
                                                                             f"{exclusion_dict['wind_100m']['value']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(
                "Excluded wind_100m with " +
                f"{exclusion_dict['wind_100m']['source']} " +
                f"{exclusion_dict['wind_100m']['value']} after " +
                f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_eco_tech.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_eco_tech.append("Wind speed at 100m")
        else:
            print("not excluding wind100m", flush=True)
        if exclusion_dict.get("wind_100m_era") is not None:
            excluded_area = self._exclude_features(exclusion_dict["wind_100m_era"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "wind_100m_era_"
                                                                             f"{exclusion_dict['wind_100m_era']['source']}_"
                                                                             f"{exclusion_dict['wind_100m_era']['value']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded wind_100m from era with " +
                  f"{exclusion_dict['wind_100m_era']['source']} " +
                  f"{exclusion_dict['wind_100m_era']['value']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_eco_tech.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_eco_tech.append("Wind speed at 100m")
        else:
            print("not excluding wind100m from era", flush=True)
        if exclusion_dict.get("wind_100m_power") is not None:
            excluded_area = self._exclude_features(exclusion_dict["wind_100m_power"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "wind_100m_power_"
                                                                             f"{exclusion_dict['wind_100m_power']['source']}_"
                                                                             f"{exclusion_dict['wind_100m_power']['value']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print("Excluded wind_100m power density with " +
                  f"{exclusion_dict['wind_100m_power']['source']} " +
                  f"{exclusion_dict['wind_100m_power']['value']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_eco_tech.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_eco_tech.append("Wind power at 100m")
        else:
            print("not excluding 100m power density", flush=True)
        if exclusion_dict.get("elevation") is not None:
            excluded_area = self._exclude_features(exclusion_dict["elevation"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "elevation_"
                                                                             f"{exclusion_dict['elevation']['source']}_"
                                                                             f"{exclusion_dict['elevation']['value']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(
                "Excluded elevation with " +
                f"{exclusion_dict['elevation']['source']} " +
                f"{exclusion_dict['elevation']['value']} after " +
                f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_eco_tech.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_eco_tech.append("Elevation")
        else:
            print("not excluding elevation", flush=True)
        if exclusion_dict.get("slope") is not None:
            slope_dict = exclusion_dict["slope"].copy()
            # convert slope (degree) in DN (copernicus)
            slope_dict["value"] = (exclusion_dict['slope']['value'][1],
                                   250 * np.cos(np.pi / 180 * exclusion_dict['slope']['value'][0]))
            excluded_area = self._exclude_features(slope_dict, _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "slope_"
                                                                             f"{exclusion_dict['slope']['source']}_"
                                                                             f"{exclusion_dict['slope']['value']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            # slope_raster = gk.raster.gradient(elevation_raster, factor=1)
            print("Excluded slope with " +
                  f"{exclusion_dict['slope']['source']} " +
                  f"{exclusion_dict['slope']['value']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_eco_tech.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_eco_tech.append("Slope")
        else:
            print("not excluding slope", flush=True)

        if exclusion_dict.get("birds") is not None:
            # 300m to Bird protection areas and "Naturschutzgebieten"
            excluded_area = self._exclude_features(exclusion_dict["birds"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "birds_"
                                                                             f"{exclusion_dict['birds']['source']}_"
                                                                             f"{exclusion_dict['birds']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded birds protected areas with " +
                  f"{exclusion_dict['birds']['source']} " +
                  f"{exclusion_dict['birds']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("Birds")
        else:
            print("not excluding birds protected areas", flush=True)

        if exclusion_dict.get("nature_protection") is not None:
            excluded_area = self._exclude_features(exclusion_dict["nature_protection"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "nature_protection_"
                                                                             f"{exclusion_dict['nature_protection']['source']}_"
                                                                             f"{exclusion_dict['nature_protection']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded nature_protection protected areas with " +
                  f"{exclusion_dict['nature_protection']['source']} " +
                  f"{exclusion_dict['nature_protection']['buffer']}"
                  f" after {(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("Nature protection areas")
        else:
            print("not excluding nature_protection", flush=True)

        if exclusion_dict.get("nationalpark") is not None:
            excluded_area = self._exclude_features(exclusion_dict["nationalpark"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "nationalpark_"
                                                                             f"{exclusion_dict['nationalpark']['source']}_"
                                                                             f"{exclusion_dict['nationalpark']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded national park protected areas with " +
                  f"{exclusion_dict['nationalpark']['source']} " +
                  f"{exclusion_dict['nationalpark']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("National parks")
        else:
            print("not excluding nationa lpark protected areas", flush=True)

        if exclusion_dict.get("habitats") is not None:
            excluded_area = self._exclude_features(exclusion_dict["habitats"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "habitats_"
                                                                             f"{exclusion_dict['habitats']['source']}_"
                                                                             f"{exclusion_dict['habitats']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded habitats protected areas with " +
                  f"{exclusion_dict['habitats']['source']} " +
                  f"{exclusion_dict['habitats']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("Habitats")
        else:
            print("not excluding habitats protected areas", flush=True)

        if exclusion_dict.get("landscape") is not None:
            excluded_area = self._exclude_features(exclusion_dict["landscape"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "landscape_"
                                                                             f"{exclusion_dict['landscape']['source']}_"
                                                                             f"{exclusion_dict['landscape']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded landscape protected areas with " +
                  f"{exclusion_dict['landscape']['source']} " +
                  f"{exclusion_dict['landscape']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("Landscape protected areas")
        else:
            print("not excluding landscape protected areas", flush=True)
        if exclusion_dict.get("biospheres_core") is not None:
            excluded_area = self._exclude_features(exclusion_dict["biospheres_core"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "biospheres_core_"
                                                                             f"{exclusion_dict['biospheres_core']['source']}_"
                                                                             f"{exclusion_dict['biospheres_core']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded biospheres core zone with " +
                  f"{exclusion_dict['biospheres_core']['source']} " +
                  f"{exclusion_dict['biospheres_core']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("Biospheres_core")
        else:
            print("not excluding biospheres core zone", flush=True)

        if exclusion_dict.get("biospheres_develop") is not None:
            excluded_area = self._exclude_features(exclusion_dict["biospheres_develop"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "biospheres_develop_"
                                                                             f"{exclusion_dict['biospheres_develop']['source']}_"
                                                                             f"{exclusion_dict['biospheres_develop']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded biospheres development zone with " +
                  f"{exclusion_dict['biospheres_develop']['source']} " +
                  f"{exclusion_dict['biospheres_develop']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("Biospheres_develop")
        else:
            print("not excluding biospheres development zone", flush=True)

        if exclusion_dict.get("biospheres_maintain") is not None:
            excluded_area = self._exclude_features(exclusion_dict["biospheres_maintain"], _ec,
                                                   intermediate=os.path.join(_intermediate_path,
                                                                             "biospheres_maintain_"
                                                                             f"{exclusion_dict['biospheres_maintain']['source']}_"
                                                                             f"{exclusion_dict['biospheres_maintain']['buffer']}_"
                                                                             f"{self.parent._id}.tif"
                                                                             ),
                                                   plot_sankey=plot_sankey
                                                   )
            print(f"Excluded biospheres maintaining zone with " +
                  f"{exclusion_dict['biospheres_maintain']['source']} " +
                  f"{exclusion_dict['biospheres_maintain']['buffer']} after " +
                  f"{(time.time() - start) / 60} minutes",
                  flush=True)
            if use_net_flows:
                excluded_area = remaining_area - _ec.areaAvailable
            excluded_areas.append(remaining_area - _ec.areaAvailable)
            excluded_areas_protected.append(excluded_area)
            remaining_area = _ec.areaAvailable
            labels_protected.append("Biospheres_maintain")
        else:
            print("not excluding biospheres maintaining zone", flush=True)

        # exclude regional features
        if exclusion_dict.get("auxiliary") is not None:
            aux_exclusion_dict = exclusion_dict["auxiliary"]
            aux_keys = aux_exclusion_dict.keys()
            for key in aux_keys:
                # function to run the exclusion
                special_feature_dict = aux_exclusion_dict[key]
                if os.path.isfile(special_feature_dict["source_path"]):
                    # if full path is given, accept it
                    pass
                else:
                    # otherwise it should be the location in datasources
                    special_feature_dict["source_path"] = os.path.join(_datasources_path,
                                                                       special_feature_dict["source_path"])
                    if not os.path.isfile(special_feature_dict["source_path"]):
                        warnings.warn(f"Cannot open {special_feature_dict['source_path']}. Give full path, or the path"
                                      f" in the datasources dictionary")
                        print(f"Cannot exclude {key}")
                        continue
                # TODO raster data can only use tuple and single number as 'value'
                if special_feature_dict["type"] == "raster":
                    if isinstance(special_feature_dict["value"], list):
                        special_feature_dict["value"] = tuple(special_feature_dict["value"])
                    intermediate = os.path.join(_intermediate_path,
                                                f"{key}_"
                                                f"{special_feature_dict['value']}_"
                                                f"{self.parent._id}.tif")
                elif special_feature_dict["type"] == "vector":
                    intermediate = os.path.join(_intermediate_path,
                                                f"{key}_"
                                                f"{special_feature_dict['buffer']}_"
                                                f"{self.parent._id}.tif")
                else:
                    warnings.warn(f"Unrecognized data type {special_feature_dict['type']}")
                    continue
                excluded_area = self._exclude_regional_features(special_feature_dict, _ec, intermediate=intermediate,
                                                                plot_sankey=plot_sankey)
                if use_net_flows:
                    excluded_area = remaining_area - _ec.areaAvailable
                remaining_area = _ec.areaAvailable
                excluded_areas_others.append(excluded_area)
                labels_others.append(f"{key}")
                if special_feature_dict["type"] == "vector":
                    print(f"Excluded {key} with " +
                          f"{special_feature_dict['buffer']} after " +
                          f"{(time.time() - start) / 60} minutes",
                          flush=True)
                elif special_feature_dict["type"] == "raster":
                    print(f"Excluded {key} with " +
                          f"{special_feature_dict['value']} after " +
                          f"{(time.time() - start) / 60} minutes",
                          flush=True)
        if exclusion_dict.get("region_edge") is not None:
            _ec.excludeRegionEdge(exclusion_dict.get("region_edge"))
            print("Excluded region edge with " +
                  f"{exclusion_dict['region_edge']} after " +
                  f"{(time.time() - start) / 60} minutes", flush=True)
            # excluded_areas.append(remaining_area - _ec.areaAvailable)
            # remaining_area = _ec.areaAvailable
            # labels.append("Region edge")
        else:
            print("not excluding region buffer", flush=True)
        if plot_sankey:
            remaining_area = _ec.areaAvailable
            excluded_areas_social, labels_social = self._cover_small_excluded_area(excluded_areas_social, labels_social,
                                                                                   init_available_areas, "Social")
            excluded_areas_infrastructure, labels_infrastructure = \
                self._cover_small_excluded_area(excluded_areas_infrastructure,
                                                labels_infrastructure,
                                                init_available_areas, "Infrastructure")
            excluded_areas_physical, labels_physical = self._cover_small_excluded_area(excluded_areas_physical,
                                                                                       labels_physical,
                                                                                       init_available_areas,
                                                                                       "Physical")
            excluded_areas_eco_tech, labels_eco_tech = self._cover_small_excluded_area(excluded_areas_eco_tech,
                                                                                       labels_eco_tech,
                                                                                       init_available_areas,
                                                                                       "Eco & Tech")
            excluded_areas_protected, labels_protected = self._cover_small_excluded_area(excluded_areas_protected,
                                                                                         labels_protected,
                                                                                         init_available_areas,
                                                                                         "Protected")
            excluded_areas_others, labels_others = self._cover_small_excluded_area(excluded_areas_others, labels_others,
                                                                                   init_available_areas,
                                                                                   "Others")
            value = [sum(excluded_areas_social),
                     sum(excluded_areas_infrastructure),
                     sum(excluded_areas_physical),
                     sum(excluded_areas_eco_tech),
                     sum(excluded_areas_protected),
                     sum(excluded_areas_others)] + \
                    excluded_areas_social + \
                    excluded_areas_infrastructure + \
                    excluded_areas_physical + \
                    excluded_areas_eco_tech + \
                    excluded_areas_protected + \
                    excluded_areas_others + \
                    [remaining_area] * 2
            labels = ["Initial available area",
                      "Social",
                      "Infrastructure",
                      "Physical",
                      "Economical & Technical",
                      "Protected",
                      "Others"] + \
                     labels_social + \
                     labels_infrastructure + \
                     labels_physical + \
                     labels_eco_tech + \
                     labels_protected + \
                     labels_others + \
                     ["Remaining Area"] * 2
            # add values to label
            for i in range(len(labels)):
                if i == 0:
                    continue
                if not use_net_flows and (i <= 6):
                    continue
                if value[i - 1] / init_available_areas * 100 < 0.1:
                    labels[i] = f"{labels[i]} <0.1%"
                else:
                    labels[i] = f"{labels[i]} {round(value[i - 1] / init_available_areas * 100, 1)}%"

            value += excluded_areas_social + \
                     excluded_areas_infrastructure + \
                     excluded_areas_physical + \
                     excluded_areas_eco_tech + \
                     excluded_areas_protected + \
                     excluded_areas_others + \
                     [remaining_area]
            value = [int(v) for v in value]
            remaining_percentage = round(remaining_area / init_available_areas * 100, 1)
            labels += [f"Excluded Area {100 - remaining_percentage}%"] + [f"Remaining Area {remaining_percentage}%"]

            color_social = 'rgba(200, 0, 0, 0.8)'
            color_infrastructure = 'rgba(200,200,0, 0.8)'
            color_physical = 'rgba(0,0,200, 0.8)'
            color_eco_tech = 'rgba(0, 160, 240, 0.8)'
            color_protected = 'rgba(0, 240, 0, 0.8)'
            color_others = 'rgba(80, 80, 80, 0.8)'
            color_remaining = 'rgba(0, 240, 160, 0.8)'
            color_exclusion = 'rgba(128, 128, 128, 0.8)'
            color = ['rgba(0, 0, 0, 0.8)',
                     color_social,
                     color_infrastructure,
                     color_physical,
                     color_eco_tech,
                     color_protected,
                     color_others] + \
                    [color_social] * len(excluded_areas_social) + \
                    [color_infrastructure] * len(excluded_areas_infrastructure) + \
                    [color_physical] * len(excluded_areas_physical) + \
                    [color_eco_tech] * len(excluded_areas_eco_tech) + \
                    [color_protected] * len(excluded_areas_protected) + \
                    [color_others] * len(excluded_areas_others) + \
                    [color_remaining] * 2 + \
                    [color_exclusion] + \
                    [color_remaining]
            source = [0] * 6 + \
                     [1] * len(excluded_areas_social) + \
                     [2] * len(excluded_areas_infrastructure) + \
                     [3] * len(excluded_areas_physical) + \
                     [4] * len(excluded_areas_eco_tech) + \
                     [5] * len(excluded_areas_protected) + \
                     [6] * len(excluded_areas_others) + \
                     [0, len(labels) - 4] + \
                     list(range(len(labels)))[7:-4] + \
                     [len(labels) - 3]
            target = list(range(len(labels)))[1:-3] + \
                     [len(labels) - 3] + \
                     [len(labels) - 2] * len(list(range(len(labels)))[7:-4]) + \
                     [len(labels) - 1]
            color_link_social = 'rgba(200, 0, 0, 0.4)'
            color_link_infrastructure = 'rgba(200,200,0, 0.4)'
            color_link_physical = 'rgba(0,0,200, 0.4)'
            color_link_eco_tech = 'rgba(0, 160, 240, 0.4)'
            color_link_protected = 'rgba(0, 240, 0, 0.4)'
            color_link_others = 'rgba(80, 80, 80, 0.4)'
            color_link_remaining = 'rgba(0, 240, 160, 0.4)'
            color_link_exclusion = 'rgba(128, 128, 128, 0.4)'
            color_link = [
                             color_link_social,
                             color_link_infrastructure,
                             color_link_physical,
                             color_link_eco_tech,
                             color_link_protected,
                             color_link_others] + \
                         [color_link_social] * len(excluded_areas_social) + \
                         [color_link_infrastructure] * len(excluded_areas_infrastructure) + \
                         [color_link_physical] * len(excluded_areas_physical) + \
                         [color_link_eco_tech] * len(excluded_areas_eco_tech) + \
                         [color_link_protected] * len(excluded_areas_protected) + \
                         [color_link_others] * len(excluded_areas_others) + \
                         [color_link_remaining] * 2 + \
                         [color_link_exclusion] * len(list(range(len(labels)))[7:-4]) + \
                         [color_link_remaining]
            node = dict(
                pad=15,
                thickness=10,
                line=dict(color="black", width=0.5),
                label=labels,
                color=color
            )
            link = dict(
                source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                target=target,
                value=value,
                color=color_link
            )
            data = dict(
                node=node,
                link=link
            )
            with open(os.path.join(self.result_path, "sankey_config.json"), "w") as f:
                json.dump(data, f, indent=2)
        return exclusion_dict

    @staticmethod
    def _cover_small_excluded_area(excluded_areas, labels, init_available_areas, category_name):
        small_excluded_area = 0
        excluded_areas_new = []
        labels_new = []
        for i in range(len(excluded_areas)):
            if excluded_areas[i] / init_available_areas * 100 < 0.2:
                small_excluded_area += excluded_areas[i]
            else:
                excluded_areas_new.append(excluded_areas[i])
                labels_new.append(labels[i])
        if small_excluded_area > 0:
            excluded_areas_new.append(small_excluded_area)
            labels_new.append(f"{category_name} others")
        return excluded_areas_new, labels_new

    def plot_sankey(self, config_dict="sankey_config", plot_in_browser=False, format="pdf"):
        """
        Draw the sankey plot based on the configuration

        Parameters
        ----------
        config_dict : dict or str
            Dictionary containing the configuration of sankey plot
            By default the sankey configuration in database will be used
        plot_in_browser : bool
            Whether to show the plot in browser
        format : str
            The format of the sankey plot, e.g. "pdf", "syg", "png" ect.

        """
        if isinstance(config_dict, dict):
            data = config_dict
        elif isinstance(config_dict, str):
            if os.path.isfile(os.path.join(self.result_path, f"{config_dict}.json")):
                with open(os.path.join(self.result_path, f"{config_dict}.json"), "r") as f:
                    data = json.load(f)
            else:
                # full path is given
                with open(config_dict, "r") as f:
                    data = json.load(f)

        fig = go.Figure(
            data=[go.Sankey(node=data["node"], link=data["link"])]
        )
        fig.update_layout(title_text="Area flows", font_size=7)
        if plot_in_browser:
            fig.show(renderer="browser")
        fig.write_image(os.path.join(self.result_path, f"Area_Flow.{format}"))

    def _get_exclusion_condition(self, exclusion_dict: Dict) -> Dict:
        """
        Read information from the exclusion dictionary to judge which data source to use.
        And then write the corresponding query text to the exclusion dictionary
        Parameters
        ----------
        exclusion_dict : dict
            Dictionary containing the information for the exclusion
        """
        # TODO for raster data, the "value" should be given in [min, max]
        # TODO there are wind atlas for some federal states, do we need to use them
        # TODO implement data type dict into run_exclusion. Additional function may help, like _exclude
        # TODO add correct type hint
        # TODO check for all categories if the "ausser Betrieb" should be considered
        # TODO do we need to use polygon data for all line features?
        _datasources_path = self.parent.datasource_path
        # _osm_path = os.path.join(self.parent.datasource_path, utils.get_osm_path(self.parent._state))
        _osm_path = os.path.join(self.parent.datasource_path, os.path.join("osm", "merged"))
        _osm_overpass_path = os.path.join(
            _datasources_path, "osm_overpass")
        _clcVectorPath = os.path.join(
            _datasources_path, "clc", "CLC2018_v2020_20u1_DE_fgdb.shp")
        _dlm250_path = os.path.join(
            _datasources_path, "dlm250")
        _dlm250_path = os.path.join(
            _datasources_path, "dlm250.utm32s.shape.ebenen", "dlm250.utm32s.shape.ebenen", "dlm250_ebenen",
            "de", "dlm250")
        _basis_dlm_path = self.parent.dlm_basis_path
        _hu_path = os.path.join(self.parent.hu_path, self.parent.state, f"hu_{self.parent.state}.shp")
        _wdpa_de_path = os.path.join(_datasources_path, "wdpa", "WDPA_DE.shp")
        _bfn_path = os.path.join(_datasources_path, "bfn")
        if exclusion_dict.get("border") is not None:
            if exclusion_dict['border'].get('source') is not None:
                if exclusion_dict['border']['source'] in ("vg250",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['border']['source']} is not supported for border!")
                    exclusion_dict['border']['source'] = "vg250"
            else:
                # by default use vg250
                exclusion_dict['border']['source'] = "vg250"
            if exclusion_dict['border']['source'] == "vg250":
                exclusion_dict['border']['path'] = os.path.join(
                    self.parent.datasource_path,
                    "border",
                    "borders.shp")
                exclusion_dict['border']['where_text'] = None

        if exclusion_dict.get("airports") is not None:
            if exclusion_dict['airports'].get('source') is not None:
                if exclusion_dict['airports']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['airports']['source']} is not supported for airports!")
                    exclusion_dict['airports']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['airports']['source'] = "basis-dlm"
            if exclusion_dict['airports']['source'] == "basis-dlm":
                exclusion_dict['airports']['path'] = os.path.join(_basis_dlm_path, "ver04_f.shp")
                exclusion_dict['airports']['where_text'] = "ART in ('5510', '5511', '5512') OR  NTZ in ('2000', " \
                                                           "'3000')  AND (ZUS IS NULL or ZUS = 'None') "
            elif exclusion_dict['airports']['source'] == "dlm250":
                exclusion_dict['airports']['path'] = os.path.join(_dlm250_path, "ver04_f.shp")
                exclusion_dict['airports']['where_text'] = "ART in ('5510', '5511', '5512') OR  NTZ in ('2000', " \
                                                           "'3000')  AND (ZUS IS NULL or ZUS = 'None') "
            elif exclusion_dict['airports']['source'] == "osm":
                exclusion_dict['airports']['path'] = os.path.join(_osm_path, "gis_osm_transport_a_free_1.shp")
                exclusion_dict['airports']['where_text'] = "fclass = 'airport'"
            elif exclusion_dict['airports']['source'] == "clc":
                exclusion_dict['airports']['path'] = _clcVectorPath
                exclusion_dict['airports']['where_text'] = "Code_18='124'"

        if exclusion_dict.get("airfields") is not None:
            # airfield and glider airfield
            if exclusion_dict['airfields'].get('source') is not None:
                if exclusion_dict['airfields']['source'] in ("basis-dlm", "osm"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['airfields']['source']} is not supported for airfields!")
                    exclusion_dict['airfields']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['airfields']['source'] = "basis-dlm"
            if exclusion_dict['airfields']['source'] == "basis-dlm":
                exclusion_dict['airfields']['path'] = os.path.join(_basis_dlm_path, "ver04_f.shp")
                exclusion_dict['airfields'][
                    'where_text'] = "ART in ('5520', '5540', '5550') AND (ZUS IS NULL or ZUS = 'None')"
            elif exclusion_dict['airfields']['source'] == "osm":
                exclusion_dict['airfields']['path'] = os.path.join(_osm_path, "gis_osm_transport_a_free_1.shp")
                exclusion_dict['airfields']['where_text'] = "fclass = 'airfield' OR fclass = 'apron'"

        if exclusion_dict.get("health_treatment_buildings") is not None:
            if exclusion_dict['health_treatment_buildings'].get('source') is not None:
                if exclusion_dict['health_treatment_buildings']['source'] in ("hu", "osm"):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['health_treatment_buildings']['source']} is not supported for health_treatment_buildings!")
                    exclusion_dict['health_treatment_buildings']['source'] = "hu"
            else:
                # by default use hu
                exclusion_dict['health_treatment_buildings']['source'] = "hu"
            if exclusion_dict['health_treatment_buildings']['source'] == "hu":
                exclusion_dict['health_treatment_buildings']['path'] = _hu_path
                exclusion_dict['health_treatment_buildings']['where_text'] = \
                    "GFK IN ('31001_3240', '31001_3241', '31001_3242', " + "'31001_3051', '31001_3052') "
            elif exclusion_dict['health_treatment_buildings']['source'] == "osm":
                exclusion_dict['health_treatment_buildings']['path'] = os.path.join(_osm_path,
                                                                                    "gis_osm_pois_a_free_1.shp")
                exclusion_dict['health_treatment_buildings']['where_text'] = "fclass = 'hospital'"
        if exclusion_dict.get("buildings") is not None:
            # residential buildings
            if exclusion_dict['buildings'].get('source') is not None:
                if exclusion_dict['buildings']['source'] in ("hu",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['buildings']['source']} is not supported for buildings!")
                    exclusion_dict['buildings']['source'] = "hu"
            else:
                # by default use hu
                exclusion_dict['buildings']['source'] = "hu"
            if exclusion_dict['buildings']['source'] == "hu":
                exclusion_dict['buildings']['path'] = _hu_path
                exclusion_dict['buildings']['where_text'] = "GFK IN ('31001_1000', '31001_1010', '31001_1020', " + \
                                                            "'31001_1021', '31001_1022', '31001_1023', " + \
                                                            "'31001_1024', '31001_1025', '31001_1210', " + \
                                                            "'31001_3064', '31001_3066', '31001_2070', " + \
                                                            "'31001_2071', '31001_2072', '31001_2074')"
                # TODO: @j.du keys documentation

        if exclusion_dict.get("buildings_commercial") is not None:
            # commercial buildings
            if exclusion_dict['buildings_commercial'].get('source') is not None:
                if exclusion_dict['buildings_commercial']['source'] in ("hu",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['buildings_commercial']['source']} is not supported for commercial buildings!")
                    exclusion_dict['buildings']['source'] = "hu"
            else:
                # by default use hu
                exclusion_dict['buildings_commercial']['source'] = "hu"
            if exclusion_dict['buildings_commercial']['source'] == "hu":
                exclusion_dict['buildings_commercial']['path'] = _hu_path
                exclusion_dict['buildings_commercial'][
                    'where_text'] = "GFK IN ('31001_2000', '31001_2010', '31001_2020', '31001_2030'," \
                                    "'31001_2040', '31001_2050', '31001_2051', '31001_2052'," \
                                    "'31001_2053', '31001_2054', '31001_2055', '31001_2056'," \
                                    "'31001_2060', '31001_2070', '31001_2071', '31001_2072'," \
                                    "'31001_2073', '31001_2074', '31001_2080', '31001_2081'," \
                                    "'31001_2082', '31001_2083', '31001_2090', '31001_2091'," \
                                    "'31001_2092', '31001_2093', '31001_2094')"

        if exclusion_dict.get("buildings_all") is not None:
            # all buildings
            if exclusion_dict['buildings_all'].get('source') is not None:
                if exclusion_dict['buildings_all']['source'] in ("hu",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['buildings_all']['source']} is not supported for buildings_all!")
                    exclusion_dict['buildings_all']['source'] = "hu"
            else:
                # by default use hu
                exclusion_dict['buildings_all']['source'] = "hu"
            if exclusion_dict['buildings_all']['source'] == "hu":
                exclusion_dict['buildings_all']['path'] = _hu_path
                exclusion_dict['buildings_all']['where_text'] = None

        if exclusion_dict.get("mixed_buildings") is not None:
            # Urban fabric
            if exclusion_dict['mixed_buildings'].get('source') is not None:
                if exclusion_dict['mixed_buildings']['source'] in ("hu",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['mixed_buildings']['source']} is not supported for mixed_buildings!")
                    exclusion_dict['mixed_buildings']['source'] = "hu"
            else:
                # by default use hu
                exclusion_dict['mixed_buildings']['source'] = "hu"
            if exclusion_dict['mixed_buildings']['source'] == "hu":
                exclusion_dict['mixed_buildings']['path'] = _hu_path
                exclusion_dict['mixed_buildings']['where_text'] = "GFK IN ('31001_1100', '31001_1110', '31001_1120', " + \
                                                                  "'31001_1121', '31001_1122', '31001_1123', " + \
                                                                  "'31001_1130', '31001_1220', '31001_1221', " + \
                                                                  "'31001_1223')"

        if exclusion_dict.get("water_still") is not None:
            if exclusion_dict['water_still'].get('source') is not None:
                if exclusion_dict['water_still']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['water_still']['source']} is not supported for water_still!")
                    exclusion_dict['water_still']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['water_still']['source'] = "basis-dlm"
            if exclusion_dict['water_still']['source'] == "basis-dlm":
                exclusion_dict['water_still']['path'] = os.path.join(_basis_dlm_path, "gew01_f.shp")
                exclusion_dict['water_still']['where_text'] = "OBJART_TXT='AX_Hafenbecken' OR OBJART_TXT='AX_Meer' OR " \
                                                              "OBJART_TXT='AX_StehendesGewaesser' "
            elif exclusion_dict['water_still']['source'] == "dlm250":
                exclusion_dict['water_still']['path'] = os.path.join(_dlm250_path, "gew01_f.shp")
                exclusion_dict['water_still']['where_text'] = "OBJART_TXT='AX_Hafenbecken' OR OBJART_TXT='AX_Meer' OR " \
                                                              "OBJART_TXT='AX_StehendesGewaesser' "
            elif exclusion_dict['water_still']['source'] == "osm":
                exclusion_dict['water_still']['path'] = os.path.join(_osm_path, "gis_osm_water_a_free_1.shp")
                exclusion_dict['water_still']['where_text'] = "fclass!='river' OR fclass!='drain' OR fclass!='canal' " + \
                                                              "OR fclass!='tidal_channel' OR fclass!='riverbank'"
            elif exclusion_dict['water_still']['source'] == "clc":
                exclusion_dict['water_still']['path'] = _clcVectorPath
                exclusion_dict['water_still']['where_text'] = "Code_18='512' OR Code_18='521' OR Code_18='523'"

        if exclusion_dict.get("water_river") is not None:
            if exclusion_dict['water_river'].get('source') is not None:
                if exclusion_dict['water_river']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['water_river']['source']} is not supported for water_river!")
                    exclusion_dict['water_river']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['water_river']['source'] = "basis-dlm"
            if exclusion_dict['water_river']['source'] == "basis-dlm":
                exclusion_dict['water_river']['path'] = os.path.join(_basis_dlm_path, "gew01_f.shp")
                exclusion_dict['water_river']['where_text'] = "OBJART_TXT='AX_Fliessgewaesser' OR " \
                                                              "OBJART_TXT='AX_Kanal' OR OBJART_TXT='AX_Wasserlauf' OR " \
                                                              "OBJART_TXT='AX_Gewaesserachse' "
            elif exclusion_dict['water_river']['source'] == "dlm250":
                exclusion_dict['water_river']['path'] = os.path.join(_dlm250_path, "gew01_f.shp")
                exclusion_dict['water_river']['where_text'] = "OBJART_TXT='AX_Fliessgewaesser' OR " \
                                                              "OBJART_TXT='AX_Kanal' OR OBJART_TXT='AX_Wasserlauf' OR " \
                                                              "OBJART_TXT='AX_Gewaesserachse' "
            elif exclusion_dict['water_river']['source'] == "osm":
                exclusion_dict['water_river']['path'] = os.path.join(_osm_path, "gis_osm_waterways_free_1.shp")
                exclusion_dict['water_river']['where_text'] = "fclass='river' OR fclass='drain' OR fclass='canal' " + \
                                                              "OR fclass='tidal_channel' OR fclass='riverbank'"
            elif exclusion_dict['water_river']['source'] == "clc":
                exclusion_dict['water_river']['path'] = _clcVectorPath
                exclusion_dict['water_river']['where_text'] = "Code_18='511' OR Code_18='522'"

        if exclusion_dict.get("water_stream") is not None:
            if exclusion_dict['water_stream'].get('source') is not None:
                if exclusion_dict['water_stream']['source'] in ("osm",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['water_stream']['source']} is not supported for water_stream!")
                    exclusion_dict['water_stream']['source'] = "osm"
            else:
                # by default use osm
                exclusion_dict['water_stream']['source'] = "osm"
            if exclusion_dict['water_stream']['source'] == "osm":
                exclusion_dict['water_stream']['path'] = os.path.join(_osm_path, "gis_osm_waterways_free_1.shp")
                exclusion_dict['water_stream']['where_text'] = "fclass='stream' OR fclass='ditch'"

        # if all distances from roads are the same --> exclude all together
        # TODO Is it necessary to implement like this?
        # _roads = ["motorway", "trunk", "primary_roads", "medium_roads"]
        # _eq_roads = []
        # for key in exclusion_dict.keys():
        #     for _key in exclusion_dict.keys():
        #         if key in _roads and _key in _roads:
        #             _eq_roads.append(
        #                 exclusion_dict[key] == exclusion_dict[_key])
        #
        # _road_shape_path = os.path.join(_osm_path,
        #                                 "gis_osm_roads_free_1.shp")
        # if all(_eq_roads):
        #
        #
        #     if exclusion_dict.get("motorway") is not None:
        #         if exclusion_dict['airports'].get('source') is not None:
        #             if exclusion_dict['airports']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
        #                 pass
        #             else:
        #                 warnings.warn(f"{exclusion_dict['airports']['source']} is not supported for airports!")
        #                 exclusion_dict['airports']['source'] = "basis-dlm"
        #         else:
        #             # by default use basis-dlm
        #             exclusion_dict['airports']['source'] = "basis-dlm"
        #         if exclusion_dict['airports']['source'] == "basis-dlm":
        #             exclusion_dict['airports']['path'] = os.path.join(_basis_dlm_path, "ver04_f.shp")
        #             exclusion_dict['airports']['where_text'] = None
        #         elif exclusion_dict['airports']['source'] == "dlm250":
        #             exclusion_dict['airports']['path'] = os.path.join(_dlm250_path, "ver04_f.shp")
        #             exclusion_dict['airports']['where_text'] = None
        #         elif exclusion_dict['airports']['source'] == "osm":
        #             exclusion_dict['airports']['path'] = os.path.join(_osm_path, "gis_osm_transport_a_free_1.shp")
        #             exclusion_dict['airports']['where_text'] = "fclass = 'airport' OR fclass = 'airfield' OR fclass = " \
        #                                                        "'apron' "
        #         elif exclusion_dict['airports']['source'] == "clc":
        #             exclusion_dict['airports']['path'] = _clcVectorPath
        #             exclusion_dict['airports']['where_text'] = "Code_18='124'"
        #         _ec.excludeVectorType(
        #             _road_shape_path,
        #             where="fclass='motorway' OR fclass='trunk' OR" +
        #                   " fclass='primary' OR fclass='secondary' OR" +
        #                   " fclass='tertiary' OR fclass='motorway_link'",
        #             buffer=exclusion_dict["motorway"])
        #     print("Excluded all streets with " +
        #           f"{exclusion_dict['motorway']} after " +
        #           f"{(time.time() - start) / 60} minutes", flush=True)
        # else:
        # TODO Use line polygon instead?
        if exclusion_dict.get("motorway") is not None:
            if exclusion_dict['motorway'].get('source') is not None:
                if exclusion_dict['motorway']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['motorway']['source']} is not supported for motorway!")
                    exclusion_dict['motorway']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['motorway']['source'] = "basis-dlm"
            if exclusion_dict['motorway']['source'] == "basis-dlm":
                exclusion_dict['motorway']['path'] = (
                    os.path.join(_basis_dlm_path, "ver01_l.shp"),
                    os.path.join(_basis_dlm_path, "ver01_f.shp")
                )
                exclusion_dict['motorway']['where_text'] = (
                    "WDM = '1301'",
                    "OBJART_TXT = 'AX_Platz' and FKT != '5310'"
                )

            elif exclusion_dict['motorway']['source'] == "dlm250":
                exclusion_dict['motorway']['path'] = (
                    os.path.join(_basis_dlm_path, "ver01_l.shp"),
                    os.path.join(_basis_dlm_path, "ver01_f.shp")
                )
                exclusion_dict['motorway']['where_text'] = (
                    "WDM = '1301'",
                    "OBJART_TXT = 'AX_Platz' and FKT != '5310'"
                )
            elif exclusion_dict['motorway']['source'] == "osm":
                exclusion_dict['motorway']['path'] = os.path.join(_osm_path, "gis_osm_roads_free_1.shp")
                exclusion_dict['motorway']['where_text'] = "fclass = 'motorway' OR fclass = 'motorway_link'"

        if exclusion_dict.get("primary_roads") is not None:
            if exclusion_dict['primary_roads'].get('source') is not None:
                if exclusion_dict['primary_roads']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['primary_roads']['source']} is not supported for primary_roads!")
                    exclusion_dict['primary_roads']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['primary_roads']['source'] = "basis-dlm"
            if exclusion_dict['primary_roads']['source'] == "basis-dlm":
                exclusion_dict['primary_roads']['path'] = os.path.join(_basis_dlm_path, "ver01_l.shp")
                exclusion_dict['primary_roads']['where_text'] = "WDM = '1303'"
            elif exclusion_dict['primary_roads']['source'] == "dlm250":
                exclusion_dict['primary_roads']['path'] = os.path.join(_dlm250_path, "ver01_l.shp")
                exclusion_dict['primary_roads']['where_text'] = "WDM = '1303'"
            # TODO see if "trunk" should be considered here
            elif exclusion_dict['primary_roads']['source'] == "osm":
                exclusion_dict['primary_roads']['path'] = os.path.join(_osm_path, "gis_osm_roads_free_1.shp")
                exclusion_dict['primary_roads']['where_text'] = "fclass = 'primary' OR fclass = 'primary_link' " \
                                                                "OR fclass = 'trunk' OR fclass = 'trunk_link'"

        if exclusion_dict.get("secondary_roads") is not None:
            if exclusion_dict['secondary_roads'].get('source') is not None:
                if exclusion_dict['secondary_roads']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['secondary_roads']['source']} is not supported for secondary_roads!")
                    exclusion_dict['secondary_roads']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['secondary_roads']['source'] = "basis-dlm"
            if exclusion_dict['secondary_roads']['source'] == "basis-dlm":
                exclusion_dict['secondary_roads']['path'] = os.path.join(_basis_dlm_path, "ver01_l.shp")
                exclusion_dict['secondary_roads']['where_text'] = "WDM = '1305'"
            elif exclusion_dict['secondary_roads']['source'] == "dlm250":
                exclusion_dict['secondary_roads']['path'] = os.path.join(_dlm250_path, "ver01_l.shp")
                exclusion_dict['secondary_roads']['where_text'] = "WDM = '1305'"
            elif exclusion_dict['secondary_roads']['source'] == "osm":
                exclusion_dict['secondary_roads']['path'] = os.path.join(_osm_path, "gis_osm_roads_free_1.shp")
                exclusion_dict['secondary_roads']['where_text'] = "fclass = 'secondary' OR fclass = 'secondary_link'"

        if exclusion_dict.get("regional_roads") is not None:
            if exclusion_dict['regional_roads'].get('source') is not None:
                if exclusion_dict['regional_roads']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['regional_roads']['source']} is not supported for regional_roads!")
                    exclusion_dict['regional_roads']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['regional_roads']['source'] = "basis-dlm"
            if exclusion_dict['regional_roads']['source'] == "basis-dlm":
                exclusion_dict['regional_roads']['path'] = os.path.join(_basis_dlm_path, "ver01_l.shp")
                exclusion_dict['regional_roads']['where_text'] = "WDM != '1301' AND WDM != '1303' AND WDM != '1305'"
            elif exclusion_dict['regional_roads']['source'] == "dlm250":
                exclusion_dict['regional_roads']['path'] = os.path.join(_dlm250_path, "ver01_l.shp")
                exclusion_dict['regional_roads']['where_text'] = "WDM != '1301' AND WDM != '1303' AND WDM != '1305'"
            elif exclusion_dict['regional_roads']['source'] == "osm":
                exclusion_dict['regional_roads']['path'] = os.path.join(_osm_path, "gis_osm_roads_free_1.shp")
                exclusion_dict['regional_roads']['where_text'] = "fclass = 'residential'  OR fclass = 'tertiary'  " \
                                                                 "OR fclass = 'tertiary_link' "

        if exclusion_dict.get("railways") is not None:
            if exclusion_dict['railways'].get('source') is not None:
                if exclusion_dict['railways']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['railways']['source']} is not supported for railways!")
                    exclusion_dict['railways']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['railways']['source'] = "basis-dlm"
            if exclusion_dict['railways']['source'] == "basis-dlm":
                exclusion_dict['railways']['path'] = (
                    os.path.join(_basis_dlm_path, "ver03_l.shp"),
                    os.path.join(_basis_dlm_path, "ver03_f.shp"),
                    os.path.join(_basis_dlm_path, "ver06_f.shp")
                )
                exclusion_dict['railways']['where_text'] = (
                    "OBJART_TXT = 'AX_Bahnstrecke'",
                    None,
                    "OBJART_TXT = 'AX_Bahnverkehrsanlage'"
                )
            elif exclusion_dict['railways']['source'] == "dlm250":
                exclusion_dict['railways']['path'] = (
                    os.path.join(_dlm250_path, "ver03_l.shp"),
                    os.path.join(_dlm250_path, "ver03_f.shp"),
                    os.path.join(_dlm250_path, "ver06_f.shp")
                )
                exclusion_dict['railways']['where_text'] = (
                    "OBJART_TXT = 'AX_Bahnstrecke'",
                    None,
                    "OBJART_TXT = 'AX_Bahnverkehrsanlage'"
                )
            elif exclusion_dict['railways']['source'] == "osm":
                exclusion_dict['railways']['path'] = os.path.join(_osm_path, "gis_osm_railways_free_1.shp")
                exclusion_dict['railways']['where_text'] = None

        # TODO a new key for railway stations?

        if exclusion_dict.get("power_lines") is not None:
            if exclusion_dict['power_lines'].get('source') is not None:
                if exclusion_dict['power_lines']['source'] in ("basis-dlm", "osm_overpass", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['power_lines']['source']} is not supported for power_lines!")
                    exclusion_dict['power_lines']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['power_lines']['source'] = "basis-dlm"
            if exclusion_dict['power_lines']['source'] == "basis-dlm":
                exclusion_dict['power_lines']['path'] = os.path.join(_basis_dlm_path, "sie03_l.shp")
                exclusion_dict['power_lines']['where_text'] = "OBJART_TXT='AX_Leitung'"
            elif exclusion_dict['power_lines']['source'] == "dlm250":
                exclusion_dict['power_lines']['path'] = os.path.join(_dlm250_path, "sie03_l.shp")
                exclusion_dict['power_lines']['where_text'] = "OBJART_TXT='AX_Leitung'"
            # TODO: Set correct path for osm_overpass. Data still need to be acquired from the Overpass turbo.
            elif exclusion_dict['power_lines']['source'] == "osm_overpass":
                exclusion_dict['power_lines']['path'] = os.path.join(_osm_overpass_path, "power_line_OSM.shp")
                exclusion_dict['power_lines']['where_text'] = None

        if exclusion_dict.get("farmland") is not None:
            if exclusion_dict['farmland'].get('source') is not None:
                if exclusion_dict['farmland']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['farmland']['source']} is not supported for farmland!")
                    exclusion_dict['farmland']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['farmland']['source'] = "basis-dlm"
            if exclusion_dict['farmland']['source'] == "basis-dlm":
                exclusion_dict['farmland']['path'] = os.path.join(_basis_dlm_path, "veg01_f.shp")
                exclusion_dict['farmland']['where_text'] = "VEG != '1020'"
            elif exclusion_dict['farmland']['source'] == "dlm250":
                exclusion_dict['farmland']['path'] = os.path.join(_dlm250_path, "veg01_f.shp")
                exclusion_dict['farmland']['where_text'] = "VEG != '1020'"
            elif exclusion_dict['farmland']['source'] == "osm":
                exclusion_dict['farmland']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['farmland']['where_text'] = "fclass='farmland' OR fclass='orchard' OR fclass='vineyard'"
            elif exclusion_dict['farmland']['source'] == "clc":
                exclusion_dict['farmland']['path'] = _clcVectorPath
                exclusion_dict['farmland']['where_text'] = "Code_18='211' OR Code_18='212' OR Code_18='213' OR " \
                                                           "Code_18='221' OR Code_18='222' OR Code_18='223' OR " \
                                                           "Code_18='241' OR Code_18='242' OR Code_18='243' OR " \
                                                           "Code_18='244' "

        if exclusion_dict.get("grassland") is not None:
            if exclusion_dict['grassland'].get('source') is not None:
                if exclusion_dict['grassland']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['grassland']['source']} is not supported for grassland!")
                    exclusion_dict['grassland']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['grassland']['source'] = "basis-dlm"
            if exclusion_dict['grassland']['source'] == "basis-dlm":
                exclusion_dict['grassland']['path'] = os.path.join(_basis_dlm_path, "veg01_f.shp")
                exclusion_dict['grassland']['where_text'] = "VEG = '1020'"
            elif exclusion_dict['grassland']['source'] == "dlm250":
                exclusion_dict['grassland']['path'] = os.path.join(_dlm250_path, "veg01_f.shp")
                exclusion_dict['grassland']['where_text'] = "VEG = '1020'"
            elif exclusion_dict['grassland']['source'] == "osm":
                exclusion_dict['grassland']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['grassland']['where_text'] = "fclass='grass' OR fclass='meadow'"
            elif exclusion_dict['grassland']['source'] == "clc":
                exclusion_dict['grassland']['path'] = _clcVectorPath
                exclusion_dict['grassland']['where_text'] = "Code_18='231'"

        if exclusion_dict.get("forests") is not None:
            if exclusion_dict['forests'].get('source') is not None:
                if exclusion_dict['forests']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['forests']['source']} is not supported for forests!")
                    exclusion_dict['forests']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['forests']['source'] = "basis-dlm"
            if exclusion_dict['forests']['source'] == "basis-dlm":
                exclusion_dict['forests']['path'] = os.path.join(_basis_dlm_path, "veg02_f.shp")
                exclusion_dict['forests']['where_text'] = None
            elif exclusion_dict['forests']['source'] == "dlm250":
                exclusion_dict['forests']['path'] = os.path.join(_dlm250_path, "veg02_f.shp")
                exclusion_dict['forests']['where_text'] = None
            elif exclusion_dict['forests']['source'] == "osm":
                exclusion_dict['forests']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['forests']['where_text'] = "fclass='forest'"
            elif exclusion_dict['forests']['source'] == "clc":
                exclusion_dict['forests']['path'] = _clcVectorPath
                exclusion_dict['forests']['where_text'] = "Code_18='311' OR Code_18='312' OR Code_18='313'"
        if exclusion_dict.get("forests_outside_FRA") is not None:
            if exclusion_dict['forests_outside_FRA'].get('source') is not None:
                if exclusion_dict['forests_outside_FRA']['source'] in ("basis-dlm",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['forests_outside_FRA']['source']} is not supported for forests_outside_FRA!")
                    exclusion_dict['forests_outside_FRA']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['forests_outside_FRA']['source'] = "basis-dlm"
            if exclusion_dict['forests_outside_FRA']['source'] == "basis-dlm":
                exclusion_dict['forests_outside_FRA']['path'] = os.path.join(_basis_dlm_path, "forest_rich_area", "de",
                                                                             "forest_outside_forest_rich_municipality_de.shp")
                exclusion_dict['forests_outside_FRA']['where_text'] = None
        if exclusion_dict.get("forests_in_FRA_without_coniferous_forests") is not None:
            if exclusion_dict['forests_in_FRA_without_coniferous_forests'].get('source') is not None:
                if exclusion_dict['forests_in_FRA_without_coniferous_forests']['source'] in ("basis-dlm",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['forests_in_FRA_without_coniferous_forests']['source']} is not supported for forests_in_FRA_without_coniferous_forests!")
                    exclusion_dict['forests_in_FRA_without_coniferous_forests']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['forests_in_FRA_without_coniferous_forests']['source'] = "basis-dlm"
            if exclusion_dict['forests_in_FRA_without_coniferous_forests']['source'] == "basis-dlm":
                exclusion_dict['forests_in_FRA_without_coniferous_forests']['path'] = os.path.join(_basis_dlm_path,
                                                                                                   "forest_rich_area",
                                                                                                   "de",
                                                                                                   "forest_in_forest_rich_municipality_de.shp")
                exclusion_dict['forests_in_FRA_without_coniferous_forests']['where_text'] = "VEG != '1200'"

        if exclusion_dict.get("trees") is not None:
            if exclusion_dict['trees'].get('source') is not None:
                if exclusion_dict['trees']['source'] in ("basis-dlm",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['trees']['source']} is not supported for trees!")
                    exclusion_dict['trees']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['trees']['source'] = "basis-dlm"
            if exclusion_dict['trees']['source'] == "basis-dlm":
                exclusion_dict['trees']['path'] = os.path.join(_basis_dlm_path, "veg03_f.shp")
                exclusion_dict['trees']['where_text'] = "OBJART_TXT = 'AX_Gehoelz'"

        if exclusion_dict.get("inner_areas") is not None:
            if exclusion_dict['inner_areas'].get('source') is not None:
                if exclusion_dict['inner_areas']['source'] in ("basis-dlm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['inner_areas']['source']} is not supported for inner_areas!")
                    exclusion_dict['inner_areas']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['inner_areas']['source'] = "basis-dlm"
            if exclusion_dict['inner_areas']['source'] == "basis-dlm":
                exclusion_dict['inner_areas']['path'] = os.path.join(_basis_dlm_path, "sie01_f.shp")
                exclusion_dict['inner_areas']['where_text'] = None
            elif exclusion_dict['inner_areas']['source'] == "dlm250":
                exclusion_dict['inner_areas']['path'] = os.path.join(_dlm250_path, "sie01_f.shp")
                exclusion_dict['inner_areas']['where_text'] = None

        if exclusion_dict.get("outer_areas") is not None:
            if exclusion_dict['outer_areas'].get('source') is not None:
                if exclusion_dict['outer_areas']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['outer_areas']['source']} is not supported for outer_areas!")
                    exclusion_dict['outer_areas']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['outer_areas']['source'] = "basis-dlm"
            # TODO if use sie_02_f to identify outer areas, it will give wrong information to Sankey
            #  It has to be ensured that the outer area is excluded after the inner area
            if exclusion_dict['outer_areas']['source'] == "basis-dlm":
                exclusion_dict['outer_areas']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['outer_areas'][
                    'where_text'] = "OBJART !='41005' AND OBJART !='41004' AND OBJART !='41002'"
            elif exclusion_dict['outer_areas']['source'] == "dlm250":
                exclusion_dict['outer_areas']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['outer_areas'][
                    'where_text'] = "OBJART !='41005' AND OBJART !='41004' AND OBJART !='41002'"

        if exclusion_dict.get("residential") is not None:
            if exclusion_dict['residential'].get('source') is not None:
                if exclusion_dict['residential']['source'] in ("basis-dlm", "dlm250", "osm", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['residential']['source']} is not supported for residential!")
                    exclusion_dict['residential']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['residential']['source'] = "basis-dlm"
            if exclusion_dict['residential']['source'] == "basis-dlm":
                exclusion_dict['residential']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['residential']['where_text'] = "OBJART = '41001' OR (OBJART='41007' AND FKT in" \
                                                              "('1110', '1120', '1130', '1150', '1160', '1170'))"
            elif exclusion_dict['residential']['source'] == "dlm250":
                exclusion_dict['residential']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['residential']['where_text'] = "OBJART = '41001' OR (OBJART='41007' AND FKT in" \
                                                              "('1110', '1120', '1130', '1150', '1160', '1170'))"
            elif exclusion_dict['residential']['source'] == "osm":
                exclusion_dict['residential']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['residential'][
                    'where_text'] = "fclass='residential' OR fclass='retail' OR fclass='allotments'"
            elif exclusion_dict['residential']['source'] == "clc":
                exclusion_dict['residential']['path'] = _clcVectorPath
                exclusion_dict['residential']['where_text'] = "Code_18='111' OR Code_18='112' OR Code_18='133' OR " \
                                                              "Code_18='141' OR Code_18='142' "
        if exclusion_dict.get("mixed_usage") is not None:
            if exclusion_dict['mixed_usage'].get('source') is not None:
                if exclusion_dict['mixed_usage']['source'] in ("basis-dlm", "osm"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['mixed_usage']['source']} is not supported for mixed_usage!")
                    exclusion_dict['mixed_usage']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['mixed_usage']['source'] = "basis-dlm"
            if exclusion_dict['mixed_usage']['source'] == "basis-dlm":
                exclusion_dict['mixed_usage']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['mixed_usage']['where_text'] = "OBJART ='41006'"
            elif exclusion_dict['mixed_usage']['source'] == "osm":
                exclusion_dict['mixed_usage']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['mixed_usage']['where_text'] = "fclass='farmyard'"

        # TODO Check if this key exist in the key list
        if exclusion_dict.get("5Houses") is not None:
            if exclusion_dict['5Houses'].get('source') is not None:
                if exclusion_dict['5Houses']['source'] in ("inner_areas",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['5Houses']['source']} is not supported for 5Houses!")
                    exclusion_dict['5Houses']['source'] = "inner_areas"
            else:
                # by default use inner_areas
                exclusion_dict['5Houses']['source'] = "inner_areas"
            if exclusion_dict['5Houses']['source'] == "inner_areas":
                exclusion_dict['5Houses']['path'] = os.path.join(_datasources_path, "inner_areas",
                                                                 "gis_osm_landuse_a_free_1_with_geomAttr.shp")
                # TODO how does this number come?
                exclusion_dict['5Houses']['where_text'] = "fclass='residential' AND area>9022"

        if exclusion_dict.get("10Houses") is not None:
            # TODO remain this?
            if exclusion_dict['10Houses'].get('source') is not None:
                if exclusion_dict['10Houses']['source'] in ("inner_areas",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['10Houses']['source']} is not supported for 10Houses!")
                    exclusion_dict['10Houses']['source'] = "inner_areas"
            else:
                # by default use inner_areas
                exclusion_dict['10Houses']['source'] = "inner_areas"
            if exclusion_dict['10Houses']['source'] == "inner_areas":
                exclusion_dict['10Houses']['path'] = os.path.join(_datasources_path, "inner_areas",
                                                                  "gis_osm_landuse_a_free_1_with_geomAttr.shp")
                # TODO how does this number come?
                exclusion_dict['10Houses']['where_text'] = "fclass='residential' AND area>18044"

        # TODO is there a problem, when we consider industry and commercial together?
        #  if not, change the key list correspondingly
        if exclusion_dict.get("industrial_commercial") is not None:
            if exclusion_dict['industrial_commercial'].get('source') is not None:
                if exclusion_dict['industrial_commercial']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['industrial_commercial']['source']} is not supported for "
                                  f"industrial_commercial!")
                    exclusion_dict['industrial_commercial']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['industrial_commercial']['source'] = "basis-dlm"
            if exclusion_dict['industrial_commercial']['source'] == "basis-dlm":
                exclusion_dict['industrial_commercial']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['industrial_commercial']['where_text'] = "OBJART ='41002'"
            elif exclusion_dict['industrial_commercial']['source'] == "dlm250":
                exclusion_dict['industrial_commercial']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['industrial_commercial']['where_text'] = "OBJART ='41002'"
            elif exclusion_dict['industrial_commercial']['source'] == "osm":
                exclusion_dict['industrial_commercial']['path'] = os.path.join(_osm_path,
                                                                               "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['industrial_commercial']['where_text'] = "fclass = 'commercial' OR fclass= 'industrial'"
            elif exclusion_dict['industrial_commercial']['source'] == "clc":
                exclusion_dict['industrial_commercial']['path'] = _clcVectorPath
                exclusion_dict['industrial_commercial']['where_text'] = "Code_18='121'"

        if exclusion_dict.get("dvor"):
            if exclusion_dict['dvor'].get('source') is not None:
                if exclusion_dict['dvor']['source'] in ("osm_overpass",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['dvor']['source']} is not supported for dvor!")
                    exclusion_dict['dvor']['source'] = "osm_overpass"
            else:
                # by default use osm_overpass
                exclusion_dict['dvor']['source'] = "osm_overpass"
            if exclusion_dict['dvor']['source'] == "osm_overpass":
                exclusion_dict['dvor']['path'] = os.path.join(_osm_overpass_path, "D-VOR_OSM.shp")
                exclusion_dict['dvor']['where_text'] = None

        if exclusion_dict.get("vor"):
            if exclusion_dict['vor'].get('source') is not None:
                if exclusion_dict['vor']['source'] in ("osm_overpass",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['vor']['source']} is not supported for vor!")
                    exclusion_dict['vor']['source'] = "osm_overpass"
            else:
                # by default use osm_overpass
                exclusion_dict['vor']['source'] = "osm_overpass"
            if exclusion_dict['vor']['source'] == "osm_overpass":
                exclusion_dict['vor']['path'] = os.path.join(_osm_overpass_path, "VOR_OSM.shp")
                exclusion_dict['vor']['where_text'] = None

        if exclusion_dict.get("seismic_station"):
            if exclusion_dict['seismic_station'].get('source') is not None:
                if exclusion_dict['seismic_station']['source'] in ("bgr",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['seismic_station']['source']} is not supported for seismic station!")
                    exclusion_dict['seismic_station']['source'] = "bgr"
            else:
                # by default use osm_overpass
                exclusion_dict['seismic_station']['source'] = "bgr"
            if exclusion_dict['seismic_station']['source'] == "bgr":
                exclusion_dict['seismic_station']['path'] = os.path.join(_datasources_path, "Seismologie",
                                                                         "seismic_station_de.shp")
                exclusion_dict['seismic_station']['where_text'] = None

        if exclusion_dict.get("military") is not None:
            if exclusion_dict['military'].get('source') is not None:
                if exclusion_dict['military']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['military']['source']} is not supported for military!")
                    exclusion_dict['military']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['military']['source'] = "basis-dlm"
            if exclusion_dict['military']['source'] == "basis-dlm":
                exclusion_dict['military']['path'] = os.path.join(_basis_dlm_path, "geb03_f.shp")
                exclusion_dict['military']['where_text'] = "ADF = '4720'"
            elif exclusion_dict['military']['source'] == "dlm250":
                exclusion_dict['military']['path'] = os.path.join(_dlm250_path, "geb03_fshp")
                exclusion_dict['military']['where_text'] = "ADF = '4720'"
            elif exclusion_dict['military']['source'] == "osm":
                exclusion_dict['military']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['military']['where_text'] = "fclass='military'"

        if exclusion_dict.get("cemetery") is not None:
            if exclusion_dict['cemetery'].get('source') is not None:
                if exclusion_dict['cemetery']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['cemetery']['source']} is not supported for cemetery!")
                    exclusion_dict['cemetery']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['cemetery']['source'] = "basis-dlm"
            if exclusion_dict['cemetery']['source'] == "basis-dlm":
                exclusion_dict['cemetery']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['cemetery']['where_text'] = "OBJART = '41009'"
            elif exclusion_dict['cemetery']['source'] == "dlm250":
                exclusion_dict['cemetery']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['cemetery']['where_text'] = "OBJART = '41009'"
            elif exclusion_dict['cemetery']['source'] == "osm":
                exclusion_dict['cemetery']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['cemetery']['where_text'] = "fclass = 'cemetery'"

        if exclusion_dict.get("recreational") is not None:
            # TODO exclude gis_osm_landuse_a_free_1 or gis_osm_pois_a_free_1?
            if exclusion_dict['recreational'].get('source') is not None:
                if exclusion_dict['recreational']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['recreational']['source']} is not supported for recreational!")
                    exclusion_dict['recreational']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['recreational']['source'] = "basis-dlm"
            if exclusion_dict['recreational']['source'] == "basis-dlm":
                exclusion_dict['recreational']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['recreational']['where_text'] = "OBJART = '41008'"
            elif exclusion_dict['recreational']['source'] == "dlm250":
                exclusion_dict['recreational']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['recreational']['where_text'] = "OBJART = '41008'"
            elif exclusion_dict['recreational']['source'] == "osm":
                exclusion_dict['recreational']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['recreational']['where_text'] = "fclass = 'park' OR fclass = 'recreation_ground'"

        if exclusion_dict.get("camping") is not None:
            # TODO camping is already included in the recreational area of basis-dlm.
            if exclusion_dict['camping'].get('source') is not None:
                if exclusion_dict['camping']['source'] in ("basis-dlm", "osm", "dlm250"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['camping']['source']} is not supported for camping!")
                    exclusion_dict['camping']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['camping']['source'] = "basis-dlm"
            if exclusion_dict['camping']['source'] == "basis-dlm":
                exclusion_dict['camping']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['camping']['where_text'] = "FKT = '4330'"
            elif exclusion_dict['camping']['source'] == "dlm250":
                exclusion_dict['camping']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['camping']['where_text'] = "FKT = '4330'"
            elif exclusion_dict['camping']['source'] == "osm":
                exclusion_dict['camping']['path'] = os.path.join(_osm_path, "gis_osm_pois_a_free_1.shp")
                exclusion_dict['camping']['where_text'] = "fclass='camp_site'"

        if exclusion_dict.get("historical") is not None:
            if exclusion_dict['historical'].get('source') is not None:
                if exclusion_dict['historical']['source'] in ("basis-dlm", "osm", "dlm250", "hu"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['historical']['source']} is not supported for historical!")
                    exclusion_dict['historical']['source'] = "osm"
            else:
                # by default use osm
                exclusion_dict['historical']['source'] = "osm"
            if exclusion_dict['historical']['source'] == "basis-dlm":
                exclusion_dict['historical']['path'] = os.path.join(_basis_dlm_path, "sie03_f.shp")
                exclusion_dict['historical']['where_text'] = "OBJART_TXT = " \
                                                             "'AX_HistorischesBauwerkOderHistorischeEinrichtung'"
            elif exclusion_dict['historical']['source'] == "dlm250":
                exclusion_dict['historical']['path'] = os.path.join(_dlm250_path, "sie03_f.shp")
                exclusion_dict['historical']['where_text'] = "OBJART_TXT = " \
                                                             "'AX_HistorischesBauwerkOderHistorischeEinrichtung'"
            elif exclusion_dict['historical']['source'] == "osm":
                exclusion_dict['historical']['path'] = os.path.join(_osm_path, "gis_osm_pois_a_free_1.shp")
                exclusion_dict['historical']['where_text'] = "fclass IN ('archaeological','monument','memorial'," \
                                                             "'castle') "
            elif exclusion_dict['historical']['source'] == "hu":
                exclusion_dict['historical']['path'] = _hu_path
                exclusion_dict['historical']['where_text'] = "GFK IN ('31001_3031', '31001_3038')"

        if exclusion_dict.get("mineral_extraction") is not None:
            if exclusion_dict['mineral_extraction'].get('source') is not None:
                if exclusion_dict['mineral_extraction']['source'] in ("basis-dlm", "osm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['mineral_extraction']['source']} is not supported for mineral_extraction!")
                    exclusion_dict['mineral_extraction']['source'] = "basis-dlm"
            else:
                # by default use basis-dlm
                exclusion_dict['mineral_extraction']['source'] = "basis-dlm"
            if exclusion_dict['mineral_extraction']['source'] == "basis-dlm":
                exclusion_dict['mineral_extraction']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['mineral_extraction']['where_text'] = "OBJART ='41005' OR OBJART ='41004'"
            elif exclusion_dict['mineral_extraction']['source'] == "dlm250":
                exclusion_dict['mineral_extraction']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['mineral_extraction']['where_text'] = "OBJART ='41005' OR OBJART ='41004'"
            elif exclusion_dict['mineral_extraction']['source'] == "osm":
                exclusion_dict['mineral_extraction']['path'] = os.path.join(_osm_path, "gis_osm_landuse_a_free_1.shp")
                exclusion_dict['mineral_extraction']['where_text'] = "fclass = 'quarry'"
            elif exclusion_dict['mineral_extraction']['source'] == "clc":
                exclusion_dict['mineral_extraction']['path'] = _clcVectorPath
                exclusion_dict['mineral_extraction']['where_text'] = "Code_18='131'"

        if exclusion_dict.get("dump_sites") is not None:
            # TODO remove this key?
            # TODO for osm is it recycling?
            if exclusion_dict['dump_sites'].get('source') is not None:
                if exclusion_dict['dump_sites']['source'] in ("basis-dlm", "dlm250", "clc"):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['dump_sites']['source']} is not supported for dump_sites!")
                    exclusion_dict['dump_sites']['source'] = "clc"
            else:
                # by default use clc
                exclusion_dict['dump_sites']['source'] = "clc"
            if exclusion_dict['dump_sites']['source'] == "basis-dlm":
                exclusion_dict['dump_sites']['path'] = os.path.join(_basis_dlm_path, "sie02_f.shp")
                exclusion_dict['dump_sites']['where_text'] = "FKT = '2600'"
            elif exclusion_dict['dump_sites']['source'] == "dlm250":
                exclusion_dict['dump_sites']['path'] = os.path.join(_dlm250_path, "sie02_f.shp")
                exclusion_dict['dump_sites']['where_text'] = "FKT = '2600'"
            elif exclusion_dict['dump_sites']['source'] == "clc":
                exclusion_dict['dump_sites']['path'] = _clcVectorPath
                exclusion_dict['dump_sites']['where_text'] = "Code_18='132'"

        if exclusion_dict.get("construction") is not None:
            # TODO remove this key? is there any study consider this key?
            if exclusion_dict['construction'].get('source') is not None:
                if exclusion_dict['construction']['source'] in ("clc",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['construction']['source']} is not supported for construction!")
                    exclusion_dict['construction']['source'] = "clc"
            else:
                # by default use clc
                exclusion_dict['construction']['source'] = "clc"
            if exclusion_dict['construction']['source'] == "clc":
                exclusion_dict['construction']['path'] = _clcVectorPath
                exclusion_dict['construction']['where_text'] = "Code_18='133'"

        if exclusion_dict.get("wind_100m") is not None:
            if exclusion_dict['wind_100m'].get('source') is not None:
                if exclusion_dict['wind_100m']['source'] in ("gwa",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['wind_100m']['source']} is not supported for wind_100m!")
                    exclusion_dict['wind_100m']['source'] = "gwa"
            else:
                # by default use gwa
                exclusion_dict['wind_100m']['source'] = "gwa"
            if exclusion_dict['wind_100m']['source'] == "gwa":
                exclusion_dict['wind_100m']['path'] = os.path.join(_datasources_path, "gwa", "DEU_wind-speed_100m.tif")
                exclusion_dict['wind_100m']['value'] = tuple(exclusion_dict["wind_100m"]["value"])

        if exclusion_dict.get("wind_100m_era") is not None:
            if exclusion_dict['wind_100m_era'].get('source') is not None:
                if exclusion_dict['wind_100m_era']['source'] in ("gwa",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['wind_100m_era']['source']} is not supported for wind_100m_era!")
                    exclusion_dict['wind_100m_era']['source'] = "gwa"
            else:
                # by default use gwa
                exclusion_dict['wind_100m_era']['source'] = "gwa"
            if exclusion_dict['wind_100m_era']['source'] == "gwa":
                exclusion_dict['wind_100m_era']['path'] = os.path.join(_datasources_path, "gwa",
                                                                       "ERA5_wind_speed_100m_mean.tiff")
                exclusion_dict['wind_100m_era']['value'] = tuple(exclusion_dict["wind_100m_era"]["value"])

        if exclusion_dict.get("wind_100m_power") is not None:
            if exclusion_dict['wind_100m_power'].get('source') is not None:
                if exclusion_dict['wind_100m_power']['source'] in ("gwa",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['wind_100m_power']['source']} is not supported for wind_100m_power!")
                    exclusion_dict['wind_100m_power']['source'] = "gwa"
            else:
                # by default use gwa
                exclusion_dict['wind_100m_power']['source'] = "gwa"
            if exclusion_dict['wind_100m_power']['source'] == "gwa":
                exclusion_dict['wind_100m_power']['path'] = os.path.join(_datasources_path, "gwa",
                                                                         "DEU_power-density_100m.tif")
                exclusion_dict['wind_100m_power']['value'] = tuple(exclusion_dict["wind_100m_power"]["value"])

        # Elevation and slope from copernicus elevation model.
        # Check what is excluded to only read file once
        if exclusion_dict.get("elevation") is not None:
            if exclusion_dict['elevation'].get('source') is not None:
                if exclusion_dict['elevation']['source'] in ("copernicus",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['elevation']['source']} is not supported for elevation!")
                    exclusion_dict['elevation']['source'] = "copernicus"
            else:
                # by default use copernicus
                exclusion_dict['elevation']['source'] = "copernicus"
            if exclusion_dict['elevation']['source'] == "copernicus":
                exclusion_dict['elevation']['path'] = os.path.join(_datasources_path, "dem", "copernicus",
                                                                   "copernicus_merged.tif")
                exclusion_dict['elevation']['value'] = tuple(exclusion_dict["elevation"]["value"])

        if exclusion_dict.get("slope") is not None:
            # TODO the relationship between raster value and degree are hard coded
            if exclusion_dict['slope'].get('source') is not None:
                if exclusion_dict['slope']['source'] in ("copernicus",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['slope']['source']} is not supported for slope!")
                    exclusion_dict['slope']['source'] = "copernicus"
            else:
                # by default use copernicus
                exclusion_dict['slope']['source'] = "copernicus"
            if exclusion_dict['slope']['source'] == "copernicus":
                exclusion_dict['slope']['path'] = os.path.join(_datasources_path, "dem", "copernicus",
                                                               "copernicus_slop_merged.tif")
                exclusion_dict['slope']['value'] = tuple(exclusion_dict["slope"]["value"])
        # TODO compare birds with ramsar from WDPA data
        if exclusion_dict.get("birds") is not None:
            # 300m to Bird protection areas and "Naturschutzgebieten"
            if exclusion_dict['birds'].get('source') is not None:
                if exclusion_dict['birds']['source'] in ("wdpa",):
                    pass
                else:
                    warnings.warn(f"{exclusion_dict['birds']['source']} is not supported for birds!")
                    exclusion_dict['birds']['source'] = "wdpa"
            else:
                # by default use wdpa
                exclusion_dict['birds']['source'] = "wdpa"
            if exclusion_dict['birds']['source'] == "wdpa":
                exclusion_dict['birds']['path'] = _wdpa_de_path
                exclusion_dict['birds']['where_text'] = "DESIG_ENG = 'Special Protection Area (Birds Directive)'"

        if exclusion_dict.get("nature_protection") is not None:
            if exclusion_dict['nature_protection'].get('source') is not None:
                if exclusion_dict['nature_protection']['source'] in ("wdpa",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['nature_protection']['source']} is not supported for nature_protection!")
                    exclusion_dict['nature_protection']['source'] = "wdpa"
            else:
                # by default use wdpa
                exclusion_dict['nature_protection']['source'] = "wdpa"
            if exclusion_dict['nature_protection']['source'] == "wdpa":
                exclusion_dict['nature_protection']['path'] = _wdpa_de_path
                exclusion_dict['nature_protection']['where_text'] = "Desig='Naturschutzgebiet'"

        if exclusion_dict.get("nationalpark") is not None:
            if exclusion_dict['nationalpark'].get('source') is not None:
                if exclusion_dict['nationalpark']['source'] in ("wdpa",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['nationalpark']['source']} is not supported for nationalpark!")
                    exclusion_dict['nationalpark']['source'] = "wdpa"
            else:
                # by default use wdpa
                exclusion_dict['nationalpark']['source'] = "wdpa"
            if exclusion_dict['nationalpark']['source'] == "wdpa":
                exclusion_dict['nationalpark']['path'] = _wdpa_de_path
                exclusion_dict['nationalpark']['where_text'] = "Desig='Nationalpark'"

        if exclusion_dict.get("habitats") is not None:
            if exclusion_dict['habitats'].get('source') is not None:
                if exclusion_dict['habitats']['source'] in ("wdpa",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['habitats']['source']} is not supported for habitats!")
                    exclusion_dict['habitats']['source'] = "wdpa"
            else:
                # by default use wdpa
                exclusion_dict['habitats']['source'] = "wdpa"
            if exclusion_dict['habitats']['source'] == "wdpa":
                exclusion_dict['habitats']['path'] = _wdpa_de_path
                exclusion_dict['habitats']['where_text'] = "Desig='Site of Community Importance (Habitats Directive)'"

        if exclusion_dict.get("landscape") is not None:
            if exclusion_dict['landscape'].get('source') is not None:
                if exclusion_dict['landscape']['source'] in ("wdpa",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['landscape']['source']} is not supported for landscape!")
                    exclusion_dict['landscape']['source'] = "wdpa"
            else:
                # by default use wdpa
                exclusion_dict['landscape']['source'] = "wdpa"
            if exclusion_dict['landscape']['source'] == "wdpa":
                exclusion_dict['landscape']['path'] = _wdpa_de_path
                exclusion_dict['landscape']['where_text'] = "DESIG_ENG = 'Landscape Protection Area'"

        if exclusion_dict.get("biospheres_core") is not None:
            if exclusion_dict['biospheres_core'].get('source') is not None:
                if exclusion_dict['biospheres_core']['source'] in ("bfn",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['biospheres_core']['source']} is not supported for biospheres core zone!")
                    exclusion_dict['biospheres_core']['source'] = "bfn"
            else:
                # by default use bfn
                exclusion_dict['biospheres_core']['source'] = "bfn"
            if exclusion_dict['biospheres_core']['source'] == "bfn":
                exclusion_dict['biospheres_core']['path'] = os.path.join(_bfn_path, "Bio_Zonierung2021_3035.shp")
                exclusion_dict['biospheres_core']['where_text'] = "ZONIERUNG = 'Kernzone'"

        if exclusion_dict.get("biospheres_develop") is not None:
            if exclusion_dict['biospheres_develop'].get('source') is not None:
                if exclusion_dict['biospheres_develop']['source'] in ("bfn",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['biospheres_develop']['source']} is not supported for biospheres development zone!")
                    exclusion_dict['biospheres_develop']['source'] = "bfn"
            else:
                # by default use bfn
                exclusion_dict['biospheres_develop']['source'] = "bfn"
            if exclusion_dict['biospheres_develop']['source'] == "bfn":
                exclusion_dict['biospheres_develop']['path'] = os.path.join(_bfn_path, "Bio_Zonierung2021_3035.shp")
                exclusion_dict['biospheres_develop']['where_text'] = "ZONIERUNG = 'Entwicklungszone'"

        if exclusion_dict.get("biospheres_maintain") is not None:
            if exclusion_dict['biospheres_maintain'].get('source') is not None:
                if exclusion_dict['biospheres_maintain']['source'] in ("bfn",):
                    pass
                else:
                    warnings.warn(
                        f"{exclusion_dict['biospheres_maintain']['source']} is not supported for biospheres maintain zone!")
                    exclusion_dict['biospheres_maintain']['source'] = "bfn"
            else:
                # by default use bfn
                exclusion_dict['biospheres_maintain']['source'] = "bfn"
            if exclusion_dict['biospheres_maintain']['source'] == "bfn":
                exclusion_dict['biospheres_maintain']['path'] = os.path.join(_bfn_path, "Bio_Zonierung2021_3035.shp")
                exclusion_dict['biospheres_maintain']['where_text'] = "ZONIERUNG = 'Pflegezone'"

        return exclusion_dict

    def _exclude_features(self, feature_dict: Dict, ec: Type[gl.ExclusionCalculator],
                          intermediate: Text = None, plot_sankey: bool = False) -> float:
        """
        Call the exclusion method from glaes. Select correct data type to exclude.
        ----------
        feature_dict : dict
            Dictionary containing the information for the exclusion
        ec : gl.ExclusionCalculator
            ExclusionCalculator from glaes
        intermediate: path, optional
            Path to an intermediate result raster file for this set of function arguments.
            All acceptable to excludeRasterType/excludeVectorType.
        """
        _data_type = {
            "basis-dlm": "vector", "dlm250": "vector", "osm": "vector", "clc": "vector", "osm_overpass": "vector",
            "copernicus": "raster", "gwa": "raster", "wdpa": "vector", "hu": "vector", "inner_areas": "vector",
            "bgr": "vector", "vg250": "vector", "bfn": "vector"
        }
        # if not use intermediate, set the path to None
        if not self.parent.use_intermediate:
            intermediate = None
        if _data_type[feature_dict["source"]] == "vector":
            ec.excludeVectorType(feature_dict["path"], where=feature_dict["where_text"], buffer=feature_dict["buffer"],
                                 intermediate=intermediate)
        elif _data_type[feature_dict["source"]] == "raster":
            ec.excludeRasterType(feature_dict["path"], value=feature_dict["value"], buffer=feature_dict["buffer"],
                                 intermediate=intermediate)
        if intermediate is not None and plot_sankey:
            matrix = gk.raster.extractMatrix(intermediate)
            matrix = np.where(matrix == 0, 1, 0)
            sum_pixels = matrix.sum()
            excluded_area = sum_pixels * self.parent.regionMask._pixelRes ** 2
            return excluded_area
        else:
            return 0.0

    def _exclude_regional_features(self, feature_dict: Dict, ec: Type[gl.ExclusionCalculator],
                                   intermediate: Text = None, plot_sankey: bool = False) -> float:
        """
        Read information from the auxiliary dictionary and use them to exclude regional features.
        ----------
        feature_dict : dict
            Dictionary containing the information for the exclusion
        ec : gl.ExclusionCalculator
            ExclusionCalculator from glaes
        intermediate: path, optional
            Path to an intermediate result raster file for this set of function arguments.
            All acceptable to excludeRasterType/excludeVectorType.
        """
        # TODO how can we set intermediate for special features? Or do we still need intermediate here?
        if not self.parent.use_intermediate:
            intermediate = None
        if feature_dict["type"] == "vector":
            ec.excludeVectorType(feature_dict["source_path"], where=feature_dict["where_text"],
                                 buffer=feature_dict["buffer"], intermediate=intermediate)
        elif feature_dict["type"] == "raster":
            ec.excludeRasterType(feature_dict["source_path"], value=feature_dict["value"],
                                 buffer=feature_dict["buffer"], intermediate=intermediate)
        if intermediate is not None and plot_sankey:
            matrix = gk.raster.extractMatrix(intermediate)
            matrix = np.where(matrix == 0, 1, 0)
            sum_pixels = matrix.sum()
            excluded_area = sum_pixels * self.parent.regionMask._pixelRes ** 2
            return excluded_area
        else:
            return 0.0

    def save_report(self, output: Text) -> None:
        """
        Save the exclusion criteria and results as a report in json format.
        ----------
        output: path
            The path to save the result
        """
        with open(output, "w", encoding='utf-8') as f:
            json.dump(self.report_dict, f, indent=4, ensure_ascii=False)
        pass

    def save_items_to_vector(self, output):
        """
        Save the exclusion criteria and results as a report in json format.
        ----------
        output: path
            The path to save the distributed items
        """
        predicted_items = self.predicted_items
        geoms = []
        for i in range(len(predicted_items)):
            itemCoord = (predicted_items.loc[i]["lon"], predicted_items.loc[i]["lat"])
            geoms.append(gk.geom.point(itemCoord))
        predicted_items["geom"] = geoms
        gk.vector.createVector(predicted_items, output=output)

        # gk.vector.createVector(geoms, output=output)

    def _merge_to_germany(self, path_states=None, technology=None):
        """Merge the potential area in federal states to germany, i.e. merge several small rasters to one large.

        Parameters
        ----------
        path_states: list
            1. Specific the external source of available area in each state. Should only be given when the raster files
            cannot be found in the case directory.
            2. By default, None. Automatically search in the case directory. Make sure Wind_potential_area.tif exist for
            each state.
        """
        assert self.parent.level == "country"
        case_path = self.parent.case_path
        path_merged = os.path.join(self.result_path, f"{technology}_potential_area.tif")
        if isinstance(path_states, list):
            pass
        elif path_states is None:
            files = os.listdir(case_path)
            path_states = []
            for sub_dir in files:
                if os.path.isdir(os.path.join(case_path, sub_dir)) and \
                        (f"{technology}_" in sub_dir) and \
                        os.path.isfile(os.path.join(case_path, sub_dir, f"{technology}_potential_area.tif")) and \
                        sub_dir not in self.result_path:
                    print(f"Found in {sub_dir}", flush=True)
                    path_states.append(os.path.join(case_path, sub_dir, f"{technology}_potential_area.tif"))
        ec_de = self.parent.new_ec()
        for path in path_states:
            # Check if no areas are available
            print(f"Start merging {path}")
            matrix = gk.raster.extractMatrix(path)
            matrix = np.where(matrix == 0, 1, 0)
            if matrix.sum() == 0:
                continue
            else:
                ec_de.excludeRasterType(path, value=0, buffer=0)
        ec_de.save(path_merged)

    @abstractmethod
    def estimate_potential(self):
        """
        Abstract method which has to be implemented by subclasses.
        Estimate potential of technology.
        """
        raise NotImplementedError

    @abstractmethod
    def get_existing_plants(self):
        raise NotImplementedError

    def load_exclusionDict(self, name):
        """Load the .json-exclusion-dict file.

        Parameters
        ----------
        name : str
            name of the json-file

        Returns
        -------
        dict
            exclusion dict
        """
        path = os.path.join(os.path.dirname(__file__), "data", "config",
                            name + ".json")
        assert os.path.isfile(path), (
                "Couldn't find config file. Check data/config " +
                "for existing files. Or create one.")
        with open(path, encoding="utf-8") as f:
            exclusion_dict = json.load(f)
        exclusion_dict["state"] = self.parent.state
        return exclusion_dict

    @abstractmethod
    def sim(self):
        raise NotImplementedError

    @abstractmethod
    def sim_existing(self):
        raise NotImplementedError

    @staticmethod
    def sim_pv(
            placements, module="LG Electronics LG370Q1C-A5", poa_bound=0,
            merge=True, year=2014, workflow="ERA5"):
        """Simulate pv items.

        Parameters TODO
        ----------
        placements : pd.DataFrame
            df with lat, lon, capacity, tilt, azi
        module : str, optional
            name of the pv module, by default "LG Electronics LG370Q1C-A5"
        poa_bound : int/float, optional
            lower bound for the plane of array irradiance, by default 0
        merge : bool, optional
            merges the time-series if true, by default True

        Returns
        -------
        pd.DataFrame
            gen: generation time-series

        pd.DataFrame
            placements: input placements with added total generation and poa
        """
        if workflow not in ["ERA5", "SARAH"]:
            raise ValueError("Only Era5 and SARAH workflow implemented")
        # TODO adjust module for ofpv
        # TODO implement different workflows
        # TODO implement different weather years
        era5_path = r"/storage/internal/data/gears/weather/ERA5/processed/4/8/5/2014"
        sarah_path = r"/storage/internal/data/gears/weather/SARAH/processed/4/8/5/2014"
        elevation_path = "/storage/internal/data/s-risch/dem/copernicus_merged.tif"
        ghi_path = "/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_GHI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/GHI.tif"
        dni_path = "/storage/internal/data/gears/geography/irradiance/global_solar_atlas_v2.5/World_DNI_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/DNI.tif"
        if isinstance(placements, pd.DataFrame):
            pass
        elif isinstance(placements, str):
            placements = pd.read_csv(placements, index_col=0)
        else:
            raise TypeError(
                "Type of input not supported. Use Dataframe or path to csv.")
        if "geom" in placements.columns:
            placements = placements.drop("geom", axis=1)
        location = None
        if "location" in placements.columns:
            location = placements["location"]
            placements = placements.drop("location", axis=1)
        if not "ID" in placements.columns:
            placements.loc[:, "ID"] = placements.index.values
        # Check if placements are empty. Important for regional workflow.
        # I.e.: Small muns sometimes miss groups.
        if len(placements) == 0:
            warnings.warn("Length of placement is 0, returning 0 time series.")
            gen = pd.DataFrame(columns=["gen"], index=range(0, 8760))
            gen["gen"] = 0
            placements["generation"] = 0
            placements["poa"] = 0
            return gen, placements

        print("Simulating {} modules".format(len(placements)), flush=True)
        # Filter 0 capacities in existing rtpvs
        placements_0cap = placements[placements["capacity"] == 0]
        placements = placements.drop(placements_0cap.index)
        if workflow == "ERA5":
            xds = rk.solar.openfield_pv_era5(
                placements=placements, era5_path=era5_path,
                global_solar_atlas_ghi_path=ghi_path,
                global_solar_atlas_dni_path=dni_path,
                elev=300, module=module)
        elif workflow == "SARAH":
            xds = rk.solar.openfield_pv_sarah_unvalidated(
                placements=placements, sarah_path=sarah_path,
                era5_path=era5_path,
                elev=300, module=module)
        placements.loc[:, "poa"] = None
        placements.loc[:, "generation"] = None
        if not merge:
            if workflow == "ERA5":
                gen = pd.DataFrame(
                    index=range(0, 8760),
                    columns=placements.ID)
            elif workflow == "SARAH":
                gen = pd.DataFrame(
                    index=range(0, 8760 * 2),
                    columns=placements.ID)
        for i in xds.location:
            _gen = pd.Series(xds.total_system_generation[:, i]).fillna(0)
            _poa = pd.Series(xds.poa_global[:, i]).fillna(0)
            if len(_gen) == 8760 * 2:
                _index = pd.date_range(
                    dt.datetime(2014, 1, 1),
                    dt.datetime(2014, 12, 31, 23, 30),
                    freq="30T")
                _gen.index = _index
                _poa.index = _index
                _gen = _gen.resample("H").mean()
                _poa = _poa.resample("H").mean()
            # check for items which have lower poa then the lower bound
            placements.loc[placements.ID ==
                           xds.ID[i].values, "poa"] = _poa.sum()
            placements.loc[placements.ID ==
                           xds.ID[i].values, "generation"] = _gen.sum()
            placements_0cap["poa"] = 0
            placements_0cap["generation"] = 0

            if _poa.sum() >= poa_bound:
                if merge:
                    if i == 0:
                        gen = xds.total_system_generation[:, i]
                        cf = xds.capacity_factor[:, i]
                        cap = xds.capacity[i]
                    else:
                        gen += xds.total_system_generation[:, i]
                        cf += xds.capacity_factor[:, i]
                        cap += xds.capacity[i]
                else:
                    gen[str(xds.ID[i].values)] = pd.Series(
                        xds.total_system_generation[:, i]).fillna(0)
        placements = pd.concat([placements, placements_0cap])
        # Eliminate items with smaller than irradiance smaller than poa_bound
        placements = placements[placements.poa >= poa_bound]
        if merge:
            _temp = pd.Series(gen).fillna(0)
            gen = pd.DataFrame(columns=["gen"])
            gen["gen"] = pd.Series(_temp).fillna(0)
        if len(gen) == 8760 * 2:
            gen.index = pd.date_range(
                dt.datetime(2014, 1, 1),
                dt.datetime(2014, 12, 31, 23, 30),
                freq="30T")
            gen = gen.resample("H").mean()
            gen.index = range(0, 8760)
        if location is not None:
            placements["location"] = location
        return gen, placements

    @staticmethod
    def sim_wind(placements, grouping_method="spagat", n_groups=7,
                 turbine=None, year=2014, **grouping_kwargs):
        """Simulate wind items.

        Parameters TODO
        ----------
        placements : pd.DataFrame
            df with lat, lon, capacity, diameter, hub height
        grouping_method: str, optional
            options are:
                - spagat for fine-spagat grouping
                - bins for mates bin grouping
                - None for no grouping TODO 
            by default 'spagat'
        n_groups : int, optional
            number of clusters, by default 7

        Returns
        -------
        pd.DataFrame
            ts: generation time-series

        pd.DataFrame
            placements: input placements with added columns "group" and "FLH"
        """
        assert grouping_method in ["spagat", "bins"], \
            f"{grouping_method} not implemented. Please choose" \
            "grouping_method from spagat and bins"
        era5_path = f"/storage/internal/data/gears/weather/ERA5/processed/4/8/5/{year}"
        gwa_100m_path = r"/storage/internal/data/gears/geography/wind/global_wind_atlas/GWA_3.0/gwa3_250_wind-speed_100m.tif"
        cci_path = r"/storage/internal/data/gears/geography/landcover/esa_cci_v2.1.1/C3S-LC-L4-LCCS-Map-300m-P1Y-2018-v2.1.1.tif"
        if "geom" in placements.columns:
            if any([type(i) == str for i in placements.geom.values]):
                warnings.warn(
                    "Deleting geom column because it seems to be string " +
                    "values and not a geom object", UserWarning)
                placements = placements.drop("geom", axis=1)
        location = None
        if "location" in placements.columns \
                and "lat" in placements.columns \
                and "lon" in placements.columns:
            location = placements["location"]
            warnings.warn(
                "Deleting location column because it may conflict " +
                "with lat/lon columns", UserWarning)
            placements = placements.drop("location", axis=1)
        if turbine is not None:
            # turbine_data = rk.wind.TurbineLibrary().loc[turbine]
            # power_curve = turbine_data["PowerCurve"]
            placements["powerCurve"] = [turbine] * len(placements)
        if placements.hub_height.isna().any():
            placements = placements[placements.hub_height.notna()]
            warnings.warn(
                f"Dropped WTs {placements[placements.hub_height.isna()]}" +
                " because no hub height was provided", UserWarning)
        if not "ID" in placements.columns:
            placements.loc[:, "ID"] = placements.index.values
        # Simulation with rk
        if len(placements) > 0:
            xds = rk.wind.onshore_wind_era5(
                placements=placements,
                era5_path=era5_path,
                gwa_100m_path=gwa_100m_path,
                esa_cci_path=cci_path,
            )
            # First get FLH for grouping
            placements.loc[:, "FLH"] = None
            for i in xds.location:
                placements.loc[placements.ID == xds.ID[i].values,
                               "FLH"] = pd.Series(xds.capacity_factor[:, i]).sum()
            ts = pd.DataFrame(index=list(range(0, 8760)))
            if grouping_method == "spagat":
                start = time.time()
                xds["region"] = (("location"), len(xds.capacity.values)*["Region"])
                _n_groups = n_groups if n_groups < len(placements) else len(placements)
                grouped_ds = represent_RE_technology(
                            non_gridded_RE_ds=xds,
                            location_dim_name='location',
                            capacity_var_name='capacity',
                            capfac_var_name='capacity_factor',
                            region_var_name='region',
                            n_timeSeries_perRegion=_n_groups)
                item_groups = {}
                for i, group in enumerate(grouped_ds.TS_ids):
                    ts[f"Wind_items_00{i}"] = pd.Series(grouped_ds.sel(TS_ids=group).capacity_factor.values.reshape(8760,))
                    item_groups[f"Wind_items_00{i}"] = grouped_ds.attrs[f"Region.{group.values}"]
                for group, items in item_groups.items():
                    for item in items:
                        placements.loc[item, "group"] = group
                print("Took ", time.time()-start)

            # Grouping with MATES
            elif grouping_method == "bins":
                groups = IG_utils.placementGrouping(
                    name="Wind_items", placements=placements, numberOfGroups=n_groups,
                    groupingIndicator="FLH", **grouping_kwargs)
                gr_arrays = {}
                for group in groups.keys():
                    gr_arrays[group] = None
                    placements.loc[groups[group].index, "group"] = group
                # Now save groups generation series
                for i in xds.location:
                    val = xds.capacity_factor[:, i] * xds.capacity[i]
                    if gr_arrays[placements.loc
                    [placements.ID == xds.ID[i].values, "group"].values
                    [0]] is None:
                        gr_arrays[
                            placements.loc[placements.ID ==
                                        xds.ID[i].values, "group"].values[0]] = val
                    else:
                        gr_arrays[
                            placements.loc[placements.ID ==
                                        xds.ID[i].values, "group"].values[0]] += val
                for group in gr_arrays.keys():
                    ts[group] = pd.Series(gr_arrays[group])
                if location is not None:
                    placements["location"] = location
        else:
            placements["group"] = None
            placements["FLH"] = None
            placements["location"] = None
            ts = pd.DataFrame(index=list(range(0, 8760)), columns=[f"Wind_items_00{i}" for i in range(0, 7)], data=0)
        return ts, placements
