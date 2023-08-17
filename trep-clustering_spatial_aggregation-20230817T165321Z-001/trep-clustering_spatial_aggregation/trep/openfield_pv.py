from trep.technology import Technology
from trep import utils
import reskit as rk
import pandas as pd
import geokit as gk
import os
import numpy as np
from trep.utils import rename_columns
import osgeo
from warnings import warn
from abc import ABC
from sqlalchemy import create_engine


class BaseOpenfieldPV(ABC):
    def __init__(self, parent):
        super(BaseOpenfieldPV, self).__init__(parent)
        self.existing_items = None

    def assign_optimal_orientation(self, modules):
        """Assign the optimal orientation to the openfield-pv-modules.

        Parameters
        ----------
        modules : pd.DataFrame
            [description]

        Returns
        -------
        pd.DataFrame
            df of modules with added optimal orientation
        """
        modules["elev"] = 300
        modules["tilt"] = modules.apply(
            lambda x: rk.solar.location_to_tilt((x["lon"], x["lat"]))[0],
            axis=1)
        modules["azimuth"] = 180

        return modules

    def distribute_items(self, minArea=500, efficiency=0.2214):
        """Distribute pv items on the eligible area

        Parameters
        ----------
        minArea : int, optional
            minimum required area for distribution of items, by default 500
        efficiency : float, optional
            efficiency of the pv-modules, by default 0.2214
        """
        modules = pd.DataFrame(columns=["capacity", "geom", "lon", "lat",
                                        "tilt", "azimuth", "elev"])
        # TODO what is the meaning to invoke ec.distributeItems() ?
        self.ec.distributeItems(separation=1000)
        self.ec.distributeAreas(minArea=minArea)
        modules["geom"] = self.ec._areas
        modules["center"] = modules.apply(
            lambda x: [(gk.geom.extractVerticies(x["geom"].Centroid())[0][0],
                        gk.geom.extractVerticies(x["geom"].Centroid())[0][1])],
            axis=1)
        modules["lon"] = modules.apply(lambda x: gk.srs.xyTransform(
            x["center"], fromSRS=self.ec.region.srs, toSRS=4326)[0][0], axis=1)
        modules["lat"] = modules.apply(lambda x: gk.srs.xyTransform(
            x["center"], fromSRS=self.ec.region.srs, toSRS=4326)[0][1], axis=1)
        modules = self.assign_optimal_orientation(modules)

        modules["area"] = modules.apply(lambda x: x["geom"].Area(), axis=1)
        # For basis scenario: From ISE Recent Facts about PV (1.4 ha/MWp)
        modules["capacity"] = modules["area"].multiply(0.5*efficiency)
        # For future scenarios determine row spacing and use high efficiency
        # modules["capacity"] = modules["area"].multiply(0.2214*0.5)
        self.predicted_items = modules

    def exclude_existing(self, **args):
        """Exclude existing plants and area around them as eligible land.

        Excludes the area of existing open field pv units.
        """
        if self.parent.OpenfieldPV.existing_items is None:
            ec = self.parent.OpenfieldPV.get_existing_plants(self.ec, **args)
        # else:
        #     if "existing Openfield" not in self.ec._additional_points:
        #         if len(self.parent.OpenfieldPV.existing_items) > 0:
        #             if self.ec._additional_points is None:
        #                 self.ec._additional_points = {}
        #             self.ec._additional_points.update({"existing Openfield": {}})
        #             self.ec._additional_points["existing Openfield"]["points"] = np.array(
        #                 [i
        #                  for i in self.parent.OpenfieldPV.existing_items
        #                  ["location"].values])
        if len(self.parent.OpenfieldPV.existing_items) > 0:
            # distance from PV-loc: sqrt(14m^2/kWp * pv_cap) -->
            # distance equally in both directions
            # (14m2 from Frauenhofer recent facts)
            self.parent.OpenfieldPV.existing_items["scale"] = self.parent.OpenfieldPV.existing_items.apply(
                lambda x: np.array([np.sqrt(x["capacity"] * 14) / 2, np.sqrt(x["capacity"] * 14) / 2]), axis=1)
            self.parent.OpenfieldPV.existing_items["direction"] = 0
            self.ec.excludePoints(self.parent.OpenfieldPV.existing_items,
                                 geometry_shape="rectangle", save_to_ec="existing Openfield")


class OpenfieldPV(BaseOpenfieldPV, Technology):
    def __init__(self,
                 parent):
        super(OpenfieldPV, self).__init__(parent)
        self.existing_items = None
        # Check the path in database
        # TODO a seperate directory only for municipality?
        self.result_path = os.path.join(self.parent.case_path, f"OpenfieldPV_{self.parent._id}")
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

    def get_existing_plants(self, ec, mastr=True):
        """Get existing wind turbines & open-field pv plants.

        Parameters
        ----------
        ec : gl.ExclusionCalculator
            exclusion calculator
        mastr : bool, optional
            Use of Marktstammdatenregister, by default False

        Returns
        -------
        gl.ExclusionCalculator
            exclusion calculator

        Raises
        ------
        ValueError
            if federal state is not implemented
        """
        self.parent.check_existing_db("OpenfieldPV")
        if self.existing_items is None:
            self.existing_items = pd.DataFrame()
            # TODO Add existing wka from more Federal states? like bw
            if self.parent._state == "05" and not mastr:
                if self.parent.level == "nuts3":
                    # TODO take parts of gkz
                    existing_str = "t_kreis='{}'".format(self.parent.region)
                elif self.parent.level == "MUN":
                    existing_str = "t_gkz='{}'".format(self.parent._ags)
                # Data from shape file from open data nrw.
                if self.parent.level == "state":
                    _exItem = gk.vector.extractFeatures(
                        source=os.path.join(
                            self.parent.datasource_path,
                            "state_" + self.parent._state,
                            "OpenEE-FreiflaechenPV_EPSG25832_Shape",
                            "Freiflaechen_PV.shp"))
                else:
                    _exItem = gk.vector.extractFeatures(
                        source=os.path.join(
                            self.parent.datasource_path,
                            "state_" + self.parent._state,
                            "OpenEE-FreiflaechenPV_EPSG25832_Shape",
                            "Freiflaechen_PV.shp"),
                        where="{}".format(existing_str))
                exItem = []
                for row in _exItem.iterrows():
                    exItem.append(
                        [row[1]["geom"].GetX(), row[1]["geom"].GetY()]
                    )
                exItem = np.array(exItem)
                exItem = \
                    gk.srs.xyTransform(exItem, fromSRS=25832,
                                       toSRS=self.parent.regionMask.srs)
                _exItem["location"] = _exItem.apply(
                    lambda x: np.array(
                        gk.srs.xyTransform(
                            np.array(
                                [[x["geom"].GetX(),
                                  x["geom"].GetY()]]),
                            fromSRS=25832, toSRS=self.parent.regionMask.srs)),
                    axis=1)
                _exItem["location"] = _exItem.apply(lambda x: np.array(
                    [x["location"][0][0], x["location"][0][1]]), axis=1)
                _coor = gk.srs.xyTransform(
                    exItem, fromSRS=self.parent.regionMask.srs, toSRS=4326)
                self.existing_items["capacity"] = _exItem["leistung"]
                self.existing_items["lat"] = [i[1] for i in _coor]
                self.existing_items["lon"] = [i[0] for i in _coor]
                self.existing_items["location"] = _exItem["location"]
                self.existing_items["geom"] = _exItem["geom"]
            else:
                if not mastr:
                    warn(
                        "For federal state {} ".format(self.parent.state) +
                        "only existing plants from mastr are available ",
                        UserWarning)
                # Get data from mastr
                _regions = []
                # First get all items in federal state.
                # TODO: Map all states, maybe in utils
                if self.parent._state == "01":
                    existing_str = "ENH_Bundesland='Schleswig-Holstein'"
                elif self.parent._state == "03":
                    existing_str = "ENH_Bundesland='Niedersachsen'"
                elif self.parent._state == "05":
                    existing_str = "ENH_Bundesland='Nordrhein-Westfalen'"
                elif self.parent._state == "07":
                    existing_str = "ENH_Bundesland='Rheinland-Pfalz'"
                elif self.parent._state == "16":
                    existing_str = "ENH_Bundesland='Thüringen'"
                else:
                    raise ValueError("State not implemented yet")
                # Get openfield PVs from db
                query = "SELECT ENH_MastrID, ENH_Nettonennleistung, " + \
                    "ENH_Plz, ENH_InbetriebnahmeDatum, ENH_Breitengrad, " + \
                    "ENH_Laengengrad FROM processed WHERE " + \
                    "ENH_EinheitenTyp='Solareinheit' and " + \
                    "ENH_Lage='Freifläche' and " +\
                    "ENH_Betriebsstatus='In Betrieb'" + \
                    " and ({})".format(existing_str)
                engine = create_engine(
                    "sqlite:///" + os.path.join(
                        self.parent.datasource_path,
                        "mastr", "mastr.db") + "/?charset=utf8mb4")
                raw_pvs = pd.read_sql(sql=query, con=engine)
                # Geometry processing.
                if len(raw_pvs) > 0:
                    raw_pvs["point"] = raw_pvs.apply(lambda x: gk.geom.point(
                        [x["ENH_Laengengrad"], x["ENH_Breitengrad"]]), axis=1)
                    _rm_lat_lon = gk.geom.transform(
                        self.parent.regionMask.geometry, toSRS=4326)
                    raw_pvs["Within"] = raw_pvs.apply(
                        lambda x: osgeo.ogr.Geometry.Within(
                            x["point"],
                            _rm_lat_lon),
                        axis=1)
                    self.existing_items = raw_pvs[raw_pvs["Within"]]
                    if len(self.existing_items) > 0:
                        self.existing_items["location"] = \
                            self.existing_items.apply(
                            lambda x: np.array(gk.srs.xyTransform(np.array(
                                [[x["ENH_Laengengrad"],
                                x["ENH_Breitengrad"]]]),
                                fromSRS=4326,
                                toSRS=self.parent.regionMask.srs)),
                            axis=1)
                        self.existing_items["geom"] = self.existing_items.apply(lambda x: gk.geom.point(
                            [x["ENH_Laengengrad"], x["ENH_Breitengrad"]]), axis=1)
                        self.existing_items["location"] = \
                            self.existing_items.apply(lambda x: np.array(
                                [x["location"][0][0], x["location"][0][1]]),
                            axis=1)
                        self.existing_items = rename_columns(self.existing_items)
                        self.existing_items = self.assign_optimal_orientation(
                            self.existing_items)
                else:
                    self.existing_items = pd.DataFrame()
            # Add points to Exclusion Calculator
            # if len(self.existing_items) > 0:
            #     if ec._additional_points is None:
            #         ec._additional_points = {}
            #     ec._additional_points.update({"existing Openfield": {}})
            #     ec._additional_points["existing Openfield"]["points"] = np.array(
            #         [i for i in self.existing_items["location"].values])

        return ec

    def run_exclusion(self, exclusion_dict=None, update=True, **args):
        """Run exclusion to estimate potential eligible area for openfield-pv.

        Types can be used to start from total area excluded and include
        certain type of areas for the analysis.

        Parameters
        ----------
        exclusion_dict : dict, optional
            Dictionary containing the information for the exclusion.
            A default dict for ofpv is present and can be updated in certain
            points. By default None
        update : bool, optional
            update the default exclusion_dict if true, by default True

        Raises
        ------
        TypeError
            if not the dict or str passed for exclusion_dict
        """
        if type(exclusion_dict) not in [dict, str] and not exclusion_dict is None:
            raise TypeError(
                "Use string as input for pre-defined exclusion, " +
                "or dict for your own exclusion.")
        if self.parent.state is None:
            _exclusion_dict = self.load_exclusionDict("openfield_PV_basis")
        else:
            _exclusion_dict = self.load_exclusionDict(f"openfield_PV_{self.parent.state}")
        if isinstance(exclusion_dict, dict) and update:
            _exclusion_dict.update(exclusion_dict)
        elif isinstance(exclusion_dict, dict) and not update:
            _exclusion_dict = exclusion_dict
        elif isinstance(exclusion_dict, str):
            _exclusion_dict = self.load_exclusionDict(exclusion_dict)

        if _exclusion_dict.get("existing") is not None:
            self.exclude_existing()
        report_dict = self._run_exclusion(exclusion_dict=_exclusion_dict, **args)
        if self.ec.percentAvailable > 0:
            self.ec.pruneIsolatedAreas(minSize=500)
        report_dict["Total_Area"] = int(self.parent.regionMask.mask.sum() * self.parent.regionMask.pixelRes ** 2)
        report_dict["Eligible_Area"] = self.ec.areaAvailable
        report_dict["Eligible_Percentage"] = self.ec.percentAvailable
        return report_dict

    def estimate_potential(self, exclusion_dict=None, predict=True, ignore_exist=False, use_lffa=True,
                           **args):
        """Estimate the potential for open-field pv.

        Parameters
        ----------
        exclusion_dict : dict, optional
            Dictionary containing the information for the exclusion.
            By default None
            TODO: exclusion_dict kein input für run_exclusion (?)
        predict : bool, optional
            weather pv items shall be predicted on eligible land,
            by default True
        ignore_exist : bool, optional
            weather to ignore the existing results
        use_lffa : bool, optional
            weather to use less-favoured farming areas to indicate eligible area
        """
        self.predicted_items = self.parent.check_db(self)
        if self.predicted_items is None:
            if os.path.isfile(os.path.join(self.result_path, "OpenfieldPV_potential_area.tif")) and not ignore_exist:
                self.report_dict = self.load_eligible_area()
            else:
                if use_lffa:
                    self.load_less_favoured_farming_areas()
                self.report_dict = self.run_exclusion(exclusion_dict=exclusion_dict, **args)
            if predict:
                self.distribute_items()
                self.report_dict["Items_Number"] = self.ec._itemCoords.shape[0]
                self.report_dict["Capacity"] = self.predicted_items['capacity'].sum()

    def load_less_favoured_farming_areas(self, path_lffa: str = None, **kwargs):
        """Load the less-favoured farming areas to exclusion calculator.

        Parameters
        ----------
        path_lffa : path, optional
            path of data to indicate less favoured farming areas
        """
        filenames = {
           "sh": "BNG_1990_10_3_f.shp",
           "hh": None,
           "ni": "BNG.shp",
           "hb": None,
           "nw": "V_OD_BENA.shp",
           "he": "DIFF_GEB_NATURA.shp",
           "rp": "BNG_ELER.shp",
           "bw": "BNG_ohne_Teilgebiet.shp",
           "by": "BNG.shp",
           "sl": "BNG.shp",
           "be": None,
           "bb": "BNG.shp",
           "mv": "BNG_1990_10_03_f.shp",
           "sn": "BNG.shp",
           "st": "BNG_f.shp",
           "th": "BNG.shp"
        }
        # Load the less-favoured farming areas
        print(f"Loading less-favoured farming areas in {self.parent.state}")
        if filenames[self.parent.state] is None:
            print(f"All areas of {self.parent.state} are less-favoured farming areas", flush=True)
        else:
            self.ec = self.parent.new_ec(initialValue=False)
            if path_lffa is None:
                path_lffa = os.path.join(self.parent.datasource_path, "benachteiligten_Gebiet", self.parent.state,
                                         filenames[self.parent.state])
            self.ec.excludeVectorType(
                path_lffa,
                where=None,
                buffer=0,
                mode="include")

        # Extract the areas to matrix
        matrix_lffa = self.ec._availability

        # Extract the farmland inside the less-favoured farming areas
        print(f"Loading the farmland in {self.parent.state}")
        path_farmland = os.path.join(self.parent.dlm_basis_path, "veg01_f.shp")
        self.ec.excludeVectorType(
            path_farmland,
            where="OBJART_TXT='AX_Landwirtschaft'",
            buffer=0,
        )
        matrix_farmland = self.ec._availability
        matrix_lffa_farmland = matrix_farmland + matrix_lffa
        self.ec._availability = np.where(matrix_lffa_farmland == 100, 100, 0)

    def load_eligible_area(self, overwrite_old: bool = False):
        """
        Load the existing result of Land Eligible Analysis to ExclusionCalculator.

        Note:
            1. If the current used pixel resolution is different with the LE result, this function
                cannot perform as you wish.
            2. Double check the resolution before set the overwrite_old to True.
        Parameters
        ----------
        overwrite_old: bool, default False. Only set to True, when you want to save the loading time
        for a very large RegionMask.
        """
        path_LE = os.path.join(self.result_path, "OpenfieldPV_potential_area.tif")
        assert os.path.isfile(path_LE), f"Can't find the LE result {path_LE}"
        if self.ec._hasEqualContext(path_LE):
            print("The LE result can be directly accepted", flush=True)
            initial_LE = gk.raster.extractMatrix(path_LE)
        else:
            print("The LE result has to be warped to the mask region", flush=True)
            path_LE_warp = os.path.join(self.result_path, "OpenfieldPV_potential_area_warp.tif")
            self.ec.region.warp(path_LE, output=path_LE_warp)
            print("Warp finished", flush=True)
            initial_LE = gk.raster.extractMatrix(path_LE_warp)
        initial_LE = np.where(initial_LE == 100, 100, 0)
        self.ec._availability = initial_LE
        print("LE result is loaded", flush=True)
        if overwrite_old:
            self.ec.save(path_LE)
        report_dict = {"Exclusion": "loaded from existing results",
                       "Total_Area": int(self.parent.regionMask.mask.sum() * self.parent.regionMask.pixelRes ** 2),
                       "Eligible_Area": self.ec.areaAvailable,
                       "Eligible_Percentage": self.ec.percentAvailable}
        return report_dict

    def merge_to_germany(self, path_states: list = None):
        """Merge the potential area in federal states to germany, i.e. merge several small rasters to one large.

        Parameters
        ----------
        path_states: list
            see technology._merge_to_germany()
        """
        technology = "OpenfieldPV"
        self._merge_to_germany(path_states, technology)

    def sim(self, **kwargs):
        """Simulate the predicted pv items."""
        # TODO: Maybe move to baseopenfield also for roads
        self.ts_predicted_items = self.parent.check_db(self, "ts")
        if self.ts_predicted_items is None:
            if self.predicted_items is None:
                self.estimate_potential()
                # TODO: Either remove or pass exclusion. Ensure consistency
                # with other sim methods (wind, rtpv, etc.)
            self.ts_predicted_items, self.predicted_items = self.sim_pv(
                placements=self.predicted_items, poa_bound=814, **kwargs)
        else:
            print("TS already in db")

    def sim_existing(self, **kwargs):
        """Simulate existing pv-items - if not already in db."""
        self.parent.check_existing_db("OpenfieldPV")
        if self.ts_existing_items is None:
            if self.existing_items is None:
                self.get_existing_plants(ec=self.ec)
            if len(self.existing_items) > 0:
                self.ts_existing_items, self.existing_items = self.sim_pv(
                    placements=self.existing_items, poa_bound=0, **kwargs)
            else:
                print("No existing ofpvs --> not simulating")
        else:
            print("TS already in db")


class OpenfieldPVRoads(BaseOpenfieldPV, Technology):
    def __init__(self,
                 parent,
                 type="both"):
        super(OpenfieldPVRoads, self).__init__(parent=parent)
        assert type in ["Roads", "Railways",
                        "both"], "{} not implemented".format(type)
        self.type = type
        # Check the path in database
        # TODO The directory name has once been changed, is there a negative effect?
        self.result_path = os.path.join(self.parent.case_path, f"OpenfieldPVRoads_{self.parent._id}")
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

    def run_exclusion(self, exclusion_dict=None, update=True, **args):
        """Estimate openfield PV potential on sides of roads and railways.

        Start from total area excluded and include areas next to roads 
        and railways.

        Parameters
        ----------
        exclusion_dict : dict, optional
            Dictionary containing the information for the exclusion.
            A default dict for ofpv is present and can be updated in certain
            points. By default None
        update : bool, optional
            weather to reload exlusion_dict if already read,
            by default True

        Raises
        ------
        TypeError
            if not the dict or str passed for exclusion_dict
        """
        if type(exclusion_dict) not in [dict, str] and not exclusion_dict is None:
            raise TypeError(
                "Use string as input for pre-defined exclusion, " +
                "or dict for your own exclusion.")
        if self.parent.state is None:
            _exclusion_dict = self.load_exclusionDict("openfield_roads_basis")
        else:
            _exclusion_dict = self.load_exclusionDict(f"openfield_roads_{self.parent.state}")
        if isinstance(exclusion_dict, dict) and update:
            _exclusion_dict.update(exclusion_dict)
        elif isinstance(exclusion_dict, dict) and not update:
            _exclusion_dict = exclusion_dict
        elif isinstance(exclusion_dict, str):
            _exclusion_dict = self.load_exclusionDict(exclusion_dict)
        self.ec = self.parent.new_ec(initialValue=False)

        # If new data source should be used, use trep.utils.line_to_area() to generate polygon features from line
        # features
        if _exclusion_dict.get("available_side_stripes") is not None:
            available_side_stripes = _exclusion_dict["available_side_stripes"]
        else:
            available_side_stripes = 200  # 200m (EEG2021), the 15m corridor for animals should be considered in buffer
        # TODO do we need to provide alternative with other data source? Otherwise, if the user can not access to
        #  basis-dlm, the ofpv_roads analyses can not be made
        if self.type == "Roads":
            self.ec.excludeVectorType(
                os.path.join(self.parent.dlm_basis_path, "ofpv", "Autobahn_a.shp"),
                where="(ZUS != '2100' OR ZUS is null) AND  HDU_X = 0",
                buffer=available_side_stripes,
                mode="include")
        elif self.type == "Railways":
            self.ec.excludeVectorType(
                os.path.join(self.parent.dlm_basis_path, "ofpv", "Bahn_Strecke_a.shp"),
                # where="fclass='motorway' or fclass='primary' OR fclass='motorway_link'",
                where="(ZUS != '2100' OR ZUS is null) AND HDU_X = 0 AND BKT='1100'",
                buffer=available_side_stripes,
                mode="include")
        elif self.type == "both":
            self.ec.excludeVectorType(
                os.path.join(self.parent.dlm_basis_path, "ofpv", "Autobahn_a.shp"),
                where="(ZUS != '2100' OR ZUS is null) AND  HDU_X = 0",
                buffer=available_side_stripes,
                mode="include")
            self.ec.excludeVectorType(
                os.path.join(self.parent.dlm_basis_path, "ofpv", "Bahn_Strecke_a.shp"),
                where="(ZUS != '2100' OR ZUS is null) AND HDU_X = 0 AND BKT='1100'",
                buffer=available_side_stripes,
                mode="include")
        if self.ec.percentAvailable == 0:
            print(f"No area alongside {self.type} available")
            return {"Info": "There is no potential areas on sides of roads and railways"}
        else:
            report_dict = self._run_exclusion(_exclusion_dict, **args)
            if self.ec.percentAvailable > 0:
                self.ec.pruneIsolatedAreas(minSize=500)
            # TODO: only if in dict
            if _exclusion_dict.get("existing") is not None:
                self.exclude_existing()
            report_dict["Total_Area"] = int(self.parent.regionMask.mask.sum() * self.parent.regionMask.pixelRes ** 2)
            report_dict["Eligible_Area"] = self.ec.areaAvailable
            report_dict["Eligible_Percentage"] = self.ec.percentAvailable
            return report_dict

    def estimate_potential(self, exclusion_dict=None, efficiency=0.2214, predict=True, ignore_exist=False, **args):
        """Estimate the potential of the predicted pv-items.

        Parameters
        ----------
        exclusion_dict : dict, optional
            Dictionary containing the information for the exclusion,
            by default None
        efficiency : float, optional
            efficiency of the pv modules, by default 0.2214
        predict : bool, optional
            if items shall be perdicted for eligible areas,
            by default True
        ignore_exist : bool, optional
            weather to ignore the existing results
        """
        self.predicted_items = self.parent.check_db(self)
        if self.predicted_items is None:
            if os.path.isfile(os.path.join(self.result_path, "OpenfieldPVRoads_potential_area.tif")) and not ignore_exist:
                self.report_dict = self.load_eligible_area()
            else:
                self.report_dict = self.run_exclusion(exclusion_dict=exclusion_dict, **args)
            if predict:
                if self.ec.percentAvailable == 0:
                    self.predicted_items = pd.DataFrame(
                        columns=["capacity", "geom", "lon", "lat", "tilt",
                                 "azimuth", "elev"])
                else:
                    self.distribute_items(efficiency=efficiency)
                    self.report_dict["Items_Number"] = self.ec._itemCoords.shape[0]
                    self.report_dict["Capacity"] = self.predicted_items['capacity'].sum()

    def load_eligible_area(self, overwrite_old: bool = False):
        """
        Load the existing result of Land Eligible Analysis to ExclusionCalculator.

        Note:
            1. If the current used pixel resolution is different with the LE result, this function
                cannot perform as you wish.
            2. Double check the resolution before set the overwrite_old to True.
        Parameters
        ----------
        overwrite_old: bool, default False. Only set to True, when you want to save the loading time
        for a very large RegionMask.
        """
        path_LE = os.path.join(self.result_path, "OpenfieldPVRoads_potential_area.tif")
        assert os.path.isfile(path_LE), f"Can't find the LE result {path_LE}"
        if self.ec._hasEqualContext(path_LE):
            print("The LE result can be directly accepted", flush=True)
            initial_LE = gk.raster.extractMatrix(path_LE)
        else:
            print("The LE result has to be warped to the mask region", flush=True)
            path_LE_warp = os.path.join(self.result_path, "OpenfieldPVRoads_potential_area_warp.tif")
            self.ec.region.warp(path_LE, output=path_LE_warp)
            print("Warp finished", flush=True)
            initial_LE = gk.raster.extractMatrix(path_LE_warp)
        initial_LE = np.where(initial_LE == 100, 100, 0)
        self.ec._availability = initial_LE
        print("LE result is loaded", flush=True)
        if overwrite_old:
            self.ec.save(path_LE)
        report_dict = {"Exclusion": "loaded from existing results",
                       "Total_Area": int(self.parent.regionMask.mask.sum() * self.parent.regionMask.pixelRes ** 2),
                       "Eligible_Area": self.ec.areaAvailable,
                       "Eligible_Percentage": self.ec.percentAvailable}
        return report_dict

    def merge_to_germany(self, path_states: list = None):
        """Merge the potential area in federal states to germany, i.e. merge several small rasters to one large.

        Parameters
        ----------
        path_states: list
            see technology._merge_to_germany()
        """
        technology = "OpenfieldPVRoads"
        self._merge_to_germany(path_states, technology)

    def sim(self, **kwargs):
        """Simulate the predicted pv-items, if not available in db."""
        self.ts_predicted_items = self.parent.check_db(self, "ts")
        if self.ts_predicted_items is None:
            if self.predicted_items is None:
                self.estimate_potential()
            self.ts_predicted_items, self.predicted_items = self.sim_pv(
                placements=self.predicted_items, poa_bound=814, **kwargs)
        else:
            print("TS already in db")

    def get_existing_plants(self):
        raise NotImplementedError("No existing plants for OpenfieldPVRoads")

    def sim_existing(self):
        raise NotImplementedError("No existing plants for OpenfieldPVRoads")
        # Frauenhofer ISE: 1.4 ha/MW
        # elif type == "lignite":
        #     self.new_ec(level=self.level, initialValue=False)
        #     self.ec.excludeRasterType(
        #         self.clcRasterPath,
        #         value=131,
        #         buffer=0,
        #         mode="include"
        #     )
        #     _exclusion_dict = {
        #         "water": None,
        #         "motorway": None,
        #         "trunk": None,
        #         "primary": None,
        #         "medium_roads": None,
        #         "railways": None,
        #         "power_lines": None,
        #         "agriculture": None,
        #         "forests": None,
        #         "urban": None,
        #         "industrial": None,
        #         "airports": None,
        #         "mineral_extraction": None,
        #         "dump_sites": None,
        #         "construction": None,
        #         "birds": None,
        #         "protected": None,
        #         "nature_protection": None,
        #         "habitats": None,
        #         "nationalpark": None,
        #         "landscape": None,
        #         "region_edge": None,
        #         "existing": None,
        #         "wind_100m": None,
        #         "wind_100m_power": None,
        #         "state": "NRW",
        #         "soft_exclusion": None,
        #     }
        #     if exclusion_dict is None:
        #         print("No exclusion dict given --> using default for"+
        #             " former lignite area pv")
        #         pass
        #     else:
        #         _exclusion_dict.update(exclusion_dict)
        #     self.run_exclusion(_exclusion_dict)
        #     # Frauenhofer ISE: 1.4 ha/MW; study: 4.9% useable of water
        #     # TODO: Not all area is going to be lakes
        #     self.of_pv_potential = self.available_area["openfield PV roads"]/(1.4/100)*0.049
        # elif type == "bena":
        #     # Landwirtschaftlich benachteiligte Gebiete
        #     # (agricultural disadvantaged areas)
        #     self.new_ec(level=self.level, initialValue=False)
        #     self.ec.excludeVectorType(
        #         os.path.join(utils.get_data_path(),
        #                      "BENA_EPSG25832_Shape",
        #                      "V_OD_BENA.shp"),
        #         buffer=0,
        #         mode="include"
        #     )
        #     _exclusion_dict = {
        #         "water_still": 0,
        #         "water_river": 0,
        #         "water_stream": 0,
        #         "motorway": 0,
        #         "primary": 0,
        #         "medium_roads": 0,
        #         "railways": 0,
        #         "power_lines": 100,
        #         "agriculture": None,
        #         "forests": 0,
        #         "urban": 0,
        #         "industrial": 0,
        #         "airports": 0,
        #         "mineral_extraction": None,
        #         "dump_sites": 0,
        #         "construction": 0,
        #         "birds": 0,
        #         "protected": 0,
        #         "nature_protection": 0,
        #         "habitats": 0,
        #         "nationalpark": 0,
        #         "landscape": 0,
        #         "region_edge": 0,
        #         "existing": (0.1, 0.1, 0),
        #         "wind_100m": None,
        #         "wind_100m_power": None,
        #         "state": "NRW",
        #         "soft_exclusion": None,
        #     }
        #     if exclusion_dict is None:
        #         print("No exclusion dict given --> using default for pv"+
        #             "in agricultural disadvantaged areas")
        #         pass
        #     else:
        #         _exclusion_dict.update(exclusion_dict)
        #     # Frauenhofer ISE: 1.4 ha/MW
        #     self.run_exclusion(_exclusion_dict)
        # self.of_pv_potential = self.available_area["openfield PV roads"]/(1.4/100)
    # self.get_of_pv_items()
