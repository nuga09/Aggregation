from trep.technology import Technology
import geokit as gk
import os
from trep import utils
import trep
import pandas as pd
import copy
import numpy as np
from trep.utils import rename_columns, fill_rotor_diameter
import osgeo
from warnings import warn
from sqlalchemy import create_engine
import time
import xarray as xr
import reskit as rk


class Wind(Technology):
    def __init__(self,
                 parent,
                 capacity=3*1e3,
                 target_diameter=101,
                 hub_height=135,
                 distance=[8, 4],
                 wind_dir="from_era"):
        """Initialize wind technology.

        Parameters
        ----------
        parent : trep.TREP
            parent instance trep
        capacity : int, optional
            Rated capacity of the wind turbine, by default 3*1e3
        target_diameter : int, optional
            Target diameter of the wind turbine, by default 101
        hub_height : int, optional
            Hub heigt of the wind turbine, by default 135
        distance : list, optional
            Distances in main wind direction and transverse to it.
             Unit is rotor diameters.
             By default [8, 4] represents 8 diameters in front and behind
             the turbine in main direction, and 4 diameters in the
             transverse direction
        wind_dir : int/str, optional
            main wind direction (0: west, 90: south) or string 'from_era' to
            use the observation from Era5. By default "from_era"
        """

        super(Wind, self).__init__(parent=parent)
        self.capacity = capacity
        self.target_diameter = target_diameter
        self.hub_height = hub_height
        self.distance = distance
        self.wind_dir = wind_dir
        self.turbine = None
        # Check the path in database
        self.result_path = os.path.join(self.parent.case_path, f"Wind_{self.parent._id}")
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

    def get_existing_plants(self, ec, mastr=True, year_built=None):
        """Get existing wind turbines & open-field pv plants.

        Parameters
        ----------
        ec :
            glaes.ExclusionCalculator
        mastr : bool, optional
            Use of Marktstammdatenregister if other data is available,
            by default False
        year_built : int, optional
            Existing nuilt later (>=) than year_built are considered.
            TODO: this is only implemented for the opendata.nrw turbines.
            We have to implement it for mastr.
            By default None.
        """
        self.parent.check_existing_db("Wind")
        if self.existing_items is None:
            self.existing_items = pd.DataFrame()
            map_regions = {
                "Heinsberg": "Kreis Heinsberg",
                "Mönchengladbach": "Mönchengladbach, Stadt"
            }
            if self.parent._state == "05" and not mastr:
                # Data from shape file from open data nrw.
                if self.parent.level == "state":
                    _exItem = gk.vector.extractFeatures(
                        source=os.path.join(
                            self.parent.datasource_path,
                            "state_" + self.parent._state,
                            "OpenEE-Windenergie_EPSG25832_Shape",
                            "Windenergie.shp"))
                else:
                    if self.parent.level == "nuts3":
                        # TODO take parts of gkz
                        existing_str = "t_kreis='{}'".format(
                            self.parent.region)
                    elif self.parent.level == "MUN":
                        existing_str = "t_gkz='{}'".format(self.parent._ags)
                    _exItem = gk.vector.extractFeatures(
                        source=os.path.join(
                            self.parent.datasource_path,
                            "state_" + self.parent._state,
                            "OpenEE-Windenergie_EPSG25832_Shape",
                            "Windenergie.shp"),
                        where="{}".format(existing_str))
                _exItem.ibjahr = _exItem.ibjahr.astype(int)
                if year_built is not None:
                    _exItem = _exItem[_exItem.ibjahr >= year_built]
                _exItem["rotordurch"] = _exItem["rotordurch"].replace(
                    0, np.nan)
                _exItem = _exItem.rename(
                    columns={"rotordurch": "ENH_Rotordurchmesser",
                             "leistung": "ENH_Nettonennleistung",
                             "nabenhoehe": "ENH_Nabenhoehe"})
                if _exItem.ENH_Rotordurchmesser.isna().any():
                    _exItem = fill_rotor_diameter(_exItem, self.parent.datasource_path)
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
                self.existing_items["capacity"] = \
                    _exItem["ENH_Nettonennleistung"]
                self.existing_items["rotor_diam"] = \
                    _exItem["ENH_Rotordurchmesser"]
                self.existing_items["hub_height"] = \
                    _exItem["ENH_Nabenhoehe"].replace(0, np.nan)
                self.existing_items["lat"] = [i[1] for i in _coor]
                self.existing_items["lon"] = [i[0] for i in _coor]
                self.existing_items["location"] = _exItem["location"]
                self.existing_items["geom"] = _exItem["geom"]
                print("Existing capacity " +
                      f"{self.existing_items['capacity'].sum()/1e3} GW",
                      flush=True)
            elif self.parent._state == "16" and not mastr:
                # Data from open data in Thuringa
                if self.parent.level == "MUN":
                    existing_str = "GEMEINDE='{}'".format(self.parent.name)
                else:
                    existing_str = "KR_NAME='{}'".format(self.parent.name)
                existing_str += "AND STATUS='Bestand'"
                _exItem = gk.vector.extractFeatures(
                    source=os.path.join(
                        self.parent.datasource_path,
                        "state_" + self.parent._state, "WKA_shp",
                        "energie_primaerenergie_windrad_Z_Wert.shp"),
                    where="{}".format(existing_str))
                if len(_exItem) == 0:
                    pass
                else:
                    _exItem["rotordurch"] = _exItem["ROTORDURCH"].replace(
                        0, np.nan)
                    _exItem = _exItem.rename(
                        columns={"ROTORDURCH": "ENH_Rotordurchmesser",
                                 "LEISTUNG": "ENH_Nettonennleistung",
                                 "NABENHOEHE": "ENH_Nabenhoehe"})
                    if _exItem.ENH_Rotordurchmesser.isna().any():
                        _exItem = fill_rotor_diameter(_exItem, self.parent.datasource_path)
                    exItem = []
                    for row in _exItem.iterrows():
                        exItem.append(
                            [row[1]["geom"].GetX(), row[1]["geom"].GetY()]
                        )
                    exItem = np.array(exItem)
                    exItem = \
                        gk.srs.xyTransform(exItem, fromSRS=25832,
                                           toSRS=self.parent.regionMask.srs)
                    _exItem["location"] = \
                        _exItem.apply(lambda x: np.array(
                            gk.srs.xyTransform(
                                np.array(
                                    [[x["geom"].GetX(), x["geom"].GetY()]]),
                                fromSRS=25832,
                                toSRS=self.parent.regionMask.srs)),
                        axis=1)
                    _exItem["location"] = _exItem.apply(lambda x: np.array(
                        [x["location"][0][0], x["location"][0][1]]), axis=1)
                    _coor = gk.srs.xyTransform(
                        exItem, fromSRS=self.parent.regionMask.srs, toSRS=4326)
                    self.existing_items["capacity"] = \
                        _exItem["ENH_Nettonennleistung"]
                    self.existing_items["rotor_diam"] = \
                        _exItem["ENH_Rotordurchmesser"]
                    self.existing_items["hub_height"] = \
                        _exItem["ENH_Nabenhoehe"].replace(0, np.nan)
                    self.existing_items["lat"] = [i[1] for i in _coor]
                    self.existing_items["lon"] = [i[0] for i in _coor]
                    self.existing_items["location"] = _exItem["location"]
                    self.existing_items["geom"] = _exItem["geom"]
            elif self.parent._state == "01" and not mastr:
                # Data from Schleswig-Holstein
                if self.parent.level == "state":
                    _exItem = gk.vector.extractFeatures(
                        source=os.path.join(
                            os.path.abspath(os.path.dirname(__file__)),
                            "data", "SH_Wind_Standorte", "WEA Standortdaten",
                            "GIS-Standortdaten WEA", "WKA_SH_20200730.shp"))
                else:
                    existing_str = "Kreis='{}'".format(self.parent.region)
                    _exItem = \
                        gk.vector.extractFeatures(
                            source=os.path.join(
                                os.path.abspath(os.path.dirname(__file__)),
                                "data",
                                "SH_Wind_Standorte",
                                "WEA Standortdaten",
                                "GIS-Standortdaten WEA",
                                "WKA_SH_20200730.shp"),
                            where=existing_str)
                _exItem = _exItem.rename(
                    columns={"ROTORDURCH": "ENH_Rotordurchmesser",
                             "LEISTUNG": "ENH_Nettonennleistung",
                             "NABENHÖHE": "ENH_Nabenhoehe"})
                # Processing German entries (, --> .)
                for idx, row in _exItem.iterrows():
                    if isinstance(row["ENH_Rotordurchmesser"], str):
                        try:
                            _el = row["ENH_Rotordurchmesser"].replace(",", ".")
                            _exItem.loc[idx, "ENH_Rotordurchmesser"] = float(
                                _el)
                        except Exception as e:
                            _exItem.loc[idx, "ENH_Rotordurchmesser"] = np.nan
                            print(e)
                    if isinstance(row["ENH_Nabenhoehe"], str):
                        try:
                            _el = row["ENH_Nabenhoehe"].replace(",", ".")
                            _exItem.loc[idx, "ENH_Nabenhoehe"] = float(_el)
                        except Exception as e:
                            print("hub height not defined")
                            _exItem.loc[idx, "ENH_Nabenhoehe"] = np.nan
                            print(e)
                if _exItem.ENH_Rotordurchmesser.isna().any():
                    _exItem = fill_rotor_diameter(_exItem, self.parent.datasource_path)
                exItem = []
                for row in _exItem.iterrows():
                    exItem.append(
                        [row[1]["geom"].GetX(), row[1]["geom"].GetY()]
                    )
                exItem = np.array(exItem)
                exItem = gk.srs.xyTransform(
                    exItem, fromSRS=_exItem["geom"][0].GetSpatialReference(),
                    toSRS=self.parent.regionMask.srs)
                _coor = gk.srs.xyTransform(
                    exItem, fromSRS=self.parent.regionMask.srs, toSRS=4326)
                self.existing_items["capacity"] = \
                    _exItem["ENH_Nettonennleistung"]
                self.existing_items["rotor_diam"] = \
                    _exItem["ENH_Rotordurchmesser"]
                self.existing_items["hub_height"] = \
                    _exItem["ENH_Nabenhoehe"].replace(0, np.nan)
                self.existing_items["lat"] = [i[1] for i in _coor]
                self.existing_items["lon"] = [i[0] for i in _coor]
                self.existing_items["BST_NR"] = _exItem["BST_NR"]
                self.existing_items["geom"] = _exItem["geom"]
                self.existing_items["STATUS"] = _exItem["STATUS"]
            else:
                if not mastr:
                    warn(
                        "For federal state {} ".format(self.parent.state) +
                        "only existing plants from mastr are available ",
                        UserWarning)
                # Get data from mastr
                _regions = []
                # First get all items in federal state.
                # TODO: Map all states, maybe in utils (@Junsong)
                if self.parent._state == "01":
                    existing_str = "ENH_Bundesland='Schleswig-Holstein'"
                elif self.parent._state == "03":
                    existing_str = "ENH_Bundesland='Niedersachsen'"
                elif self.parent._state == "05":
                    existing_str = "ENH_Bundesland='Nordrhein-Westfalen'"
                elif self.parent._state == "07":
                    existing_str = "ENH_Bundesland='Rheinland-Pfalz'"
                elif self.parent._state == "12":
                    existing_str = "ENH_Bundesland='Brandenburg'"
                elif self.parent._state == "16":
                    existing_str = "ENH_Bundesland='Thüringen'"
                else:
                    raise ValueError("State not implemented yet")
                # Get raw data for wind turbines from db
                query = "SELECT ENH_MastrID, ENH_Nettonennleistung," + \
                    "ENH_Plz, ENH_InbetriebnahmeDatum, " + \
                    "ENH_Rotordurchmesser, ENH_Nabenhoehe, " + \
                    "ENH_Breitengrad, ENH_Laengengrad, ENH_Seelage " + \
                    "FROM processed WHERE " + \
                    "ENH_EinheitenTyp='Windeinheit' and " +\
                    "ENH_Betriebsstatus='In Betrieb' and ({})".format(
                        existing_str)
                # raw_wts = db_query(query)
                engine = create_engine(
                    "sqlite:///" + os.path.join(
                        self.parent.datasource_path,
                        "mastr", "mastr.db") + "/?charset=utf8mb4")
                raw_wts = pd.read_sql(sql=query, con=engine)
                # TODO: Filter in DB!
                # raw_wts = fill_rotor_diameter(raw_wts)
                # Geometry processing. First get point geometry of coordinate.
                raw_wts["geom"] = raw_wts.apply(lambda x: gk.geom.point(
                    [x["ENH_Laengengrad"], x["ENH_Breitengrad"]]), axis=1)
                # Then check if point is within regionMask.
                _rm_lat_lon = gk.geom.transform(
                    self.parent.regionMask.geometry, toSRS=4326)
                raw_wts["Within"] = raw_wts.apply(
                    lambda x: osgeo.ogr.Geometry.Within(
                        x["geom"],
                        _rm_lat_lon),
                    axis=1)
                raw_wts = raw_wts[raw_wts["Within"]]
                # Some filtering (No wts > 10MW, No diameter >500)
                _filtered_wts = \
                    raw_wts[raw_wts["ENH_Nettonennleistung"] < 10*1e3]
                _filtered_wts = \
                    _filtered_wts[_filtered_wts["ENH_Rotordurchmesser"] < 500]
                # Filter unreasonable geo coordinates
                self.existing_items = \
                    _filtered_wts[_filtered_wts["ENH_Breitengrad"] > 0]
                self.existing_items = \
                    self.existing_items[
                        self.existing_items["ENH_Laengengrad"] > 0]
                self.existing_items = \
                    self.existing_items[
                        self.existing_items["ENH_Breitengrad"] < 55]
                if len(self.existing_items) < len(_filtered_wts):
                    warn(
                        str(len(_filtered_wts) - len(self.existing_items)) +
                        " turbines don't have coordinates or are " +
                        "unreasonably large and are not included in analysis.",
                        UserWarning)
                print("Existing Turbines", len(self.existing_items), flush=True)
                if len(self.existing_items) > 0:
                    self.existing_items["location"] = \
                        self.existing_items.apply(
                            lambda x: np.array(gk.srs.xyTransform(np.array(
                                [[x["ENH_Laengengrad"],
                                  x["ENH_Breitengrad"]]]),
                                fromSRS=4326,
                                toSRS=self.parent.regionMask.srs)),
                        axis=1)
                    self.existing_items["location"] = \
                        self.existing_items.apply(
                            lambda x: np.array(
                                [x["location"][0][0], x["location"][0][1]]),
                            axis=1)
                    self.existing_items = rename_columns(self.existing_items)
                    # Drop not needed columns
                    for col in self.existing_items.columns:
                        if "ENH_" in col:
                            self.existing_items = self.existing_items.drop(
                                columns=col)
            # Add points to Exclusion Calculator
            if len(self.existing_items) > 0:
                ec._existingItemCoords = np.array(
                    [i for i in self.existing_items["location"].values])
        # else:
        #     if ec._existingItemCoords is None:
        #         if len(self.existing_items) > 0:
        #             ec._existingItemCoords = np.array(
        #                 [i for i in self.existing_items["location"].values])

    def exclude_existing(self, geometry_shape="ellipse", **args):
        """Exclude existing plants and area around them as eligible land.

        Excludes the area around existing wind turbines. Area can be specified
        by giving the distance in main wind direction and transverse.

        Parameters
        ----------
        how : str, optional
            Options are 'rectangle' or 'ellipse'., by default "ellipse"
        """
        plotted = True
        import time
        start = time.time()
        existing = None
        # if self.ec._existingItemCoords is None:
        print("Getting existing Items")
        self.get_existing_plants(self.ec, **args)
        if len(self.existing_items) > 0:
            if self.wind_dir == "from_era":
                wind_dir_raster = gk.raster.loadRaster(
                    os.path.join(self.parent.datasource_path, "era5",
                                 "ERA5_wind_direction_100m_mean.tiff"))
                # TODO: Rename to direction
                # Direction has to be in vector file or dataframe
                self.existing_items["direction"] = \
                    self.existing_items.apply(
                        lambda x: gk.raster.interpolateValues(
                            wind_dir_raster, x["geom"]),
                        axis=1)
            for idx, row in self.existing_items.iterrows():
                self.existing_items.loc[idx, "distance"] = row["rotor_diam"] if row["rotor_diam"] > self.target_diameter else self.target_diameter
            self.existing_items["scale"] = self.existing_items.apply(lambda x: np.array([self.distance[0]*x["distance"], self.distance[1]*x["distance"]]), axis=1)
            # TODO: Moved to glaes#
            self.ec.excludePoints(self.existing_items, geometry_shape, save_to_ec="Existing Turbines")
        print("Done excluding existing, took {} minutes".format(
            (time.time()-start)/60), flush=True)
        if self.parent.OpenfieldPV.existing_items is None:
            self.parent.OpenfieldPV.get_existing_plants(self.ec, **args)
        if len(self.parent.OpenfieldPV.existing_items) > 0:
                # distance from PV-loc: sqrt(14m^2/kWp * pv_cap) -->
                # distance equally in both directions
                # (14m2 from Frauenhofer recent facts)
            self.parent.OpenfieldPV.existing_items["scale"] = self.parent.OpenfieldPV.existing_items.apply(
                lambda x: np.array([np.sqrt(x["capacity"] * 14) / 2, np.sqrt(x["capacity"] * 14) / 2]), axis=1)
            self.parent.OpenfieldPV.existing_items["direction"] = 0
            self.ec.excludePoints(self.parent.OpenfieldPV.existing_items,
                                 geometry_shape="rectangle", save_to_ec="existing Openfield")

    def run_exclusion(self, exclusion_dict=None, update=True, **kwargs):
        """Run exclusion to estimate potential eligible area for wind.

        Parameters
        ----------
        exclusion_dict : dict, optional
            Dictionary containing the information for the exclusion,
            by default None
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
        # load default exclusion dictionary
        _exclusion_dict = self.load_exclusionDict(f"wind_{self.parent.state}")
        if isinstance(exclusion_dict, dict) and update:
            _exclusion_dict.update(exclusion_dict)
        elif isinstance(exclusion_dict, dict) and not update:
            _exclusion_dict = exclusion_dict
        elif isinstance(exclusion_dict, str):
            _exclusion_dict = self.load_exclusionDict(exclusion_dict)

        if _exclusion_dict.get("existing") is not None:
            _params = _exclusion_dict.get("existing")
            if _params.get("target_diameter") is not None:
                self.target_diameter = _params.get("target_diameter")
            if _params.get("distance") is not None:
                self.distance = _params.get("distance")
            if _params.get("wind_dir") is not None:
                self.wind_dir = _params.get("wind_dir")
            self.exclude_existing()
        report_dict = self._run_exclusion(_exclusion_dict, **kwargs)
        # Not exclude small area, when all areas are already excluded
        if self.ec.percentAvailable > 0:
            self.ec.pruneIsolatedAreas(minSize=10000)
            # write results in dictionary
        report_dict["Total_Area"] = int(self.parent.regionMask.mask.sum() * self.parent.regionMask.pixelRes ** 2)
        report_dict["Eligible_Area"] = self.ec.areaAvailable
        report_dict["Eligible_Percentage"] = self.ec.percentAvailable
        return report_dict

    def distribute_items(self, **args):
        """Distribute wind turbines on the eligible land."""
        print(self.distance)
        distance = tuple((i*self.target_diameter for i in self.distance))
        print("Distance between turbines ", distance, flush=True)
        if self.wind_dir == "from_era":
            _wind_dir = os.path.join(
                self.parent.datasource_path, "era5",
                "ERA5_wind_direction_100m_mean.tiff")
        else:
            assert isinstance(self.wind_dir, int)
            _wind_dir = self.wind_dir
        coordinates = self.ec.distributeItems(axialDirection=_wind_dir,
                                              separation=distance,
                                              outputSRS=4326,
                                              **args)
        df_items = pd.DataFrame(columns=["capacity", "hub_height",
                                         "rotor_diam", "lat", "lon"])
        # coordinates = self.ec.itemCoords
        df_items["lat"] = [i[1] for i in self.ec.itemCoords]
        df_items["lon"] = [i[0] for i in self.ec.itemCoords]
        df_items["capacity"] = self.capacity
        df_items["rotor_diam"] = self.target_diameter
        df_items["hub_height"] = self.hub_height
        self.predicted_items = df_items

    def restrict_area(self, share=0.01, tolerance=0.00001, step=0.00001):
        """Restrict usable area for wind to certain share.

        Parameters 
        ----------
        share : float, optional
            desired share of area, by default 0.01
        tolerance : float, optional
            acceptable tolerance in share, by default 0.00001
        step : float, optional
            step size, by default 0.00001
        """
        _exclusion_dict = self.load_exclusionDict("restrict_areas")
        # while self.ec.percentAvailable/100 > (share + 0.5):
        #     self.run_exclusion(_exclusion_dict)
        #     _exclusion_dict["wind_100m_era"] += 0.1
        temp0 = copy.copy(self.ec)
        temp1 = copy.copy(self.ec)
        temp2 = copy.copy(self.ec)
        temp3 = copy.copy(self.ec)
        temp4 = copy.copy(self.ec)
        temp5 = copy.copy(self.ec)
        temp6 = copy.copy(self.ec)
        print("0st step size")
        while temp6.percentAvailable / 100 > (share):
            _exclusion_dict["wind_100m"] += step * 1000000
            self._run_exclusion(ec=temp6, exclusion_dict=_exclusion_dict)
        _exclusion_dict["wind_100m"] -= step * 1000000
        while temp5.percentAvailable / 100 > (share):
            _exclusion_dict["wind_100m"] += step * 100000
            self._run_exclusion(ec=temp5, exclusion_dict=_exclusion_dict)
        _exclusion_dict["wind_100m"] -= step * 100000
        while temp0.percentAvailable / 100 > (share):
            _exclusion_dict["wind_100m"] += step * 10000
            self._run_exclusion(ec=temp0, exclusion_dict=_exclusion_dict)
        _exclusion_dict["wind_100m"] -= step * 10000
        print("First step size")
        while temp1.percentAvailable/100 > (share):
            _exclusion_dict["wind_100m"] += step*1000
            self._run_exclusion(ec=temp1, exclusion_dict=_exclusion_dict)
        _exclusion_dict["wind_100m"] -= step*1000
        print("Second step size")
        while temp2.percentAvailable/100 > (share):
            _exclusion_dict["wind_100m"] += step*100
            self._run_exclusion(ec=temp2, exclusion_dict=_exclusion_dict)
            # temp1.pruneIsolatedAreas(minSize=10000)
        _exclusion_dict["wind_100m"] -= step*100
        print("Third step size")
        while temp3.percentAvailable/100 > (share):
            _exclusion_dict["wind_100m"] += step*10
            self._run_exclusion(ec=temp3, exclusion_dict=_exclusion_dict)
        _exclusion_dict["wind_100m"] -= step*10
        while temp4.percentAvailable/100 > (share):
            _exclusion_dict["wind_100m"] += step
            self._run_exclusion(ec=temp4, exclusion_dict=_exclusion_dict)
        print("Final step size")
        temp = copy.copy(self.ec)
        self._run_exclusion(ec=temp, exclusion_dict=_exclusion_dict)
        if temp.percentAvailable > 0:
            temp.pruneIsolatedAreas(minSize=10000)
        count = 0
        while temp.percentAvailable / 100 < (share):
            count += 1
            temp = copy.copy(self.ec)
            _exclusion_dict["wind_100m"] -= step*100
            self._run_exclusion(ec=temp, exclusion_dict=_exclusion_dict)
            if temp.percentAvailable > 0:
                temp.pruneIsolatedAreas(minSize=10000)
            if count % 100 == 0:
                print(count)
                print(temp.percentAvailable)
        print(temp.percentAvailable)
        _exclusion_dict["wind_100m"] += step * 100
        temp = copy.copy(self.ec)
        self._run_exclusion(ec=temp, exclusion_dict=_exclusion_dict)
        if temp.percentAvailable > 0:
            temp.pruneIsolatedAreas(minSize=10000)
        count = 0
        while temp.percentAvailable / 100 < (share - 0.00004):
            count += 1
            temp = copy.copy(self.ec)
            _exclusion_dict["wind_100m"] -= step
            self._run_exclusion(ec=temp, exclusion_dict=_exclusion_dict)
            if temp.percentAvailable > 0:
                temp.pruneIsolatedAreas(minSize=10000)
            if count % 100 == 0:
                print(count)
                print(temp.percentAvailable)
        if temp.percentAvailable / 100 < (
                share - 0.002) or temp.percentAvailable / 100 > (
                share + 0.002):
            raise ValueError(
                "Couldn't meet percentage within tolerance. " +
                "Percentage achieved by wind speed is: ",
                temp.percentAvailable)
        print(temp.percentAvailable)
        size = 10000
        while temp.percentAvailable / 100 > (share + 0.00004):
            if temp.percentAvailable > 0:
                temp.pruneIsolatedAreas(minSize=size)
            size += 100
        print(temp.percentAvailable)
        self._run_exclusion(_exclusion_dict)
        if self.ec.percentAvailable > 0:
            self.ec.pruneIsolatedAreas(minSize=size)
        if self.ec.percentAvailable/100 < (share - tolerance):
            print("Conflict with tolerance, please use smaller step size.")

    def estimate_potential(
            self, predict=True, exclusion_dict=None, restrict_area=None, **args):
        """Estimate wind potential in region.

        Uses run_exclusion and get_wind_items.

        Parameters
        ----------
        predict : bool, optional
            If elements shall be predicted / placed by GLAES, by default True
        exclusion_dict : dict or str, optional
            Dictionary containing the information for the exclusion.
            A default dict for wind is present and can be updated in certain
            points. If str is given config file is used (data/config).
            By default None.
        restrict_area : float, optional
            desired share of area, by default None
        """
        self.predicted_items = self.parent.check_db(self)
        if self.predicted_items is None:
            self.report_dict = self.run_exclusion(exclusion_dict, **args)
            if restrict_area is not None:
                self.restrict_area(share=restrict_area)

            if predict:
                self.distribute_items()
                self.report_dict["Items_Number"] = self.ec._itemCoords.shape[0]
                self.report_dict["Capacity"] = self.predicted_items['capacity'].sum()

    def sim(self):
        """Simulate time-series of predicted wind turbines."""
        self.ts_predicted_items = self.parent.check_db(self, "ts")
        if self.ts_predicted_items is None:
            self.ts_predicted_items, self.predicted_items = self.sim_wind(
                self.predicted_items, turbine=self.turbine)

    def sim_existing(self):
        """Simulate time-series of existing wind turbines."""
        if self.existing_items is None:
            self.get_existing_plants(self.ec)
        if len(self.existing_items) > 0:
            self.ts_existing_items, self.existing_items = self.sim_wind(
                self.existing_items)
        else:
            print("No existing wts --> not simulating")

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
        path_LE = os.path.join(self.result_path, "Wind_potential_area.tif")
        assert os.path.isfile(path_LE), f"Can't find the LE result {path_LE}"
        if self.ec._hasEqualContext(path_LE):
            print("The LE result can be directly accepted", flush=True)
            initial_LE = gk.raster.extractMatrix(path_LE)
        else:
            print("The LE result has to be warped to the mask region", flush=True)
            path_LE_warp = os.path.join(self.result_path, "Wind_potential_area_warp.tif")
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
        technology = "Wind"
        self._merge_to_germany(path_states, technology)

    def distribute_items_germany(self, path_LE=None, geometry_shape="ellipse", mode="QuWind100",
                                 optional_turbines=("E-126_7580", "E115_3200"), path_netCDF=None,
                                 **kwargs):
        """

        :param path_LE:
        :return:
        """
        assert self.parent.level == "country"
        res = self.parent.regionMask._pixelRes
        srs = self.parent.regionMask.srs
        path_mun = os.path.join(self.parent.datasource_path,
                                       "germany_administrative",
                                       "vg250_ebenen",
                                       "VG250_GEM.shp")

        if path_LE is None:
            path_LE = self.parent.case_path
        elif isinstance(path_LE, str):
            pass
        else:
            raise TypeError("path_LE has to be path to the dictionary of LE results or be set to None")
        # path_available_area = os.path.join(self.result_path, "remaining_available_area.tif")
        path_items = os.path.join(self.result_path, "Wind_turbine_coordinate.csv")
        all_mun_features = gk.vector.extractFeatures(path_mun)
        all_mun = all_mun_features[["geom", "RS"]]

        # load wind direction from era
        if self.wind_dir == "from_era":
            wind_dir_raster = gk.raster.loadRaster(
                os.path.join(self.parent.datasource_path, "era5",
                             "ERA5_wind_direction_100m_mean.tiff"))
        # load netCDF data of turbines. Or get path of wind speed data from GlobalWindAtlas
        if mode == "QuWind100":
            files = os.listdir(path_netCDF)
            path_turbines_CF = {}
            turbines = rk.wind.TurbineLibrary().loc[optional_turbines, :]
            for turbine in optional_turbines:
                for file in files:
                    # TODO right now only the highest hub height is considered
                    if file.endswith(".nc") and turbine in file and str(int(float(turbines.loc[turbine, "Hub_Height"][-1]))) in file:
                        path_turbines_CF[turbine] = os.path.join(path_netCDF, file)
                        print(f"find {file}", flush=True)
                        break
                else:
                    print(f"can not find the CF file of {turbine}", flush=True)
            turbines_disk = {}
            for turbine in path_turbines_CF.keys():
                turbines_disk[turbine] = xr.open_dataset(path_turbines_CF[turbine])
        elif mode == "fromRK":
            path_gwa_de = os.path.join(self.parent.datasource_path, "gwa", "DEU_wind-speed_100m.tif")

        mun_geom_with_items = []
        mun_rs_with_items = []
        all_items = []
        mun_use_turbine = {key: list() for key in optional_turbines}
        for i in range(len(all_mun)):
            start = time.time()
            print(f"distribute in {all_mun.loc[i]['RS']}", flush=True)
            state = all_mun.loc[i]['RS'][0:2]
            turbine = {"Rotordiameter": self.target_diameter, "Capacity": self.capacity, "Hub_Height": self.hub_height}
            trep_mun = trep.TREP(region=all_mun.loc[i]["RS"], level="MUN", case="temp",
                                 db_path=self.parent.db_path,
                                 datasource_path=self.parent.datasource_path,
                                 pixelRes=res,
                                 srs=srs)
            trep_mun.Wind.target_diameter = turbine["Rotordiameter"]
            trep_mun.Wind.capacity = turbine["Capacity"]
            trep_mun.Wind.distance = self.distance
            # print("load available area", flush=True)
            trep_mun.Wind.ec.excludeRasterType(os.path.join(path_LE, f"Wind_{state}", "Wind_potential_area.tif"),
                                               value=0, buffer=0)
            # print(f"loaded available area after {time.time() - start} sec", flush=True)
            for k in range(len(mun_geom_with_items)):
                if mun_geom_with_items[k].Intersect(all_mun.loc[i]['geom']):
                    print(f"exclude items of neighbour {mun_rs_with_items[k]}", flush=True)
                    trep_mun.Wind.ec.excludePoints(source=all_items[k], geometry_shape=geometry_shape,
                                                   direction=self.wind_dir)
            # print(f"exclude items after {time.time() - start} sec", flush=True)
            trep_mun.Wind.distribute_items()
            if len(trep_mun.Wind.predicted_items) == 0:
                columns = list(trep_mun.Wind.predicted_items.columns)
                columns.append("Power_Generation")
                columns.append("LCOE")
                trep_mun.Wind.predicted_items = pd.DataFrame(columns=columns)
                pass
            elif mode == "fromRK":
                wind_speeds_100 = gk.raster.interpolateValues(path_gwa_de, trep_mun.Wind.ec.saveItems()["geom"])
                # Adjust the wind speed to hub height
                # TODO consider the roughness by CLC Land Cover
                roughness = rk.wind.roughness_from_clc(
                    os.path.join(self.parent.datasource_path, "clc", "U2018_CLC2018_V2020_20u1.tif"),
                    trep_mun.Wind.ec.saveItems()["geom"],
                    window_range=1)
                wind_speeds = rk.wind.apply_logarithmic_profile_projection(wind_speeds_100, 100, self.hub_height, roughness)
                # TODO how to change base line turbine, or can we change base line turbine?
                turbines_parameters = rk.wind.onshore_turbine_from_avg_wind_speed(wind_speeds,
                                                                                  base_rotor_diam=self.target_diameter,
                                                                                  base_hub_height=self.hub_height,
                                                                                  base_capacity=self.capacity)
                trep_mun.Wind.predicted_items["capacity"] = turbines_parameters["capacity"]
                trep_mun.Wind.predicted_items["hub_height"] = turbines_parameters["hub_height"]
                trep_mun.Wind.predicted_items["rotor_diam"] = turbines_parameters["rotor_diam"]
                trep_mun.Wind.predicted_items["specific_power"] = turbines_parameters["specific_power"]
            elif mode == "QuWind100":
                # print(f"find possible locations after {time.time() - start} sec", flush=True)
                optimal_turbine = self.optimal_turbine(trep_mun.Wind.ec.saveItems()["geom"], turbines, turbines_disk,
                                                       **kwargs)
                # print(f"select optimal turbine after {time.time() - start} sec", flush=True)
                del trep_mun.Wind.predicted_items
                mun_use_turbine[optimal_turbine["Name"]].append(all_mun.loc[i]['RS'])
                trep_mun.Wind.target_diameter = optimal_turbine["Rotordiameter"]
                trep_mun.Wind.capacity = optimal_turbine["Capacity"]
                trep_mun.Wind.hub_height = optimal_turbine["Hub_Height"]
                trep_mun.Wind.distribute_items()
                # estimate energy yield based on the new locations, unit TWh
                trep_mun.Wind.predicted_items["Power_Generation"] = \
                    self.estimate_generation_with_QuWind100(trep_mun.Wind.ec.saveItems()["geom"],
                                                            optimal_turbine, turbines_disk)
                # TODO estimate LCOE
                trep_mun.Wind.predicted_items["LCOE"] = \
                    self.estimate_LCOE_at_locations(trep_mun.Wind.ec.saveItems()["geom"],
                                                    turbines_disk[optimal_turbine["Name"]],
                                                    trep_mun.Wind.capacity,
                                                    trep_mun.Wind.hub_height,
                                                    trep_mun.Wind.target_diameter)
            # print(f"distributed items after {time.time() - start} sec", flush=True)
            if i == 0:
                trep_mun.Wind.predicted_items.to_csv(path_items, mode="w")
            else:
                trep_mun.Wind.predicted_items.to_csv(path_items, mode="a", header=False)
            if len(trep_mun.Wind.predicted_items) == 0:
                # print("no items can be distributed", flush=True)
                continue
            else:
                # print("save items and municipality in list", flush=True)
                mun_geom_with_items.append(all_mun.loc[i]['geom'])
                mun_rs_with_items.append(all_mun.loc[i]['RS'])
                # prepare "scale" and "direction" for excludePoints()
                trep_mun.Wind.predicted_items["scale"] = trep_mun.Wind.predicted_items.apply(
                    lambda x: np.array([self.distance[0] * x["rotor_diam"], self.distance[1] * x["rotor_diam"]]), axis=1)
                trep_mun.Wind.predicted_items["geom"] = trep_mun.Wind.ec.saveItems()["geom"]
                if self.wind_dir == "from_era":
                    # Direction has to be in vector file or dataframe
                    trep_mun.Wind.predicted_items["direction"] = \
                        trep_mun.Wind.predicted_items.apply(lambda x:
                                                            gk.raster.interpolateValues(wind_dir_raster, x["geom"]),
                                                            axis=1)
                # print(f"prepared exclusion point after {time.time() - start} sec", flush=True)
                all_items.append(trep_mun.Wind.predicted_items)
        # save for each optional turbine the municipalities, that use this turbine
        for turbine, mun_list in mun_use_turbine.items():
            if len(mun_list) > 0:
                mun_features_use_turbine = all_mun_features.loc[all_mun_features["RS"].isin(mun_list)]
                gk.vector.createVector(mun_features_use_turbine,
                                       output=os.path.join(self.result_path, f"Municipalities_use_{turbine}.shp")
                                       )

    # def distribute_items_state(self, path_LE=None, geometry_shape="ellipse",
    #                            optional_turbines=("E-82_E4_2350", "E115_3200"), path_netCDF=None):
    #     """
    #     :param path_LE:
    #     :return:
    #     """
    #     assert self.parent.level == "state"
    #     res = self.parent.regionMask._pixelRes
    #     srs = self.parent.regionMask.srs
    #     path_mun = os.path.join(self.parent.datasource_path,
    #                                    "germany_administrative",
    #                                    "vg250_ebenen",
    #                                    "VG250_GEM.shp")
    #
    #     if path_LE is None:
    #         path_LE = os.path.join(self.result_path, "Wind_potential_area.tif")
    #         assert os.path.isfile(path_LE)  # The eligible area must exist
    #     elif isinstance(path_LE, str):
    #         pass
    #     else:
    #         raise TypeError("path_LE has to be path to a tif data or be set to None")
    #     path_available_area = os.path.join(self.result_path, "remaining_available_area.tif")
    #     path_items = os.path.join(self.result_path, "Wind_turbine_coordinate.csv")
    #
    #     all_mun_features = gk.vector.extractFeatures(path_mun)
    #     all_mun = all_mun_features[["geom", "RS"]]
    #     all_mun["RS_nuts"] = all_mun.apply(lambda x: x["RS"][0:2], axis=1)
    #     all_mun = all_mun.loc[all_mun.RS_nuts == self.parent._state]
    #     all_mun = all_mun.reset_index(drop=True)
    #
    #     # initialize the available area
    #     print("initialize_the_remaining available area", flush=True)
    #     # mask_de = gk.RegionMask.fromVector(path_mask_de, pixelRes=res, srs=srs, limitOne=False, where="RS='07'")
    #     # ec_de = gl.ExclusionCalculator(mask_de, pixelRes=res, srs=srs)
    #     ec_de = self.parent.new_ec()
    #     if ec_de._hasEqualContext(path_LE):
    #         print("The LE result can be directly accepted", flush=True)
    #         initial_LE = gk.raster.extractMatrix(path_LE)
    #     else:
    #         print("The LE result has to be warped to the mask region", flush=True)
    #         path_LE_warp = os.path.join(self.result_path, "Wind_potential_area_warp.tif")
    #         ec_de.region.warp(path_LE, output=path_LE_warp)
    #         print("Warp finished", flush=True)
    #         initial_LE = gk.raster.extractMatrix(path_LE_warp)
    #     initial_LE = np.where(initial_LE == 100, 100, 0)
    #     ec_de._availability = initial_LE
    #     print("LE result is loaded", flush=True)
    #     ec_de.save(path_available_area)
    #
    #     # load wind direction from era
    #     if self.wind_dir == "from_era":
    #         wind_dir_raster = gk.raster.loadRaster(
    #             os.path.join(self.parent.datasource_path, "era5",
    #                          "ERA5_wind_direction_100m_mean.tiff"))
    #
    #     # load netCDF data of turbines
    #     files = os.listdir(path_netCDF)
    #     path_turbines_CF = {}
    #     turbines = rk.wind.TurbineLibrary().loc[optional_turbines, :]
    #     for turbine in optional_turbines:
    #         for file in files:
    #             # TODO right now only the highest hub height is considered
    #             if file.endswith(".nc") and turbine in file and str(
    #                     int(float(turbines.loc[turbine, "Hub_Height"][-1]))) in file:
    #                 path_turbines_CF[turbine] = os.path.join(path_netCDF, file)
    #                 print(f"find {file}", flush=True)
    #                 break
    #         else:
    #             print(f"can not find the CF file of {turbine}", flush=True)
    #     turbines_disk = {}
    #     for turbine in path_turbines_CF.keys():
    #         turbines_disk[turbine] = xr.open_dataset(path_turbines_CF[turbine])
    #
    #     for i in range(len(all_mun)):
    #         start = time.time()
    #         print(f"distribute in {all_mun.loc[i]['RS']}", flush=True)
    #         turbine = {"Rotordiameter": self.target_diameter, "Capacity": self.capacity, "Hub_Height": self.hub_height}
    #         trep_mun = trep.TREP(region=all_mun.loc[i]["RS"], level="MUN", case="temp",
    #                              db_path=self.parent.db_path,
    #                              datasource_path=self.parent.datasource_path,
    #                              pixelRes=res,
    #                              srs=srs)
    #         trep_mun.Wind.target_diameter = turbine["Rotordiameter"]
    #         trep_mun.Wind.capacity = turbine["Capacity"]
    #         trep_mun.Wind.distance = self.distance
    #         print("load available area", flush=True)
    #         trep_mun.Wind.ec.excludeRasterType(path_available_area, value=0, buffer=0)
    #         print(f"loaded available area after {time.time() - start} sec", flush=True)
    #         trep_mun.Wind.distribute_items()
    #         if len(trep_mun.Wind.predicted_items) == 0:
    #             pass
    #         else:
    #             # print(f"find possible locations after {time.time() - start} sec", flush=True)
    #             geoms = trep_mun.Wind.ec.saveItems()["geom"]
    #             optimal_turbine = self.optimal_turbine(geoms, turbines, turbines_disk)
    #             # print(f"select optimal turbine after {time.time() - start} sec", flush=True)
    #             del trep_mun.Wind.predicted_items
    #             trep_mun.Wind.target_diameter = optimal_turbine["Rotordiameter"]
    #             trep_mun.Wind.capacity = optimal_turbine["Capacity"]
    #             trep_mun.Wind.hub_height = optimal_turbine["Hub_Height"]
    #             trep_mun.Wind.distribute_items()
    #             # estimate energy yield based on the new locations, unit TWh
    #             trep_mun.Wind.predicted_items["Power_Generation"] = \
    #                 self.estimate_generation_with_QuWind100(trep_mun.Wind.ec.saveItems()["geom"],
    #                                                         optimal_turbine, turbines_disk)
    #         print(f"distributed items after {time.time() - start} sec", flush=True)
    #         if i == 0:
    #             trep_mun.Wind.predicted_items.to_csv(path_items, mode="w")
    #         else:
    #             trep_mun.Wind.predicted_items.to_csv(path_items, mode="a", header=False)
    #         if len(trep_mun.Wind.predicted_items) == 0:
    #             print("no items can be distributed", flush=True)
    #             continue
    #         else:
    #             print("update available area", flush=True)
    #             # prepare "scale" and "direction" for excludePoints()
    #             trep_mun.Wind.predicted_items["scale"] = trep_mun.Wind.predicted_items.apply(
    #                 lambda x: np.array([self.distance[0] * x["rotor_diam"], self.distance[1] * x["rotor_diam"]]), axis=1)
    #             trep_mun.Wind.predicted_items["geom"] = trep_mun.Wind.ec.saveItems()["geom"]
    #
    #             if self.wind_dir == "from_era":
    #                 # Direction has to be in vector file or dataframe
    #                 trep_mun.Wind.predicted_items["direction"] = \
    #                     trep_mun.Wind.predicted_items.apply(lambda x:
    #                                                         gk.raster.interpolateValues(wind_dir_raster, x["geom"]),
    #                                                         axis=1)
    #             print(f"prepared exclusion point after {time.time() - start} sec", flush=True)
    #             ec_de.excludePoints(source=trep_mun.Wind.predicted_items, geometry_shape=geometry_shape,
    #                                 direction=self.wind_dir)
    #             print(f"excluded points after {time.time() - start} sec", flush=True)
    #             ec_de.save(path_available_area)
    #             print(f"save available area after {time.time() - start} sec", flush=True)

    def optimal_turbine(self, geoms, optional_turbines, turbines_disk, KPI="density"):
        """
        Choose the best turbines from the given optional turbines at specific locations.

        Parameters
        ----------
        geoms: array like, objects of gdal.Geometry
            Points geometries of the locations.
        optional_turbines: pandas.DataFrame
            Informations of turbines in question.
        turbines_disk: dict
            Dictionary contains the xarray.Dataset of all turbines.
        KPI: str
            "density" or "CF" or "LCOE"
                "density": use power generation density as KPI for comparision between turbines.
                "CF": use capacity factor (full load hours) as KPI for comparision between turbines.
                "LCOE": use levelized cost of electricity as KPI for comparision between turbines.

        Returns
        --------
        pandas.Series
            Information of the optimal turbine
        """
        # calculate KPI to select optimal turbine
        turbine_KPI = {}
        for turbine in turbines_disk.keys():
            CF = turbines_disk[turbine].CF
            cf_list = self.get_CF_at_locations(geoms, CF)
            power_list = [cf * optional_turbines.loc[turbine, "Capacity"] for cf in cf_list]
            # print(f"{turbine} has capacity: {optional_turbines.loc[turbine, 'Capacity']}", flush=True)
            # print(f"{turbine} has cf_mean: {cf_mean}", flush=True)
            A_demand = self.distance[0] * optional_turbines.loc[turbine, "Rotordiameter"] * \
                       self.distance[1] * optional_turbines.loc[turbine, "Rotordiameter"]
            # print(f"{turbine} has A demand: {A_demand}", flush=True)
            if KPI == "density":
                turbine_KPI[turbine] = sum(power_list) / (len(power_list) * A_demand)
            elif KPI == "CF":
                turbine_KPI[turbine] = sum(cf_list)/len(cf_list)
            elif KPI == "LCOE":
                # TODO use arithmetic mean value or weighted mean value?
                # lcoe = [self.estimate_LCOE(optional_turbines.loc[turbine, "Capacity"],
                #                            int(float(turbines_disk[turbine].hub_height)),
                #                            optional_turbines.loc[turbine, "Rotordiameter"],
                #                            cf) * cf for cf in cf_list]
                # turbine_KPI[turbine] = sum(lcoe) / sum(cf_list)
                lcoe = self.estimate_LCOE(optional_turbines.loc[turbine, "Capacity"],
                                           int(float(turbines_disk[turbine].hub_height)),
                                           optional_turbines.loc[turbine, "Rotordiameter"],
                                           sum(cf_list)/len(cf_list))
                turbine_KPI[turbine] = -lcoe  # use max() to compare KPI
            # print(f"{turbine} has KPI: {turbine_KPI[turbine]}", flush=True)
        optimal_turbine_name_name = max(turbine_KPI, key=turbine_KPI.get)
        optimal_turbine = optional_turbines.loc[optimal_turbine_name_name].copy()
        optimal_turbine["Hub_Height"] = int(float(turbines_disk[optimal_turbine_name_name].hub_height))
        optimal_turbine["Name"] = optimal_turbine_name_name
        return optimal_turbine

    def estimate_generation_with_QuWind100(self, geoms, turbine, turbines_disk):
        """
        Estimate the power generations of a given turbine at the specific locations.

        Parameters
        ----------
        geoms: array like, objects of gdal.Geometry
            Points geometries of the locations.
        turbine: pandas.Series
            Information of the given turbine.
        turbines_disk: dict
            Dictionary contains the xarray.Dataset of all turbines.

        Returns
        --------
        numeric or array-like
            Expected power generation in [TWh/a]
        """
        CF = turbines_disk[turbine["Name"]].CF
        cf_list = self.get_CF_at_locations(geoms, CF)
        cf_list = np.array(cf_list)
        power_generation = cf_list * turbine["Capacity"] * 365 * 24 / 1E9  # unit TWh
        return power_generation

    def estimate_LCOE_at_locations(self, geoms, turbine_disk, capacity, hub_height, rotor_diam):
        """
        Estimate the power generations of a given turbine at the specific locations.

        Parameters
        ----------
        geoms: array like, objects of gdal.Geometry
            Points geometries of the locations.
        turbine_disk: xarray.Dataset
            Dataset, that is loaded from netCDF data.
        capacity: numeric
            Capacity of turbine in [m].
        hub_height: numeric
            Hub height of turbine in [m].
        rotor_diam: numeric
            Rotor diameter of turbine in [m].

        Returns
        --------
        numeric or array-like
            Levelized cost of electricity (LCOE) in [ct/KWh]
        """
        CF = turbine_disk.CF
        cf_list = self.get_CF_at_locations(geoms, CF)
        LCOE = [self.estimate_LCOE(capacity, hub_height, rotor_diam, cf) for cf in cf_list]
        return LCOE

    @staticmethod
    def estimate_LCOE(capacity, hub_height, rotor_diam, cf):
        capex = rk.wind.onshore_turbine_capex(capacity, hub_height, rotor_diam)
        sp_opex = 2/100
        n = 20
        r = 8/100
        gen = cf * capacity * 365 * 24  # unit [KWh]
        if gen == 0:
            LCOE = np.nan
        else:
            LCOE = capex / gen * (r / (1 - (1+r)**(-n)) + sp_opex) * 100  # unit [Euro cent/ KWh]
        return LCOE

    @staticmethod
    def get_CF_at_locations(geoms, CF_data_array):
        # transform to CRS 3035
        points = [gk.geom.transform(p, toSRS=3035) for p in geoms]
        cf_list = []
        for p in points:
            x = p.GetX()
            y = p.GetY()
            x_index = int((x - 4030050) // 100)
            y_index = int((y - 2680050) // 100)
            cf = float(CF_data_array[y_index, x_index].data)
            if np.isnan(cf):
                cf_list.append(0)
            else:
                cf_list.append(cf)
        return cf_list
