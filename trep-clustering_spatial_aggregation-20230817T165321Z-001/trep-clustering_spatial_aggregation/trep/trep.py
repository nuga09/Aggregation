"""Created 15.06.2020 by s.risch."""

import glaes as gl
import os
import geokit as gk
import numpy as np
from warnings import warn
import pandas as pd
import trep.utils as utils
from trep.wind import Wind
from trep.openfield_pv import OpenfieldPV, OpenfieldPVRoads
from trep.rooftop_pv import RooftopPV
import shutil
import osgeo
import time


class TREP(object):
    """Object to Estimate Regional Renewable Energy Potentials."""

    def __init__(self,
                 region,
                 level="nuts3",
                 case="base",
                 db_path=None,
                 datasource_path=None,
                 intermediate_path=None,
                 dlm_basis_path=None,
                 hu_path=None,
                 use_intermediate=False,
                 pixelRes=10,
                 srs=3035):
        """Initialize trep.

        Parameters
        ----------
        region : str, any
            name or id of region to estimate potential
        level : str, optional
            ["nuts3","state","Mun", "country"], by default "nuts3"
        case : str, optional
            name of observed case, by default "base"
        db_path : str, optional
            path to the database, by default None
        datasource_path: str, optional
            path to external data source, use internal data source if not given, by default None
        intermediate_path: str, optional
            path to intermediate files, by default None
        dlm_basis_path: str, optional
            path to dlm-basis data, by default None
        pixelRes : int, optional
            pixel resolution, by default 10
        srs : int, optional
            spatial reference system, by default 3035
        use_intermediate: bool, optional
            if true use intermediate file for exclusion calculation, by default false
        """
        if not isinstance(region, list):
            self.region = [region]
        else:
            self.region = region
        self.case = case
        self.techs = {"Wind": None,
                      "OpenfieldPV": None,
                      "OpenfieldPVRoads": None,
                      "RooftopPV": None}
        self.exclusionCalculators = {}
        self.available_areas = {}
        self.available_areas_old = {}
        self.level = level
        self.db_path = db_path
        if self.db_path is None:
            self.db_path = os.path.join(utils.get_data_path(), "database")
        elif self.db_path == "CAESAR":
            self.db_path = r"/storage/internal/data/s-risch/db_TREP/"
        elif isinstance(self.db_path, str):
            pass  # Assume is path
        self.datasource_path = datasource_path
        if self.datasource_path is None:
            self.datasource_path = os.path.join(utils.get_datasources_path())
        elif self.datasource_path == "CAESAR":
            self.datasource_path = r"/storage/internal/data/s-risch/shared_datasources/"
        elif isinstance(self.datasource_path, str):
            pass  # Assume is path
        self.use_intermediate = use_intermediate
        self.intermediate_path = intermediate_path
        if self.intermediate_path is None:
            self.intermediate_path = os.path.join(utils.get_datasources_path(), "intermediates")
        elif self.intermediate_path == "CAESAR":
            self.intermediate_path = r"/storage/internal/data/s-risch/shared_datasources/shared_intermediates/"
        elif isinstance(self.intermediate_path, str):
            pass  # Assume is path
        self.dlm_basis_path = dlm_basis_path
        if self.dlm_basis_path is None:
            self.dlm_basis_path = os.path.join(self.datasource_path, "basis-dlm")
        elif self.dlm_basis_path == "CAESAR":
            self.dlm_basis_path = "/storage/internal/data/res/bkg/merged/300001227_2053_Basis-DLM/basis-dlm-aaa_ebenen/"
        elif isinstance(self.dlm_basis_path, str):
            pass  # Assume is path
        self.hu_path = hu_path
        if self.hu_path is None:
            self.hu_path = os.path.join(self.datasource_path, "hu")
        elif self.hu_path == "CAESAR":
            self.hu_path = "/storage/internal/data/res/bkg/merged/300001227_2054_HU-DE/hu-de/"
        elif isinstance(self.hu_path, str):
            pass  # Assume is path
        self.get_regionMask(srs, pixelRes)
        # TODO see how to add path for each case
        self.case_path = os.path.join(self.db_path, case)
        if not os.path.exists(self.case_path):
            os.mkdir(self.case_path)
        if level == "nuts3":
            self.get_municipalities()
        # # TODO why only check for RoofPV? And why only for municipality?
        # path = os.path.join(self.db_path, case, "RooftopPV_{}".format(self.id))
        # if not os.path.exists(path) and self.level == "MUN":
        #     os.mkdir(path)

        self.add_all()

    def get_regionMask(self, srs, pixelRes):
        """Load RegionMask.

        Parameters
        ----------
        srs : int
            spatial reference system
        pixelRes : int
            pixel resolution
        """
        if self.level == "MUN":
            path = os.path.join(self.datasource_path,
                                "germany_administrative",
                                "vg250_ebenen",
                                "VG250_GEM.shp")
            for i, el in enumerate(self.region):
                if el[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    _str = "RS='{}'".format(el)
                else:
                    _str = "GEN='{}'".format(el)
                if i == 0:
                    region_str = _str
                else:
                    region_str += " OR {}".format(_str)
        elif self.level == "nuts3":
            path = os.path.join(self.datasource_path,
                                "germany_administrative",
                                "vg250_ebenen",
                                "VG250_KRS.shp")
            for i, el in enumerate(self.region):
                if el[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    _str = "RS='{}'".format(el)
                else:
                    _str = "GEN='{}'".format(el)
                if i == 0:
                    region_str = _str
                else:
                    region_str += " OR {}".format(_str)

        elif self.level == "state":
            for i, el in enumerate(self.region):
                path = os.path.join(self.datasource_path,
                                    "germany_administrative",
                                    "vg250_ebenen",
                                    "VG250_LAN.shp")
                if el[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    _str = "RS='{}'".format(el)
                else:
                    _str = "GEN='{}'".format(el)
                if i == 0:
                    region_str = _str
                    if el == "NRW":
                        region_str = "RS='05'"
                    elif el == "Bayern":
                        region_str = "RS='09'"
                    elif el == "Baden-Württemberg":
                        region_str = "RS='08'"
                    elif el == "Thüringen":
                        region_str = "RS='16'"
                    elif el == "Schleswig-Holstein":
                        region_str = "RS='01'"
                else:
                    if el == "Bayern":
                        _str = "RS='09'"
                    elif el == "Baden-Württemberg":
                        _str = "RS='08'"
                    region_str += " OR {}".format(_str)
            region_str = f"({region_str}) and GF != 1"
        elif self.level == "country":
            path = os.path.join(self.datasource_path,
                                "germany_administrative",
                                "vg250_ebenen",
                                "VG250_LAN.shp")
            print(self.region, flush=True)
            if self.region[0] in ("germany", "Deutschland"):
                region_str = "GF != 1"
            else:
                # TODO for off shore may also need to add a "elif"
                region_str = None
        print(region_str)
        self.features = gk.vector.extractFeatures(path, where=region_str)
        if self.level == "MUN":
            self._id = self.features["RS"][0]
            self.rs = self.features["RS"][0]
            self._ags = self.features["AGS"][0]
            self.nuts3 = self.features["NUTS"]
        elif self.level == "nuts3":
            self._id = self.features["RS"][0]
            self.rs = self.features["RS"][0]
            self.nuts3 = self.features["NUTS"]
        elif self.level == "state":
            self._id = self.features["RS"].values[0]
            self.rs = self.features["RS"][0]
        elif self.level == "country":
            self._id = "00"
            self.rs = "00"
        self._state = self.rs[0:2]
        if not all([geom.IsValid() for geom in self.features.geom]):
            self.features.geom = self.features.apply(lambda x: x["geom"].Buffer(0),
                                                    axis=1)
            for i, geom in enumerate(self.features.geom.values):
                if i == 0:
                    _geom = geom
                else:
                    _geom = _geom.Union(geom)
            self.regionMask = gk.RegionMask.fromGeom(_geom,
                                                    srs=srs,
                                                    pixelRes=pixelRes)
        else:
            self.regionMask = gk.RegionMask.fromVector(path,
                                                    where=region_str,
                                                    limitOne=False,
                                                    srs=srs,
                                                    pixelRes=pixelRes)

    def get_municipalities(self):
        """Get the municipalities in a region."""
        path = os.path.join(self.datasource_path,
                            "germany_administrative",
                            "vg250_ebenen",
                            "VG250_GEM.shp")
        # mun_string = "RS='{}'".format(self._id)
        self.municipalities = gk.vector.extractFeatures(path)
        self.municipalities["RS_nuts"] = self.municipalities.apply(
            lambda x: x["RS"][0:5], axis=1)
        self.municipalities = \
            self.municipalities.loc[
                self.municipalities.RS_nuts == self.rs]["RS"].values

    def get_population(self):
        """Get the population of the region.

        Raises
        ------
        NotImplementedError
            when level of region is not implemented
        """
        path = os.path.join(
            self.datasource_path,
            "other", "Zensus2011_Bevoelkerung",
            "Zensus11_Datensatz_Bevoelkerung.csv")
        df_zensus = pd.read_csv(
            path, sep=";", encoding="utf-8",
            dtype={"AGS_12": str, "RS_Land": str, "RS_RB_NUTS2": str,
                   "RS_Kreis": str, "RS_VB": str, "RS_Gem": str})
        if self.level == "nuts3":
            df_zensus = df_zensus[df_zensus.Reg_Hier ==
                                  "Stadtkreis/kreisfreie Stadt/Landkreis"]
            df_zensus["RS"] = df_zensus["RS_Land"] + \
                df_zensus["RS_RB_NUTS2"] + df_zensus["RS_Kreis"]
        elif self.level == "MUN":
            df_zensus["RS"] = df_zensus["RS_Land"] + \
                df_zensus["RS_RB_NUTS2"] + df_zensus["RS_Kreis"] + \
                df_zensus["RS_VB"] + df_zensus["RS_Gem"]
        else:
            raise NotImplementedError
        self.zensus = df_zensus[df_zensus.RS == self.rs]
        self.population = self.zensus["AEWZ"].values[0]

    def new_ec(self, **args):
        """Load new GLAES Exclusion calculator."""
        return gl.ExclusionCalculator(self.regionMask, **args)

    def add_tech(self, tech, **kwargs):
        """Add the technology.

        Parameters
        ----------
        tech : str
            name of technology
        """
        if tech == "Wind":
            self.techs[tech] = Wind(parent=self, **kwargs)
        elif tech == "OpenfieldPV":
            self.techs[tech] = OpenfieldPV(parent=self, **kwargs)
        elif tech == "OpenfieldPVRoads":
            self.techs[tech] = OpenfieldPVRoads(parent=self, **kwargs)
        elif tech == "RooftopPV":
            self.techs[tech] = RooftopPV(parent=self, **kwargs)

    def add_all(self):
        """Add all technologies"""
        for tech in self.techs.keys():
            self.add_tech(tech)

    def estimate_shared_potential(
            self, techs=["Wind", "OpenfieldPV", "OpenfieldPVRoads"]):
        """Estimate the shared potential on areas with potential for several
        technologies.

        Parameters
        ----------
        techs : list, optional
            name of technologies,
            by default ["Wind", "OpenfieldPV", "OpenfieldPVRoads"]
        """
        rasters = []
        ecs = []
        for tech in techs:
            if self.techs[tech] is None:
                self.add_tech(tech)
            if self.techs[tech].ec.percentAvailable == 100 or self.techs[tech].ec.percentAvailable == 0:
                self.techs[tech].run_exclusion()
            ecs.append(self.techs[tech].ec)
        for i, ec in enumerate(ecs):
            if i == len(ecs):
                pass
            else:
                rasters.append(ec.region.createRaster(
                    data=ec._availability, noData=255,
                    pixelRes=self.regionMask.pixelRes))
            if i > 0:
                for j, raster in enumerate(rasters[0:i]):
                    ecs[i].excludeRasterType(rasters[j], value=(50, 100))
        for i, tech in enumerate(techs):
            self.techs[tech].ec = ecs[i]
            self.techs[tech].predicted_items = None
            self.techs[tech].distribute_items()

    def estimate_hybrid_potential(self):
        """Estimate the potential of using OFPV in the usable wind areas."""
        # Make all Openfield PV area unavailable
        self.OpenfieldPV.ec._availability.fill(0)
        # If wind potential has not been evaluated, do so
        if self.Wind.ec.percentAvailable == 100:
            self.Wind.estimate_potential()
        # Re-Include areas, which are used for wind
        raster = self.Wind.ec.region.createRaster(
            data=self.Wind.ec._availability, noData=255,
            pixelRes=self.regionMask.pixelRes)
        self.OpenfieldPV.ec.excludeRasterType(
            raster, value=(50, 100), mode='include')
        # Exclude agricultural areas and forests
        self.OpenfieldPV._run_exclusion(
            exclusion_dict={'agriculture': 0, 'forests': 0})
        self.OpenfieldPV.distribute_items()

    def sim_all(self):
        """Simulate predicted and existing items of technologies."""
        for name, tech in self.techs.items():
            if tech is None:
                print("Technology {} has not been analysed yet. ".format(
                    name) + "Cannot simulate.")
            else:
                tech.sim()
                tech.sim_existing()

    def existing_to_db(self, tech):
        """Save existing technologies and their time-series to database.

        Parameters
        ----------
        tech : str
            name of technology
        """
        if tech == "RooftopPV":
            if self.techs[tech].existing_items is not None:
                self.techs[tech].existing_items.to_csv(os.path.join(
                    self.db_path, self.case, "RooftopPV_{}".format(self.id),
                    "existing_{}_{}.csv".format(tech, self.id)
                ))
            if self.techs[tech].ts_existing_items is not None:
                self.techs[tech].ts_existing_items.to_csv(os.path.join(
                    self.db_path, self.case, "RooftopPV_{}".format(self.id),
                    "ts_existing_{}_{}.csv".format(tech, self.id)
                ))
        else:
            if self.techs[tech].existing_items is not None:
                self.techs[tech].existing_items.to_csv(
                    os.path.join(
                        self.db_path, self.case, "existing_{}_{}.csv".format(
                            tech, self.id)))
            if self.techs[tech].ts_existing_items is not None:
                self.techs[tech].ts_existing_items.to_csv(os.path.join(
                    self.db_path,
                    self.case,
                    "ts_existing_{}_{}.csv".format(tech, self.id)
                ))

    def check_existing_db(self, tech):
        """Get existing items and their time-series from db - if available.

        Parameters
        ----------
        tech : str
            name of technology
        """
        if tech == "RooftopPV":
            path = os.path.join(self.db_path, self.case, "RooftopPV_{}".format(
                self.id), "{}_{}.csv".format("existing_RooftopPV", self.id))
            if os.path.exists(path):
                self.RooftopPV.existing_items = pd.read_csv(path, index_col=0)
            path = os.path.join(self.db_path, self.case, "RooftopPV_{}".format(
                self.id), "{}_{}.csv".format("ts_existing_RooftopPV", self.id))
            if os.path.exists(path):
                self.RooftopPV.ts_existing_items = pd.read_csv(
                    path, index_col=0)
        else:
            path = os.path.join(self.db_path, self.case,
                                "existing_{}_{}.csv".format(tech, self.id))
            if os.path.exists(path):
                self.techs[tech].existing_items = pd.read_csv(
                    path, index_col=0)
            path = os.path.join(self.db_path, self.case,
                                "ts_existing_{}_{}.csv".format(tech, self.id))
            if os.path.exists(path):
                self.techs[tech].ts_existing_items = pd.read_csv(
                    path, index_col=0)

    def to_db(self, tech, group=None):
        """Save data of tech to database.

        Parameters
        ----------
        tech : str
            name of technology
        group : str, optional
            group in the technology e.g. orientation in rooftop-pv,
            by default None
        """
        if tech == "RooftopPV":
            if self.level == "nuts3":
                # If level is nuts3 (Landkreis) we operate rooftop pv
                # per municipality
                for mun, glr_mun in self.RooftopPV.glr_muns.items():
                    glr_mun.to_db("RooftopPV")
            else:
                def _group_items(_group):
                    """[summary] TODO

                    [extended_summary]

                    Parameters
                    ----------
                    _group : str
                        [description]
                    """
                    # set max_size of file to parallize sim
                    max_size = 10000
                    # calculate number of files
                    n = int(
                        len(
                            self.RooftopPV.predicted_items.loc
                            [self.RooftopPV.predicted_items.group == _group]) /
                        max_size)
                    # If modulo != 0 we add a group, to stay within max_size.
                    # If n==0 we want to create an empty file.
                    if (len(self.RooftopPV.predicted_items.loc[self.RooftopPV.predicted_items.group == _group])
                            % max_size != 0 or n == 0):
                        n += 1
                    print("Splitting group {} for {} in {} parts".format(
                        _group, self.name, n), flush=True)
                    size = int(len(
                        self.RooftopPV.predicted_items.loc[self.RooftopPV.predicted_items.group == _group]) / n)
                    for i in range(n):
                        # Last file has extra element if elements are uneven
                        if i + 1 == n:
                            size2 = size + len(
                                self.RooftopPV.predicted_items.loc
                                [self.RooftopPV.predicted_items.group
                                 == _group]) % n
                            self.RooftopPV.predicted_items.loc[self.RooftopPV.predicted_items.group == _group].iloc[i * size:(
                                i + 1) * size2].to_csv(os.path.join(path, "{}_{}_{}_{}.csv".format(tech, "".join(self.id), _group, i)))
                        else:
                            self.RooftopPV.predicted_items.loc[self.RooftopPV.predicted_items.group == _group].iloc[i * size:(
                                i + 1) * size].to_csv(os.path.join(path, "{}_{}_{}_{}.csv".format(tech, "".join(self.id), _group, i)))
                path = os.path.join(self.db_path, self.case,
                                    "RooftopPV_{}".format(self.id))
                if not os.path.exists(path):
                    os.mkdir(path)
                if group is None:
                    if self.RooftopPV.predicted_items is not None:
                        for _group in [
                            'E1', 'S3', 'NW4', 'SE2', 'SW1', 'SE1', 'NW3',
                            'S2', 'N3', 'E3', 'NW1', 'W4', 'SE4', 'N2', 'S1',
                            'E4', 'N4', 'SE3', 'SW2', 'SW4', 'W2', 'S4', 'E2',
                            'NE3', 'N1', 'SW3', 'W3', 'NW2', 'NE4', 'W1',
                                'NE1', 'NE2']:
                            _group_items(_group=_group)
                            if self.RooftopPV.ts_predicted_items is not None:
                                self.RooftopPV.ts_predicted_items[_group].to_csv(
                                    os.path.join(
                                        path,
                                        "ts_{}_{}_{}.csv".format(tech, "".join(self.id), _group)))
                else:
                    if self.RooftopPV.predicted_items is not None:
                        _group_items(_group=group)
                    if self.RooftopPV.ts_predicted_items is not None:
                        self.RooftopPV.ts_predicted_items[group].to_csv(
                            os.path.join(
                                path,
                                "ts_{}_{}_{}.csv".format(tech, "".join(self.id), group)))
        else:
            # TODO Save the csv in the technology directory?
            if self.techs[tech].predicted_items is not None:
                self.techs[tech].predicted_items.to_csv(
                    os.path.join(self.db_path, self.case,
                                 "{}_{}.csv".format(tech, "".join(self.id))))
            if self.techs[tech].ts_predicted_items is not None:
                self.techs[tech].ts_predicted_items.to_csv(
                    os.path.join(
                        self.db_path, self.case,
                        "ts_{}_{}.csv".format(tech, "".join(self.id))))
            if self.techs[tech].report_dict is not None:
                self.techs[tech].save_report(os.path.join(self.techs[tech].result_path, "report.json"))
                self.techs[tech].ec.save(os.path.join(self.techs[tech].result_path, f"{tech}_potential_area.tif"))
                if self.techs[tech].ec._itemCoords is not None and self.techs[tech].ec._itemCoords.shape[0] > 0:
                    self.techs[tech].save_items_to_vector(os.path.join(self.techs[tech].result_path,
                                                                       f"{tech}_potential_items.shp")
                                                          )

    def all_to_db(self):
        """Save all techs to database."""
        for tech in self.techs.keys():
            if self.techs[tech] is not None:
                self.to_db(tech)
        # TODO what is existing_wind_to_db()
        # self.existing_wind_to_db()

    def flush_db(self, tech):
        """Remove existing csv-files for technology from database.

        Parameters
        ----------
        tech : str
            name of technology
        """
        os.remove(
            os.path.join(utils.get_data_path(), "database", self.case,
                         "{}_{}.csv".format(tech, "".join(self.id))))
        os.remove(
            os.path.join(utils.get_data_path(), "database", self.case,
                         "ts_{}_{}.csv".format(tech, "".join(self.id))))

    def check_db(self, tech, db_type="", group=None):
        """Check if technology is in db.

        Parameters
        ----------
        tech : str
            technology
        db_type : str, optional
            time-series or capacity db, either "", or "ts". 
            By default ""
        group : str, optional
            rooftop-pv group, by default None

        Returns
        -------
        pd.DataFrame
            df with db content
        """
        if db_type == "ts":
            db_type += "_"

        if isinstance(tech, Wind):
            check_file = db_type + "Wind_{}.csv".format(self.id)
        elif isinstance(tech, OpenfieldPV):
            check_file = db_type + "OpenfieldPV_{}.csv".format(self.id)
        elif isinstance(tech, OpenfieldPVRoads):
            check_file = db_type + "OpenfieldPVRoads_{}.csv".format(self.id)
        elif isinstance(tech, RooftopPV):
            if group is None:
                check_file = "RooftopPV_{}".format(self.id)
            else:
                check_file = db_type + \
                    "RooftopPV_{}_{}.csv".format(self.id, group)
        # try:
        #     self.download_items("RooftopPV", group)
        # except Exception as e:
        #     print(e)
        if check_file in self.db:
            if isinstance(tech, RooftopPV):
                if group is None:
                    _db_return = []
                    # for _group in ['E1', 'S3', 'NW4', 'SE2', 'SW1', 'SE1',
                    #               'NW3', 'S2', 'N3', 'E3', 'NW1', 'W4',
                    #               'SE4', 'N2', 'S1', 'E4', 'N4', 'SE3',
                    #               'SW2', 'SW4', 'W2', 'S4', 'E2', 'NE3',
                    #               'N1', 'SW3', 'W3', 'NW2', 'NE4', 'W1',
                    #               'NE1', 'NE2']:
                    path = os.path.join(
                        self.db_path, self.case, "RooftopPV_{}".format(
                            self.id))
                    if db_type == "":
                        for file in os.listdir(path):
                            if file[0:9] == "RooftopPV":
                                _db_return.append(pd.read_csv(
                                    os.path.join(path,
                                                 file),
                                    index_col=0))
                        if len(_db_return) > 0:
                            db_return = pd.concat(_db_return)
                        else:
                            db_return = None
                    elif db_type == "ts_":
                        db_return = pd.DataFrame(index=range(0, 8760))
                        for file in os.listdir(path):
                            start = time.time()

                            if file[0:12] == "ts_RooftopPV":
                                temp = pd.read_csv(
                                    os.path.join(path,
                                                 file),
                                    index_col=0)
                                if file[27] in ["1", "2", "3", "4"]:
                                    id_group = (26, 28)
                                else:
                                    id_group = (26, 29)
                                if file[id_group[0]:id_group[1]] not in db_return.columns:
                                    db_return[
                                        file[id_group[0]:id_group[1]]] = temp
                                else:
                                    db_return[
                                        file[id_group[0]:id_group[1]]] = db_return[file[id_group[0]:id_group[1]]].add(
                                        temp["gen"])
                        for gr in [
                            'E1', 'S3', 'NW4', 'SE2', 'SW1', 'SE1', 'NW3',
                            'S2', 'N3', 'E3', 'NW1', 'W4', 'SE4', 'N2', 'S1',
                            'E4', 'N4', 'SE3', 'SW2', 'SW4', 'W2', 'S4', 'E2',
                            'NE3', 'N1', 'SW3', 'W3', 'NW2', 'NE4', 'W1',
                                'NE1', 'NE2']:
                            if gr not in db_return.columns:
                                warn(gr +
                                     " time series not in db for "
                                     + self.id,
                                     UserWarning)
                    # for _group in ['E1', 'S3', 'NW4', 'SE2', 'SW1', 'SE1',
                    #               'NW3', 'S2', 'N3', 'E3', 'NW1', 'W4',
                    #               'SE4', 'N2', 'S1', 'E4', 'N4', 'SE3',
                    #               'SW2', 'SW4', 'W2', 'S4', 'E2', 'NE3',
                    #               'N1', 'SW3', 'W3', 'NW2', 'NE4', 'W1',
                    #               'NE1', 'NE2']:
                    #     try:
                    #       _db_return.append(pd.read_csv(
                    #           os.path.join(utils.get_data_path(),
                    #                        "database",
                    #                        self.case,
                    #                        "RooftopPV_{}".format(self.id),
                    #                        check_file + "_{}.csv".format(_group)),
                    #           index_col=0))
                    #     except FileNotFoundError:
                    #       missing = True
                    # if missing:
                    #   db_return = None
                    # else:
                    #   db_return = pd.concat(_db_return)
                else:
                    db_return = pd.read_csv(
                        os.path.join(self.db_path,
                                     self.case,
                                     "RooftopPV_{}".format(self.id),
                                     check_file),
                        index_col=0)
            else:
                db_return = pd.read_csv(
                    os.path.join(self.db_path,
                                 self.case,
                                 check_file),
                    index_col=0)
            # print(db_type + " {} already in db".format(type(tech)))
        else:
            db_return = None
        return db_return

    def download_items(self, tech, remote_path=None, group=None):
        """Download items of a technology from the database.

        Parameters
        ----------
        tech : str
            name of technology
        remote_path : [type], optional TODO wofür nötig?
            [description], by default None
        group : str, optional
            group in the technology e.g. orientation in rooftop-pv,
            by default None
        """
        if tech == "RooftopPV":
            if group is not None:
                path = os.path.join(
                    utils.get_data_path(),
                    "database", self.case, "RooftopPV_{}".format(self.id),
                    "{}_{}_{}.csv".format(tech, self.id, group))
                if remote_path is None:
                    remote_path = os.path.join(
                        "R:\git\TREP\TREP\data\database",
                        self.case,
                        "RooftopPV_{}".format(self.id),
                        "{}_{}_{}.csv".format(tech, self.id, group))
            else:
                _remote_path = []
                path = os.path.join(
                    utils.get_data_path(),
                    "database", self.case, "RooftopPV_{}".format(self.id))
                for file in os.listdir(
                    os.path.join(
                        remote_path, self.case, "RooftopPV_{}".format(
                            self.id))):
                    if not os.path.exists(
                        os.path.join(
                            path, self.case, "RooftopPV_{}".format(self.id),
                            file)):
                        shutil.copy(
                            os.path.join(
                                remote_path, self.case, "RooftopPV_{}".format(
                                    self.id),
                                file),
                            os.path.join(
                                path, self.case, "RooftopPV_{}".format(
                                    self.id),
                                file))
        else:
            path = os.path.join(
                utils.get_data_path(),
                "database", self.case, "{}_{}.csv".format(tech, self.id))
            remote_path = os.path.join("R:\git\TREP\TREP\data\database",
                                       self.case,
                                       "{}_{}.csv".format(tech, self.id)),
        # if not os.path.exists(path):
        #     shutil.copy(remote_path, path)
        #     print("Copied ", path)
        # else:
        #     print("Didn't download from CAESAR,"
        #           + f" because {} already exists".format(path))

    def download_sim(self, tech):
        path = os.path.join(
            utils.get_data_path(),
            "database", self.case, "ts_{}_{}.csv".format(tech, self.id))
        if not os.path.exists(path):
            shutil.copy(
                os.path.join(
                    "R:\git\TREP\TREP\data\database",
                    self.case,
                    "ts_{}_{}.csv".format(tech, self.id)),
                path)
        else:
            print("Didn't download from CAESAR, because " +
                  "{} already exists".format(path))

    def update_db(self, source="CAESAR", replace=False):
        """Update the db with the data from CAESAR.

        Parameters
        ----------
        source : str, optional
            [description], by default "CAESAR"
        replace : bool, optional
            by default False
        """
        # TODO: ALso from local to caesar. And replace all items.
        source_path = r"R:\data\s-risch\db_TREP\{}".format(self.case)
        destination_path = os.path.join(self.db_path, self.case)
        print(source_path)
        print(self.db_path)
        for file in os.listdir(source_path):
            if not os.path.exists(os.path.join(destination_path, file)):
                print("Downloading file/folder: ", file)
                if os.path.isdir(os.path.join(source_path, file)):
                    shutil.copytree(os.path.join(source_path, file),
                                    os.path.join(destination_path, file))
                else:
                    shutil.copy(os.path.join(source_path, file),
                                os.path.join(destination_path, file))
            else:
                if os.path.isdir(os.path.join(source_path, file)):
                    for _file in os.listdir(os.path.join(source_path, file)):

                        if not os.path.exists(os.path.join(
                                destination_path, file,
                                _file)):
                            print("Downloading file: ", file + "/" + _file)
                            shutil.copy(
                                os.path.join(source_path, file, _file),
                                os.path.join(destination_path, file, _file))

    @ property
    def id(self):
        return self._id

    @ property
    def name(self):
        return self.features["GEN"].values[0]

    @ property
    def state(self):
        map = {'00': None,
               '01': "sh",
               '02': "hh",
               '03': "ni",
               '04': "hb",
               '05': "nw",
               '06': "he",
               '07': "rp",
               '08': "bw",
               '09': "by",
               '10': "sl",
               '11': "be",
               '12': "bb",
               '13': "mv",
               '14': "sn",
               '15': "st",
               '16': "th"}
        return map[self._state]

    @ property
    def area(self):
        """Regions area in km2."""
        area = self.regionMask.mask.sum(dtype=np.int64) * \
            self.regionMask.pixelWidth*self.regionMask.pixelHeight
        area = area / 1000000
        return area

    @ property
    def db(self):
        if self.level == "MUN":
            return os.listdir(
                path=os.path.join(self.db_path, self.case)) + os.listdir(
                path=self.db_path) + os.listdir(
                path=os.path.join(
                    self.db_path, self.case, "RooftopPV_{}".format(self.id)))
        else:
            return os.listdir(path=os.path.join(self.db_path, self.case)) + \
                os.listdir(path=self.db_path)

    @ property
    def RooftopPV(self):
        return self.techs["RooftopPV"]

    @ property
    def Wind(self):
        return self.techs["Wind"]

    @ property
    def OpenfieldPV(self):
        return self.techs["OpenfieldPV"]

    @ property
    def OpenfieldPVRoads(self):
        return self.techs["OpenfieldPVRoads"]
