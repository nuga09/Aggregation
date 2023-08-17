from trep.technology import Technology
import os
from trep import utils
import pandas as pd
import geokit as gk
from geoalchemy2 import Geometry  # <= not used but must be imported
from sqlalchemy import create_engine, MetaData, select, func, and_, or_
import osgeo
import numpy as np
import reskit as rk
import scipy
import time
import trep
import warnings
import json


class RooftopPV(Technology):
    def __init__(self, parent):
        super(RooftopPV, self).__init__(parent=parent)
        self.time = time.time()
        self.existing_items = None
        self.ts_existing_items = None
        # Save municipalities glrs so data handling is only happening once
        self.glr_muns = {}
        # Check the path in database
        # TODO a separate directory only for municipality?
        self.result_path = os.path.join(self.parent.case_path, f"RooftopPV_{self.parent._id}")
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

    def get_population(self):
        """Get population of region based on Zensus2011 data.

        Returns
        -------
        int
            population of region
        """
        path = os.path.join(
            self.parent.datasource_path,
            "other", "Zensus2011_Bevoelkerung",
            "Zensus11_Datensatz_Bevoelkerung.csv")
        df_zensus = pd.read_csv(
            path, sep=";", encoding="utf-8",
            dtype={"AGS_12": str, "RS_Land": str, "RS_RB_NUTS2": str,
                   "RS_Kreis": str, "RS_VB": str, "RS_Gem": str})
        df_zensus["RS"] = df_zensus["RS_Land"] + df_zensus["RS_RB_NUTS2"] + \
            df_zensus["RS_Kreis"] + df_zensus["RS_VB"] + df_zensus["RS_Gem"]
        map_regions = {
            "Mönchengladbach": "Mönchengladbach, Stadt",
            "Ilmenau": "Ilmenau, Stadt",
            "Suhl": "Suhl, Stadt"
        }
        population = 0
        _region = map_regions[self.parent.name] \
            if self.parent.name in map_regions.keys() \
            else self.parent.name
        if self.parent.level == "nuts3":
            df_zensus = df_zensus[df_zensus.Reg_Hier ==
                                  "Stadtkreis/kreisfreie Stadt/Landkreis"]
        elif self.parent.level == "MUN":
            df_zensus = df_zensus[df_zensus.Reg_Hier == "Gemeinde"]
        if self.parent.level == "nuts3":
            print(_region)
            population += \
                df_zensus[df_zensus.Name == _region]["AEWZ"].values[0]
        elif self.parent.level == "MUN":
            population += \
                df_zensus[df_zensus.RS == self.parent.id]["AEWZ"].values[0]
        print(population)
        return population

    def estimate_roof_pv_potential(self, efficiency=0.2214):
        """Estimate the roof pv potential based on Ryberg.

        A_pv = 172.3*population*populationDensity^(-0.352)
        P_pv = A_Pv*efficiency*0.5

        Parameters
        ----------
        efficiency : float, optional
            efficiency used for estimation, by default 0.2214

        Returns
        -------
        float
            P_pv total PV capacity potential in region in MW
        """
        population = self.get_population()
        print("pop", population)
        print("ar", self.parent.area)
        pv_area = 172.3*population*(population/(self.parent.area))**(-0.352)
        print("pvarea", pv_area)
        # 3.33m^2 module/kWp (30% efficiency) and
        # 50% utilization factor portion of roofs that are north facing
        P_pv = pv_area*efficiency*0.5/1e3  # in MWp

        return P_pv

    def get_roof_pv_items_ryberg(self, P_pv, elevation=300, resolution=1):
        """Get the rooftop PV items.

        Distributes the items based on distributions for azimuth (uni[90,270])
        and tilt angle (normal[mu=35, std=15])

        Parameters
        ----------
        P_pv : float
            total PV capacity potential in region
        elevation : float, optional
            elevation in region, by default 300
        resolution : int, optional
            resolution of distributions in degrees. Trade off between accuracy
            and computational burden. I.e. a lot of (approx 16200) have to be
            simulated with 1 degree resolution. By default 1
        """
        azis = np.array([i for i in range(90, 271, resolution)])
        tilts = np.array([i for i in range(0, 91, resolution)])
        tilt_dist = scipy.stats.norm(35, 15)
        # tilts = [tilt_dist.cdf(i) for i in range(0, 90)]
        modules = pd.DataFrame(
            columns=["capacity", "lon", "lat", "tilt", "azimuth",
                     "elev", "prob"])
        for i, azi in enumerate(azis):
            for j, tilt in enumerate(tilts):
                modules.loc[j+i*len(tilts), "azimuth"] = azi
                modules.loc[j+i*len(tilts), "tilt"] = tilt
                modules.loc[j+i*len(tilts), "prob"] = \
                    (tilt_dist.cdf(tilt+resolution/2) -
                     tilt_dist.cdf(tilt-resolution/2)) * 1/len(azis)
        # calculate module capacity:
        # Probability of azi and tilt * total pv capacity and conversion to kWp
        modules["capacity"] = modules["prob"]*P_pv*1e3
        x = (self.ec.region.extent.xMin + self.ec.region.extent.xMax)/2
        y = (self.ec.region.extent.yMin + self.ec.region.extent.yMax)/2
        xy = gk.srs.xyTransform(
            (x, y),
            fromSRS=self.parent.regionMask.srs, toSRS=4326)
        modules["lon"] = xy[0][0]
        modules["lat"] = xy[0][1]
        modules["elev"] = elevation
        # modules["tilts"] = tilts
        # modules["prob"] = modules["azis"]*modules["tilts"]
        self.predicted_items = modules

    def get_roofs(self):
        """Get roofs from 3D-citydb and determine the basic properties.

        Basic properties of roof: Azi, Tilt, Area
        Fills self.predicted_items with DataFrame of roof items.
        """

        def query_roofs(bdg_ids, bdg_fct):
            """Query, which gets Roofs belonging to the buildings represented
            by the bdg_ids.

            Parameters
            ----------
            bdg_ids : list
                list with building ids

            Returns
            -------
            pd.DataFrame
                roof items
            """
            # Select corresponding thematic surface which are roofs
            ts_table = meta.tables["thematic_surface"]
            ts = select([ts_table.c.lod2_multi_surface_id, ts_table.c.building_id], 
                        and_(ts_table.c.building_id.in_(bdg_ids),
                        (ts_table.c.objectclass_id == 33)))
            map_bdg_fct = {}
            for i in bdg_fct.itertuples():
                map_bdg_fct[i[1]] = i[2]
            ts_info = pd.read_sql_query(ts, conn)
            ts_bdg = {}
            start = time.time()
            for i in ts_info.itertuples():
                ts_bdg[i[1]] = i[2]
            print("took", time.time()-start)
            # This is not working, but would be more efficient. ts_ids are equal in both ways?!
            # ts_ids = tuple(ts_info.lod2_multi_surface_id.values)
            ts_ids = conn.execute(ts).fetchall()
            ts_ids = list(zip(*ts_ids))[0]
            # Select corresponding roof surface geometries
            surfaces = meta.tables["surface_geometry"]
            # Select geometry and SRID
            rs = select([func.ST_AsText(surfaces.c.geometry),
                         func.ST_AsText(surfaces.c.solid_geometry),
                         surfaces.c.geometry,
                         func.ST_GeomFromText(
                             func.ST_AsText(surfaces.c.geometry),
                             func.ST_SRID(surfaces.c.geometry)),
                         func.ST_GeomFromWKB(surfaces.c.geometry),
                         func.ST_SRID(surfaces.c.geometry),
                         surfaces.c.parent_id],
                         surfaces.c.parent_id.in_(ts_ids),
                        )
            # rs = select([func.ST_AsText(func.ST_DumpPoints(surfaces.c.geometry).geom)], 
            #           surfaces.c.parent_id.in_(ts_ids),
            #           )
            df = pd.read_sql_query(rs, conn)
            start = time.time()
            df["function"] = df.apply(lambda x: (map_bdg_fct[ts_bdg[x["parent_id"]]]), axis=1)
            print("took", time.time()-start)

            return df

        # TODO: this should all be with the single state dbs,
        # when they are fully imported.
        db_config_path = os.path.join(
            utils.get_data_path(),
            "3DCityDB_config", "3DCityDB_config.json")
        if not os.path.exists(db_config_path):
            raise PermissionError("Config file not found. Therefore " +
                                  "couldn't establish db connection.")
        with open(db_config_path) as f:
            db_config_file = json.load(f)
        # Get relevant databases
        fs = gk.vector.extractFeatures(os.path.join(utils.get_datasources_path(),
                                    "germany_administrative",
                                    "vg250_ebenen",
                                    "VG250_LAN.shp"))
        states = []
        engines = []
        _extent = gk.geom.transform(self.parent.regionMask.extent._box, toSRS=25832)
        for idx, row in fs.iterrows():
            if _extent.Intersects(row["geom"]):
                states.append(row["RS"])
        print("Federal States", states, flush=True)
        states = set(states)
        for i, state in enumerate(states):
            db_info = db_config_file[state]
            engine = create_engine(
                f'postgresql+psycopg2://{db_info["user"]}:{db_info["password"]}@{db_info["ip"]}:{db_info["port"]}'
                + f'/{db_info["db_name"]}')
            meta = MetaData()
            meta.reflect(bind=engine)
            conn = engine.connect()
            # Create ewkt of extent to query buildings and roofs
            if state in ["11", "12", "13", "14"]:
                srid = 25833
            else:
                srid = 25832
            region_geom = gk.geom.transform(self.parent.regionMask.geometry, toSRS=srid)
            pts = self.parent.regionMask.extent.xXyY
            pts = [(pts[0], pts[2]), (pts[0], pts[3]), (pts[1], pts[3]), (pts[1], pts[2])]
            pts = gk.srs.xyTransform(pts, fromSRS=self.parent.regionMask.srs, toSRS=srid)
            pts = [(pt[0], pt[1]) for pt in pts]
            pol = gk.geom.polygon(pts)
            wkt = pol.ExportToWkt()
            ewkt = f'SRID={srid};' + wkt
            cityobject = meta.tables["cityobject"]

            # Select bdg_ids which are within the considered region
            select_bdg_ids = select([cityobject.c.id],
                                    and_(func.ST_Within(cityobject.c.envelope, (func.ST_GeomFromEWKT(ewkt))),
                                        or_(cityobject.c.objectclass_id == 25, cityobject.c.objectclass_id == 26)))
            bdg_ids = conn.execute(select_bdg_ids).fetchall()
            print("got buildings")
            print(len(bdg_ids), flush=True)
            if len(bdg_ids) > 0:
                bdg_ids = list(zip(*bdg_ids))[0]
                bdgs_meta = meta.tables["building"]
                select_bdg_ids_fct = select([bdgs_meta.c.id, bdgs_meta.c.function],
                                            bdgs_meta.c.id.in_(bdg_ids))
                bdg_fct = pd.read_sql_query(select_bdg_ids_fct, conn)
                _df = query_roofs(bdg_ids, bdg_fct)
                _df["Within"] = _df.apply(lambda x: osgeo.ogr.Geometry.Within(gk.geom.convertWKT(x["ST_AsText_1"],
                                                                                srs=int(x["ST_SRID_1"])),
                                                                region_geom),
                            axis=1)
                _df = _df[_df["Within"]]
                if i == 0:
                    df = _df
                else:
                    df = pd.concat([df, _df])
            else:
                # No buildings in municipalities --> empty dataframe
                print("No buildings in mun", flush=True)
                bdg_ids = []
                if i == 0:
                    df = pd.DataFrame()
        print("Read DB {}".format((time.time()-self.time)/60), flush=True)

        def unit_vector(vector):
            """Return the unit vector of the vector."""
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """Return the angle in degree between vectors 'v1' and 'v2'."""
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return 180/np.pi*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        def get_normal(geom):
            """Return the normal vector on a geometry."""
            pts = np.array(list(set(geom.GetGeometryRef(0).GetPoints())))
            normal = np.cross(pts[1]-pts[0], pts[2]-pts[0])
            normal = unit_vector(normal)
            if normal[2] < 0:
                normal *= -1

            return normal

        def calc_azi(normal):
            """Calculate the azimuth from the normal vector."""
            vec1 = normal[0:2]
            vec2 = np.array([0, 1])
            azi = angle_between(vec1, vec2)
            # if west: 360 - angle
            if normal[0] < 0 and normal[1] != 0:
                azi = 360 - azi
            else:
                pass

            return azi

        def calc_tilt(normal):
            """Calculate the tilt from the normal vector."""
            vec1 = normal
            vec2 = np.array([0, 0, 1])
            tilt = angle_between(vec1, vec2)
            # tilt = np.arccos(normal[2])*180/np.pi

            return tilt

        def get_loc(df):
            """Return the center of geometries in the df. 

            Parameters
            ----------
            df : pd.DataFrame
                df with geometries in "geom" column 

            Returns
            -------
            pd.DataFrame
                input df with added "center" column of geometries
            """
            df["center"] = df.apply(
                lambda x:
                    [(gk.geom.extractVerticies(x["geom"].Centroid())[0][0],
                      gk.geom.extractVerticies(x["geom"].Centroid())[0][1])],
                axis=1)
            df["corrupt"] = df.apply(
                lambda x: x["center"][0][0] < 0 or x["center"][0][1] < 0,
                axis=1)
            _corrupt = df[df["corrupt"]]
            if len(_corrupt) > 0:
                if _corrupt.area.sum() < 1:
                    warnings.warn(
                        "Element with negative center, " +
                        f"dropping {_corrupt.area.sum()}m2")
                    df = df.drop(_corrupt.index)
                else:
                    raise Exception(
                        f"Elements with {_corrupt.area.sum()}m2 " +
                        f"were corrupt, please check {self.parent.id}")
            # Dropping elements with area smaller than 1e-5m2
            # because of problems with the location
            df = df[df.area > 1e-5]
            df["lon"] = df.apply(
                lambda x: gk.srs.xyTransform(
                    x["center"],
                    fromSRS=int(x["ST_SRID_1"]), toSRS=4326)[0][0],
                axis=1)
            df["lat"] = df.apply(
                lambda x: gk.srs.xyTransform(
                    x["center"],
                    fromSRS=int(x["ST_SRID_1"]), toSRS=4326)[0][1],
                axis=1)

            return df

        def calc_all(df):
            """Execute all needed functions."""
            df["geom"] = df.apply(
                lambda x: gk.geom.convertWKT(
                    x["ST_AsText_1"],
                    srs=int(x["ST_SRID_1"])),
                axis=1)
            df["normal"] = df.apply(lambda x: get_normal(x["geom"]), axis=1)
            df["area"] = df.apply(lambda x: x["geom"].Area(), axis=1)
            df["tilt"] = df.apply(lambda x: calc_tilt(x["normal"]), axis=1)
            df["azimuth"] = df.apply(lambda x: calc_azi(x["normal"]), axis=1)
            df["group"] = None
            return df

        if len(df) > 0:
            self.predicted_items = calc_all(df)
            self.predicted_items = get_loc(self.predicted_items)
        else:
            print("No roofs in mun", flush=True)
            self.predicted_items = pd.DataFrame(
                columns=['area', 'tilt', 'azimuth', 'lon', 'lat', 'capacity',
                            'flat', 'function', 'group'])

    def get_roof_pv_items(self, minCapacity=1):
        """Get roof PV items after getting roofs in area."""
        if self.predicted_items is None:
            self.get_roofs()
            print("Got roofs {}".format(
                (time.time() - self.time) / 60), flush=True)
        if len(self.predicted_items) > 0:
            tilted = self.predicted_items[self.predicted_items["tilt"] > 10]
            tilted = tilted.assign(capacity=tilted["area"]*0.6*0.2214)
            tilted = tilted.assign(flat=False)
            flat = self.predicted_items[self.predicted_items["tilt"] <= 10]
            if "LANUV" in self.parent.case:
                flat = flat.assign(capacity=flat["area"] * 0.6 * 0.2214 * 0.4)
                flat = flat.assign(tilt=32)
            else:
                flat = flat.assign(capacity=flat["area"] * 0.6 * 0.2214 * 0.5)
                # TODO: no loop.
                # flat = flat.assign(tilt=lambda x: rk.solar.location_to_tilt(gk.Location(lon=x["lon"], lat=x["lat"]))) # This one fails
                start = time.time()
                for idx, row in flat.iterrows():
                    flat.loc[idx, "tilt"] = rk.solar.location_to_tilt(
                        [gk.Location(lon=row["lon"], lat=row["lat"])])
                print("Loop took", (time.time()-start)/60)

            flat = flat.assign(azimuth=180)
            flat = flat.assign(flat=True)
            self.predicted_items = pd.concat([tilted, flat])
            self.predicted_items = self.predicted_items[self.predicted_items.capacity > minCapacity]
            print("Total number of roofs: ", len(self.predicted_items))
            print("Got items {}".format((time.time()-self.time)/60),
                  flush=True)

    def group_items(self):
        """Group items according to MaStr.

        8 Groups for azimuth (i.e. North, North-East ...). 4 Groups for tilt.
        """
        azi = self.predicted_items.azimuth
        self.predicted_items.loc[(0 <= azi) & (
            azi <= 22.5), "group"] = "N"  # N=0°
        self.predicted_items.loc[(22.5 < azi) & (
            azi <= 67.5), "group"] = "NE"  # NE=45°
        self.predicted_items.loc[(67.5 < azi) & (
            azi <= 112.5), "group"] = "E"  # E=90°
        self.predicted_items.loc[(112.5 < azi) & (
            azi <= 157.5), "group"] = "SE"  # SE=135°
        self.predicted_items.loc[(157.5 < azi) & (
            azi <= 202.5), "group"] = "S"  # S=180°
        self.predicted_items.loc[(202.5 < azi) & (
            azi <= 247.5), "group"] = "SW"  # SW=225°
        self.predicted_items.loc[(247.5 < azi) & (
            azi <= 292.5), "group"] = "W"  # W=270°
        self.predicted_items.loc[(292.5 < azi) & (
            azi <= 337.5), "group"] = "NW"  # NW# =315°
        self.predicted_items.loc[(337.5 < azi) & (
            azi <= 360), "group"] = "N"  # N=360°

        tilt = self.predicted_items.tilt
        self.predicted_items.loc[(tilt < 20), "group"] += "1"  # tilt < 20°
        self.predicted_items.loc[(20 <= tilt) & (
            tilt < 40), "group"] += "2"  # tilt 20 - 40°
        self.predicted_items.loc[(40 <= tilt) & (
            tilt < 60), "group"] += "3"  # tilt 40-60 °
        self.predicted_items.loc[(tilt >= 60), "group"] += "4"  # tilt >= 60°

    def get_existing_plants(self):
        """Get existing plants from MaStR in groups."""
        def _get_existing(mastr_existing, ags):
            mastr_existing_mun = \
                mastr_existing[mastr_existing.ENH_Gemeindeschluessel == ags]
            mastr_existing_mun = \
                mastr_existing_mun[mastr_existing_mun.ENH_Lage != "Freifläche"]
            # print("Not using {} kW, because no group is given".format(
            #     mastr_existing_mun[mastr_existing_mun.group.isna()].ENH_Nettonennleistung.sum()))
            mastr_existing_mun = \
                mastr_existing_mun[mastr_existing_mun.group.notna()]
            groups = set(mastr_existing_mun.group.values)
            existing = pd.DataFrame(
                index=['E1', 'S3', 'NW4', 'SE2', 'SW1', 'SE1', 'NW3', 'S2',
                       'N3', 'E3', 'NW1', 'W4', 'SE4', 'N2', 'S1', 'E4', 'N4',
                       'SE3', 'SW2', 'SW4', 'W2', 'S4', 'E2', 'NE3', 'N1',
                       'SW3', 'W3', 'NW2', 'NE4', 'W1', 'NE1', 'NE2'],
                data=0,
                columns=["capacity", "lat", "lon", "azimuth", "tilt", "share"])

            if self.predicted_items is None:
                self.estimate_potential()
                self.group_items()
            map_azi = {"N": 0, "NE": 45, "E": 90, "SE": 135,
                       "S": 180, "SW": 225, "W": 270, "NW": 315}
            # TODO: Check values from distribution
            map_tilt = {"1": 10, "2": 30, "3": 50, "4": 60}
            for group in groups:
                # East-West azimuth --> Flat Roof --> S2
                if group in ["EW", "EW1", "EW2", "EW3", "EW4"]:
                    existing.loc["S2", "capacity"] += mastr_existing_mun[
                        mastr_existing_mun.group == group].ENH_Nettonennleistung.sum()
                elif group[-1:] not in ["1", "2", "3", "4"]:
                    print("Not using {} kW, because no tilt is given".format(
                        mastr_existing_mun[mastr_existing_mun.group == group].ENH_Nettonennleistung.sum()))
                else:
                    existing.loc[group, "capacity"] += \
                        mastr_existing_mun[mastr_existing_mun.group ==
                                           group].ENH_Nettonennleistung.sum()
                    existing.loc[group, "azimuth"] = map_azi[group[:-1]]
                    existing.loc[group, "tilt"] = map_tilt[group[-1:]]
                    existing.loc[group, "share"] = \
                        existing.loc[group, "capacity"] / \
                        self.predicted_items[
                            self.predicted_items.group == group].capacity.sum()
                [(existing["lon"], existing["lat"], z)] = \
                    gk.srs.xyTransform((
                        (self.parent.regionMask.extent.xXyY[0] +
                         self.parent.regionMask.extent.xXyY[1]) / 2,
                        (self.parent.regionMask.extent.xXyY[2] +
                         self.parent.regionMask.extent.xXyY[3]) / 2),
                    fromSRS=self.parent.regionMask.srs, toSRS="latlon")
            return existing
        if self.existing_items is None:
            mastr_existing = pd.read_csv(
                os.path.join(
                    self.parent.datasource_path,
                    "mastr", "pv_groups.csv"),
                dtype={"ENH_Gemeindeschluessel": str},
                index_col=0)
            if self.parent.level == "nuts3":
                for i, mun in enumerate(self.parent.municipalities):
                    glr_mun = trep.TREP(
                        mun, level="MUN", case=self.parent.case,
                        db_path=self.parent.db_path)
                    if i == 0:
                        self.existing_items = _get_existing(
                            mastr_existing, glr_mun._ags)
                    else:
                        self.existing_items["capacity"] = \
                            self.existing_items["capacity"].add(
                                _get_existing(mastr_existing, glr_mun._ags)["capacity"])
                    if self.predicted_items is None:
                        self.estimate_potential()
                        self.group_items()
                    for group in self.existing_items.index:
                        self.existing_items.loc[group, "share"] = \
                            self.existing_items.loc[group, "capacity"] / \
                            self.predicted_items[
                                self.predicted_items.group == group].capacity.sum()
            else:
                self.existing_items = _get_existing(
                    mastr_existing, self.parent._ags)

    def calc_existing(self):
        """Calc share of used space on roofs."""
        if self.existing_items is None:
            self.get_existing_plants()
        for group in set(self.predicted_items.group.values):
            if group in self.existing_items.index:
                share = self.existing_items.loc[group, "share"]
            else:
                share = 0
            self.predicted_items.loc[
                self.predicted_items.group == group, "capacity"] = \
                self.predicted_items.loc[self.predicted_items.group ==
                                         group, "capacity"] * (1 - share)

    def estimate_potential(self, deduct_existing=False):
        """Estimate potential of RooftopPV.

        Parameters
        ----------
        deduct_existing : bool, optional
            If the existing plants shall be deducted from the potential.
            By default False
        """
        if self.predicted_items is None:
            if self.parent.level == "nuts3":
                items = []
                for i, mun in enumerate(self.parent.municipalities):
                    print("{}/{} municipalities".format(i+1,
                          len(self.parent.municipalities)), flush=True)
                    print(mun, flush=True)
                    if mun not in self.glr_muns.keys():
                        self.glr_muns[mun] = trep.TREP(
                            mun, case=self.parent.case, level="MUN",
                            db_path=self.parent.db_path)
                    self.glr_muns[mun].RooftopPV.predicted_items = \
                        self.glr_muns[mun].check_db(
                        self.glr_muns[mun].techs["RooftopPV"])
                    if deduct_existing:
                        self.calc_existing()
                    if self.glr_muns[mun].RooftopPV.predicted_items is None:
                        self.glr_muns[mun].RooftopPV.get_roof_pv_items()
                        self.glr_muns[mun].RooftopPV.group_items()
                    items.append(self.glr_muns[mun].RooftopPV.predicted_items)
                self.predicted_items = pd.concat(items)
            else:
                self.predicted_items = self.parent.check_db(self)
                if self.predicted_items is None:
                    self.get_roof_pv_items()
                    if len(self.predicted_items) > 0:
                        self.group_items()
                    if deduct_existing:
                        self.calc_existing()
                    self.predicted_items = self.predicted_items[['area', 'tilt', 'azimuth', 'lon', 'lat', 'capacity', 'flat', 'function', 'group']]
                else:
                    print("RooftopPV in db. " +
                          "Flush db if you want to re-evaluate.")

    def sim(self, group=None):
        """Simulate the RooftopPV items.

        Parameters
        ----------
        group : str, optional
            group of the RooftopPV items e.g. 'E1', by default None
        """
        # TODO this method is not working properly
        if group is not None:
            if self.parent.level == "nuts3":
                for i, mun in enumerate(self.parent.municipalities):
                    if mun not in self.glr_muns.keys():
                        self.glr_muns[mun] = \
                            trep.TREP(mun, level="MUN", case=self.parent.case,
                                      db_path=self.parent.db_path)
                    self.glr_muns[mun].add_tech("RooftopPV")
                    temp = self.glr_muns[mun].check_db(
                        self.glr_muns[mun].techs["RooftopPV"], "ts", group)
                    if temp is not None:
                        if temp.isna().any.any():
                            print(self.glr_muns[mun].id,
                                  self.glr_muns[mun].name, temp)
                        if i == 0:
                            self.ts_predicted_items[group] = temp
                        else:
                            self.ts_predicted_items[group] = \
                                self.ts_predicted_items.add(temp)
                    else:
                        print("{}/{} municipalities".format(i,
                              len(self.parent.municipalities)), flush=True)
                        self.glr_muns[mun].techs["RooftopPV"].estimate_potential()
                        print("Start simulating {}", flush=True)
                        (self.glr_muns[mun].techs["RooftopPV"].ts_predicted_items,
                         self.glr_muns[mun].techs["RooftopPV"].predicted_items) = \
                            self.glr_muns[mun].techs["RooftopPV"].sim()
                        print("Done simulating {}", flush=True)
                        if i == 0:
                            self.ts_predicted_items[group] = \
                                self.glr_muns[mun].techs["RooftopPV"].ts_predicted_items
                            self.predicted_items = \
                                self.glr_muns[mun].techs["RooftopPV"].predicted_items
                        else:
                            self.ts_predicted_items[group] = self.ts_predicted_items.add(
                                self.glr_muns[mun].techs["RooftopPV"].ts_predicted_items)
                            self.predicted_items = pd.concat(
                                [self.predicted_items, self.glr_muns[mun].techs["RooftopPV"].predicted_items])
            else:
                if self.ts_predicted_items is None:
                    self.ts_predicted_items = pd.DataFrame()
                self.ts_predicted_items[group] = self.parent.check_db(
                    self, "ts", group=group)
                if len(self.ts_predicted_items[group]) == 0:
                    if self.predicted_items is None:
                        self.estimate_potential()
                    if isinstance(self.predicted_items, pd.DataFrame):
                        self.group_items()
                    print("Start simulating {}".format(
                        time.time() - self.time), flush=True)
                    # TODO: make single groups working with db
                    if group in self.predicted_items.group:
                        (self.ts_predicted_items[group],
                            self.predicted_items[group]) = \
                            self.sim_pv(
                                self.predicted_items[group], poa_bound=0)
                    else:
                        self.ts_predicted_items[group] = 8760*[0]
                    self.parent.to_db("RooftopPV", group)
                    print("Done simulating {}".format(
                        time.time() - self.time), flush=True)
        else:
            print("Group needed to simulate. Trying to read DB.", flush=True)
            if self.parent.level == "nuts3":
                ts = []
                for i, mun in enumerate(self.parent.municipalities):
                    print("{}/{} municipalities".format(i + 1,
                          len(self.parent.municipalities)), flush=True)
                    print(mun, flush=True)
                    if mun not in self.glr_muns.keys():
                        self.glr_muns[mun] = trep.TREP(
                            mun, case=self.parent.case, level="MUN",
                            db_path=self.parent.db_path)
                    self.glr_muns[mun].RooftopPV.sim()
                    if i == 0:
                        ts = self.glr_muns[mun].RooftopPV.ts_predicted_items
                    else:
                        ts = ts.add(
                            self.glr_muns[mun].RooftopPV.ts_predicted_items)
                self.ts_predicted_items = ts
            else:
                if self.predicted_items is None:
                    self.predicted_items = self.parent.check_db(self)
                self.ts_predicted_items = self.parent.check_db(
                    self, db_type="ts")

    def sim_existing(self):
        """Sim the existing RooftopPV items."""
        if self.parent.level == "nuts3":
            for i, mun in enumerate(self.parent.municipalities):
                if mun not in self.glr_muns.keys():
                    self.glr_muns[mun] = trep.TREP(
                        mun, level="MUN", case=self.parent.case,
                        db_path=self.parent.db_path)
                self.glr_muns[mun].RooftopPV.sim_existing()
                if i == 0:
                    self.ts_existing_items = \
                        self.glr_muns[mun].RooftopPV.ts_existing_items
                else:
                    self.ts_existing_items = self.ts_existing_items.add(
                        self.glr_muns[mun].RooftopPV.ts_existing_items)
        else:
            self.parent.check_existing_db("RooftopPV")
            if self.existing_items is None:
                self.get_existing_plants()
            if self.ts_existing_items is None:
                if all(self.existing_items.capacity == 0):
                    self.ts_existing_items = pd.DataFrame(index=range(0, 8760))
                else:
                    self.ts_existing_items, self.existing_items = self.sim_pv(
                        placements=self.existing_items,
                        merge=False,
                        poa_bound=0)
        # TODO in sim
        for group in [
            'E1', 'S3', 'NW4', 'SE2', 'SW1', 'SE1', 'NW3', 'S2', 'N3', 'E3',
            'NW1', 'W4', 'SE4', 'N2', 'S1', 'E4', 'N4', 'SE3', 'SW2', 'SW4',
            'W2', 'S4', 'E2', 'NE3', 'N1', 'SW3', 'W3', 'NW2', 'NE4', 'W1',
                'NE1', 'NE2']:
            if group not in self.ts_existing_items.columns:
                self.ts_existing_items[group] = 0
        self.ts_existing_items = self.ts_existing_items.fillna(0)
        # for group in [
        #     'E1', 'S3', 'NW4', 'SE2', 'SW1', 'SE1', 'NW3', 'S2', 'N3', 'E3',
        #     'NW1', 'W4', 'SE4', 'N2', 'S1', 'E4', 'N4', 'SE3', 'SW2', 'SW4',
        #     'W2', 'S4', 'E2', 'NE3', 'N1', 'SW3', 'W3', 'NW2', 'NE4', 'W1',
        #         'NE1', 'NE2']:
        #     print(pd.DataFrame(self.existing_items.loc[group]).T, flush=True)
        #     self.ts_existing_items[group], test = \
        #         self.sim_pv(placements=pd.DataFrame(self.existing_items.loc[group]).T)
        #     print(test, flush=True)
