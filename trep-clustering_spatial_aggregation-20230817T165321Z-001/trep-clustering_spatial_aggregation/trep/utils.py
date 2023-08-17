import os
from sqlalchemy import create_engine
import pandas as pd
import statsmodels.formula.api as smf
import geokit as gk
import numpy as np


def get_data_path():
    """Return the data path."""
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")


def get_datasources_path():
    """Return datasources-path."""
    return os.path.join(get_data_path(), "datasources")


def get_osm_path(state):
    """Return the osm path.

    Parameters
    ----------
    state : str
        number of federal state ("1" to "16")

    Returns
    -------
    Path
        osm path of the federal state
    """
    _map_state_osm = {
        "01": "schleswig-holstein",
        "02": "hamburg",
        "03": "niedersachsen",
        "04": "bremen",
        "05": "nordrhein-westfalen",
        "06": "hessen",
        "07": "rheinland-pfalz",
        "08": "baden-wuerttemberg",
        "09": "bayern",
        "10": "saarland",
        "11": "brandenburg",  # Berlin (11) is integrated in brandenburg
        "12": "brandenburg",
        "13": "mecklenburg-vorpommern",
        "14": "sachsen",
        "15": "sachsen-anhalt",
        "16": "thueringen"
    }
    return os.path.join("osm", "unzipped",
                        _map_state_osm[state])


def df_to_sqlite(df, name, sqlite_path, **kwargs):
    """Write dataframe to sqlite database.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        name of sql table
    sqlite_path : Path
        path to sqlite database
    """
    engine = create_engine(f'sqlite:///{sqlite_path}?charset=utf8mb4')
    dbconnection = engine.connect()
    df.to_sql(name=name, con=engine, **kwargs)
    dbconnection.close()

def rename_columns(df_data, type="wind"):
    """Renames the columns of mastr df to naming conventions in our analyses

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame with mastr-data
    type : str, optional
        type of items, by default "wind"

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
    """
    df_data = df_data.rename(columns={"ENH_Nettonennleistung": "capacity",
                                      "ENH_Nabenhoehe": "hub_height",
                                      "ENH_Rotordurchmesser": "rotor_diam",
                                      "ENH_Breitengrad": "lat",
                                      "ENH_Laengengrad": "lon"})
    return df_data

def fill_rotor_diameter(df_data, datasource_path, how="root"):
    """Fills missing rotor diameter by capacity for mastr data

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame with mastr-data
    datasource_path : str
        Path that indicate the datasources
    how : str, optional
        how the diameter is filled (for now just root), by default "root"

    Returns
    -------
    pd.DataFrame
        [description]
    """
    query = "SELECT ENH_MastrID, ENH_Nettonennleistung," + \
            "ENH_Plz, ENH_InbetriebnahmeDatum, " + \
            "ENH_Rotordurchmesser, ENH_Nabenhoehe, " + \
            "ENH_Breitengrad, ENH_Laengengrad, ENH_Seelage " + \
            "FROM processed WHERE " + \
            "ENH_EinheitenTyp='Windeinheit' and ENH_Betriebsstatus='In Betrieb'"
    # raw_wts = db_query(query)
    engine = create_engine("sqlite:///" + os.path.join(datasource_path, "mastr", "mastr.db")
                           + "/?charset=utf8mb4")
    raw_wts = pd.read_sql(sql=query, con=engine)
        # raw_wts = db_query("SELECT ENH_MastrID, ENH_Nettonennleistung, ENH_Rotordurchmesser, ENH_Nabenhoehe, ENH_Breitengrad, ENH_Laengengrad, ENH_Seelage FROM processed WHERE ENH_EinheitenTyp='Windeinheit'")
    filtered_wts = raw_wts[raw_wts["ENH_Nettonennleistung"] < 10*1e3]
    filtered_wts = filtered_wts[filtered_wts["ENH_Rotordurchmesser"] < 500]
    if how == "root":
        # Polynom (2nd degree) regression
        df = pd.DataFrame()
        df["d"] = filtered_wts["ENH_Rotordurchmesser"]
        df["P"] = filtered_wts["ENH_Nettonennleistung"]
        results = smf.ols(formula='d ~ I(P**0.5)', data=df).fit()
        print(results.summary())
        df_pred = pd.DataFrame()
        all_wts = df_data.copy()
        df_pred["P"] = df_data[df_data["ENH_Rotordurchmesser"].isna()]["ENH_Nettonennleistung"]
        print("Predicting {} of {} diameters".format(len(df_pred), len(df_data)))
        diam = results.predict(df_pred)
        df_data["ENH_Rotordurchmesser"] = df_data["ENH_Rotordurchmesser"].fillna(diam)
        
    return df_data


def get_nuts3_mapping2010(datasource_path, reverse=False):
    """Map nuts3 to names

    Parameters
    ----------
    datasource_path : str
        Path that indicate the datasources
    reverse : bool, optional
        reverses the mapping to {Code: name} if true, by default False

    Returns
    -------
    dict
        dict with mapping
    """
    db_path = os.path.join(
        datasource_path,
        "other",
        "NUTS2010-NUTS2013.xls"
    )
    map_nuts3 = {}
    nuts_xls = pd.read_excel(db_path, sheet_name=1, header=[1])
    nuts_xls = nuts_xls[nuts_xls["Code 2010"].isna()==False]
    for idx, row in nuts_xls.iterrows():
        if row["Code 2010"][0:2]=="DE" and len(row["Code 2010"])==5:
            if reverse:
                map_nuts3[row["Code 2010"]] = row["NUTS level 3"]
            else:
                map_nuts3[row["NUTS level 3"]] = row["Code 2010"]
    return map_nuts3

def get_mapping_RS_nuts_2010(datasource_path):
    """Map nuts3 to RS
    Parameters
    ----------
    datasource_path : str
        Path that indicate the datasources
    Returns
    -------
    dict
        dict with mapping {RS: nuts3}
    """
    path = os.path.join(
        datasource_path,
        "other",
        "VG250_1Jan2011_UTM32",
        "VG250_Kreise.shp")
    name_to_nuts = get_nuts3_mapping2010(datasource_path)
    features = gk.vector.extractFeatures(path)
    rs_nuts = {}
    for idx, row in features.iterrows():
        name = ""
        # There are whitespaces in the excel and sometimes not
        # In general the data quality is bad, therefore so many if clauses
        for i, char in enumerate(row["GEN"]):
            name += char
            if len(name) < len(row["GEN"]):
                if char == "." and row["GEN"][i+1] != " ":
                        name += " "
        names = [row["GEN"], name]
        is_inside = False
        for _name in names:
            if _name + ", " + row["DES"] in name_to_nuts.keys():
                rs_nuts[row["RS"]] = name_to_nuts[_name + ", " + row["DES"]]
                is_inside = True
            elif _name + "," + row["DES"] in name_to_nuts.keys():
                rs_nuts[row["RS"]] = name_to_nuts[_name + "," + row["DES"]]
                is_inside = True
            elif "Region "+ _name in name_to_nuts.keys():
                rs_nuts[row["RS"]] = name_to_nuts["Region " + _name]
                is_inside = True
            elif "Regionalverband "+ _name in name_to_nuts.keys():
                rs_nuts[row["RS"]] = name_to_nuts["Regionalverband " + _name]
                is_inside = True
            elif "Städteregion " + _name in name_to_nuts.keys():
                rs_nuts[row["RS"]] = name_to_nuts["Städteregion " + _name]
                is_inside = True
            elif _name + " (DE)" in name_to_nuts.keys():
                rs_nuts[row["RS"]] = name_to_nuts[_name + " (DE)"]
                is_inside = True
            # Dont know why there is a (Oldb)
            elif _name == "Oldenburg (Oldb)":
                rs_nuts[row["RS"]] = name_to_nuts["Oldenburg (Oldenburg), Kreisfreie Stadt"]
                is_inside = True
            # Got renamed to Heidekreis
            elif _name == "Soltau-Fallingbostel":
                rs_nuts[row["RS"]] = name_to_nuts["Heidekreis"]
                is_inside = True
            # ...
            elif _name == "Burgenlandkreis":
                rs_nuts[row["RS"]] = name_to_nuts["Burgenland (DE)"]
                is_inside = True
            # Point in the end
            elif _name == "Weiden i.d. OPf.":
                rs_nuts[row["RS"]] = name_to_nuts["Weiden i. d. Opf, Kreisfreie Stadt"]
                is_inside = True
            # Check if name by itself is in excel
            elif _name in name_to_nuts.keys():
                rs_nuts[row["RS"]] = name_to_nuts[_name]
                is_inside = True
        if not is_inside:
            print(row["GEN"])
    return rs_nuts


def line_to_area(source, output, where_text=None, key=None, default_width=None, max_width=100):
    """Transform line features to area features (polygon) using a width. Either use the width from the source or use
    the default_width given by the user.

    Please note that, this function will not change the SRS of the features. It is recommended to transform the SRS of
    the data source to a SRS using meter as it unit, if it is not.

    If multiple sources are given, all parameters must also be given in the same size.

        Parameters
        ----------
        source : path, array like
            The path of the source
        output : path, array like
            The path to save the result
        where_text : str, array like
            The where text to locate the right features
        key : str, array like
            The key (attribute name) to get the width from data source. If not given will use the default width for all
            features
        default_width : float/int, array like
            The default width of the features. This parameter must be given
        max_width : float/int, array like
            Limit the maximum width of the features, to avoid unrealistic information, by default 100.
    """
    # Check if multi inputs are given
    multi = np.array(source).size > 1

    if multi:
        source = np.array(source)
        output = np.array(output)
        where_text = np.array(where_text)
        key = np.array(key)
        default_width = np.array(default_width)
        max_width = np.array(max_width)

        num_multi = source.size
        if not (output.size == num_multi and where_text.size == num_multi and key.size == num_multi and default_width.size == num_multi
                and max_width.size == num_multi):
            raise ValueError("The size of inputs do not coincide with each other!")
        for n in range(num_multi):
            print(f"Start computing {output[n]}", flush=True)
            features = gk.vector.extractFeatures(source[n], where=where_text[n])
            features_width = features.copy()

            num_no_data = 0
            # buffer the line features with its width
            for i in range(len(features)):
                if not key[n]:
                    width = default_width[n]
                    num_no_data += 1
                # TODO better way to detect no data?
                elif not isinstance(features[key[n]][i], float) or not 0 < features[key[n]][i] < max_width[n]:
                    width = default_width[n]
                    num_no_data += 1
                else:
                    width = features[key[n]][i]
                geom = features.at[i, "geom"]
                geom_width = geom.Buffer(width / 2)
                # features_width["geom"][i] = geom_width
                features_width.at[i, "geom"] = geom_width

            print(f"Use default value for {num_no_data} features among {len(features)} features.", flush=True)

            gk.vector.createVector(features_width, output=output[n])

    else:
        features = gk.vector.extractFeatures(source, where=where_text)
        features_width = features.copy()

        num_no_data = 0
        # buffer the line features with its width
        for i in range(len(features)):
            if not key:
                width = default_width
                num_no_data += 1
            # TODO better way to detect no data?
            elif not isinstance(features[key][i], float) or not 0 < features[key][i] < max_width:
                width = default_width
                num_no_data += 1
            else:
                width = features[key][i]
            # geom = features["geom"][i]
            geom = features.at[i, "geom"]
            geom_width = geom.Buffer(width / 2)
            # features_width["geom"][i] = geom_width
            features_width.at[i, "geom"] = geom_width

        print(f"Use default value for {num_no_data} features among {len(features)} features.", flush=True)

        gk.vector.createVector(features_width, output=output)
