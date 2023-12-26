import sys
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from cpymad.madx import Madx, TwissFailed

sys.path.append("../")
sys.path.append("../LOCOinPython/")
from LOCOinPython.file_loader.structure_loader import read_structure, collect_structure
from LOCOinPython.numerical_model.imperfections import Imperfections
from LOCOinPython.numerical_model.orbit_correction import OrbitCorrectionSVD
from LOCOinPython.numerical_model.response_matrix import run_madx
from LOCOinPython.numerical_model.lattice_correction import LatticeCorrection


def apply_kicks(madx, kicks_corrs, opposite=False) -> None:
    """
    Apply kicks of correctors.

    :param madx: Madx instance
    :param kicks_corrs: corr_name-kick_value dict
    :param opposite: whether to apply kicks with the opposite sign
    :return: None
    """
    if opposite:
        kicks_corrs["kick"]["Values"] *= -1
    for idx, corr in enumerate(kicks_corrs["kick"]["Knobs"]):
        madx.elements[corr].kick += kicks_corrs["kick"]["Values"][idx]


def get_optics(structure: dict,
               imperfections_file: str = None,
               aligns: dict = None,
               old_aligns: dict = None,
               kicks_corrs: dict = None,
               closing: bool = False,
               elems_for_closing: dict = None,
               save_etable: bool = False,
               file_to_save: str = None,
               file_with_kicks: str = None,
               verbose: bool = False) -> dict:
    """
    Get optical functions, etc.

    :param structure: obtained from read_structure func
    :param imperfections_file: file name with imperfections in the madx format
    :param aligns: imperfections
    :param old_aligns: preexisted imperfections
    :param kicks_corrs: corr_name-kick_value dict
    :param closing: whether to perform closing
    :param elems_for_closing: dict with elems to use as knobs in closing
    :param save_etable: whether to save error table in the standard madx format
    :param file_to_save: a file name to save an error table to
    :param file_with_kicks: a file name with corrector kicks obtained from the madx orbit correction command
    :param verbose: whether to print debugging info to a console
    :return: dict with optical functions, orbits, etc.
    """
    madx = Madx(stdout=verbose)
    if not verbose:
        madx.input("option, echo=false, warn=false, info=false, twiss_print=false;")
    else:
        madx.input("debug=true;")
    collect_structure(structure, madx)
    if imperfections_file:
        madx.input(f"readtable, file={imperfections_file}, table=tabl;")
        madx.input("seterr, table=tabl;")
    else:
        Imperfections.add_to_model(madx, aligns)
        Imperfections.add_to_model(madx, old_aligns)

    if file_with_kicks:
        madx.input(f"call, file={file_with_kicks};")

    if kicks_corrs:
        for corr_type, corr_list in kicks_corrs.items():
            for corr, kick_val in corr_list.items():
                madx.elements[corr].kick += kick_val

    if closing:
        kick_amplitude = 1e-3  # Adjust me
        grad_amplitude = 1e-2  # Adjust me
        verbosing_freq = 100  # Adjust me
        iteration = 0
        closed = False

        while not closed:
            kicks = (np.random.random(len(elems_for_closing["kick"]["Knobs"])) - 0.5) * kick_amplitude
            elems_for_closing["kick"]["Values"] = kicks
            k1 = (np.random.random(len(elems_for_closing["k1l"]["Knobs"])) - 0.5) * grad_amplitude
            elems_for_closing["k1l"]["Values"] = k1
            try:
                apply_kicks(madx, elems_for_closing)
                Imperfections.add_to_model(madx, elems_for_closing)

                madx.select('FLAG = Twiss', 'class = monitor')
                madx.twiss(table='twiss', centre=True)
                madx.input('select, flag = twiss, clear;')
                closed = True
            except TwissFailed:
                apply_kicks(madx, elems_for_closing, opposite=True)
                elems_for_closing["k1l"]["Values"] *= -1
                Imperfections.add_to_model(madx, elems_for_closing)
                iteration += 1
                if iteration % verbosing_freq == 0:
                    print(iteration)

    try:
        madx.select('FLAG = Twiss', 'class = monitor')
        madx.twiss(table='twiss', centre=True)
        madx.input('select, flag = twiss, clear;')
        res = {"x": madx.table.twiss.selection().x,
               "y": madx.table.twiss.selection().y,
               "betx": madx.table.twiss.selection().betx,
               "bety": madx.table.twiss.selection().bety,
               "dx": madx.table.twiss.selection().dx,
               "dy": madx.table.twiss.selection().dy,
               "s": madx.table.twiss.selection().s,
               "name": [name.split(":")[0] for name in madx.table.twiss.selection().name]}
    except TwissFailed:
        print("Twiss Failed!")
        res = None

    if save_etable:
        madx.input("select, flag=error, full;")
        madx.input(f"esave, file = {file_to_save};")

    madx.quit()
    del madx

    return res


def get_ptc_optics(structure: dict,
                   imperfections_file: str = None,
                   aligns: dict = None,
                   old_aligns: dict = None,
                   kicks_corrs: dict = None,
                   closing: bool = False,
                   elems_for_closing: dict = None,
                   save_etable: bool = False,
                   file_to_save: str = None,
                   file_with_kicks: str = None,
                   verbose: bool = False) -> dict:
    """
    Get optical functions, etc via the PTC environment.

    :param structure: obtained from read_structure func
    :param imperfections_file: file name with imperfections in the madx format
    :param aligns: imperfections
    :param old_aligns: preexisted imperfections
    :param kicks_corrs: corr_name-kick_value dict
    :param closing: whether to perform closing
    :param elems_for_closing: dict with elems to use as knobs in closing
    :param save_etable: whether to save error table in the standard madx format
    :param file_to_save: a file name to save an error table to
    :param file_with_kicks: a file name with corrector kicks obtained from the madx orbit correction command
    :param verbose: whether to print debugging info to a console
    :return: dict with optical functions, orbits, etc.
    """
    madx = Madx(stdout=verbose)
    if not verbose:
        madx.input("option, echo=false, warn=false, info=false, twiss_print=false;")
    else:
        madx.input("debug=true;")
    collect_structure(structure, madx)
    if imperfections_file:
        madx.input(f"readtable, file={imperfections_file}, table=tabl;")
        madx.input("seterr, table=tabl;")
    else:
        Imperfections.add_to_model(madx, aligns)
        Imperfections.add_to_model(madx, old_aligns)

    if file_with_kicks:
        madx.input(f"call, file={file_with_kicks};")

    if kicks_corrs:
        for corr_type, corr_list in kicks_corrs.items():
            for corr, kick_val in corr_list.items():
                madx.elements[corr].kick += kick_val

    if closing:
        kick_amplitude = 1e-3  # Adjust me
        grad_amplitude = 1e-2  # Adjust me
        verbosing_freq = 100  # Adjust me
        iteration = 0
        closed = False

        while not closed:
            kicks = (np.random.random(len(elems_for_closing["kick"]["Knobs"])) - 0.5) * kick_amplitude
            elems_for_closing["kick"]["Values"] = kicks
            k1 = (np.random.random(len(elems_for_closing["k1l"]["Knobs"])) - 0.5) * grad_amplitude
            elems_for_closing["k1l"]["Values"] = k1
            try:
                apply_kicks(madx, elems_for_closing)
                Imperfections.add_to_model(madx, elems_for_closing)

                madx.select('FLAG = Twiss', 'class = monitor')
                madx.twiss(table='twiss', centre=True)
                madx.input('select, flag = twiss, clear;')
                closed = True
            except TwissFailed:
                apply_kicks(madx, elems_for_closing, opposite=True)
                elems_for_closing["k1l"]["Values"] *= -1
                Imperfections.add_to_model(madx, elems_for_closing)
                iteration += 1
                if iteration % verbosing_freq == 0:
                    print(iteration)

    try:
        madx.input("ptc_create_universe; ptc_create_layout, time = True, model = 2, method = 6, nst = 3; ptc_align;")
        madx.select('FLAG = twiss', 'class = monitor')
        madx.ptc_twiss(icase=6, closed_orbit=True, center_magnets=True, file="ptc_file", table="twiss")
        madx.input('select, flag = twiss, clear;')
        res = {"x": madx.table.twiss.selection().x[::2],
               "y": madx.table.twiss.selection().y[::2],
               "betx": madx.table.twiss.selection().betx[::2],
               "bety": madx.table.twiss.selection().bety[::2],
               "dx": madx.table.twiss.selection().dx[::2],
               "dy": madx.table.twiss.selection().dy[::2],
               "s": madx.table.twiss.selection().s[::2],
               "name": [name.split(":")[0] for name in madx.table.twiss.selection().name[::2]]}
    except TwissFailed:
        print("Twiss Failed!")
        res = None

    if save_etable:
        madx.input("select, flag=error, full;")
        madx.input(f"esave, file = {file_to_save};")

    madx.quit()
    del madx

    return res


def read_tracking_file(file: str, unlost_only: bool) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Read a tracking file, which is created after the ptc_track command.
    This file contains information about all particles and their coordinates at every pre-defined locations.

    :param file: file name created by Madx ptc_track
    :param unlost_only: whether to retrieve unlost particles only
    :return list[DataFrame]: list of DataFrames. list size = number of locations. DataFrame size = [number of particles, coordinates]
    """
    with open(file, 'r') as f:
        table = f.readlines()
    list_df = []
    list_df_per_turn = []

    for n, i in enumerate(tqdm(table)):
        # Drop the first 7 unneccessary lines
        if n > 7:
            i_ = i.split()
            if i_[0] == '#segment' or i_[0] == '':
                if list_df_per_turn:
                    df = pd.DataFrame(list_df_per_turn)
                    df = df.rename(columns={0: "number", 1: "turn", 2: "x", 3: "px", 4: "y", 5: "py", 6: "t", 7: "pt", 8: "s", 9: "E"})
                    list_df.append(df)
                    list_df_per_turn = []
                    if unlost_only and len(list_df) == 1:
                        break
                continue
            list_df_per_turn.append([float(k.replace(",", ".")) for k in i_])
            # df=pd.concat([df,pd.DataFrame(np.array([[float(k) for k in i.split()]]))],ignore_index=True)

    if n == len(table) - 1:
        df = df.rename(columns={0: "number", 1: "turn", 2: "x", 3: "px", 4: "y", 5: "py", 6: "t", 7: "pt", 8: "s", 9: "E"})
        list_df.append(df)

    if unlost_only:
        # Starting from the end to get the last turn skipping intermediate ones
        list_df_per_turn = []
        for n, i in enumerate(tqdm(reversed(table))):
            i_ = i.split()
            if i_[0] == '#segment' or i_[0] == '':
                df = pd.DataFrame(list_df_per_turn)
                df = df.rename(columns={0: "number", 1: "turn", 2: "x", 3: "px", 4: "y", 5: "py", 6: "t", 7: "pt", 8: "s", 9: "E"})
                df = df.iloc[::-1].reset_index(drop=True)
                list_df.append(df)
                list_df_per_turn = []
                break
            list_df_per_turn.append([float(k.replace(",", ".")) for k in i_])
        nums = list_df[-1]["number"].tolist()
        list_df = list_df[0][list_df[0]["number"].isin(nums)]

    return list_df


def plot_optics(data: dict, params_to_show: str, title: str):
    """
    Plot optical functions, orbits, etc.

    :param data: dict with values for betas, orbits, location, etc.
    :param params_to_show: what params to show
    :param title: plot name
    :return: None
    """
    if params_to_show == "beta":
        plt.plot(data["s"], data["betx"], label='betx')
        plt.plot(data["s"], data["bety"], label='bety')
        plt.xlabel("s [m]")
        plt.ylabel("Beta function [m]")
        plt.title(title)
        plt.legend()
    elif params_to_show == "orbit":
        plt.plot(data["s"], data["x"], label='x')
        plt.plot(data["s"], data["y"], label='y')
        plt.xlabel("s [m]")
        plt.ylabel("Orbit [m]")
        plt.title(title)
        plt.legend()


def plot_dynap(data: dict, dependency_to_show: dict, title: str, plot_name: str):
    """
    Plot dynamical aperture.

    :param data: dict with values for particles and their 6D coordinates
    :param dependency_to_show: what dynap dependency to show
    :param title: overall plot name
    :param plot_name: plot name for a dependency
    :return: None
    """
    x, y = list(dependency_to_show.items())
    x, x_units = x[0], x[1]
    y, y_units = y[0], y[1]
    plt.scatter(data[x], data[y], 3, label=plot_name)
    plt.xlabel(x + f" [{x_units}]")
    plt.ylabel(y + f" [{y_units}]")
    plt.title(title)
    plt.legend()


def create_err_table(err_types: List[str], elems_with_errs: List[str], seed: int = 42) -> dict:
    """
    Create a table with errors normally distributed.

    :param err_types: error types passed
    :param elems_with_errs: elems to add errors to
    :param seed: seed
    :return: a table with error types and values introduced to elements
    """
    ERR_TYPES = {"dx": 80e-6, "dy": 80e-6, "ds": 80e-6, "dpsi": 200e-6, "dphi": 200e-6, "dtheta": 200e-6}
    np.random.seed(seed)

    align_errs = {}

    for err in err_types:
        err_type_to_add = {err: {"Knobs": {}}}
        for elem in elems_with_errs:
            err_type_to_add[err]["Knobs"][elem] = {"Elements": [elem]}
        err_type_to_add[err]["Values"] = (np.random.normal(scale=ERR_TYPES[err], size=len(elems_with_errs))).tolist()
        align_errs.update(err_type_to_add)

    return align_errs


def match_optics(structure: dict,
                 imperfections_file: str = None,
                 aligns: dict = None,
                 old_aligns: dict = None,
                 target_vars: List[str] = None,
                 target_optical_funcs: dict = None,
                 elem_and_params_to_match: dict = None,
                 param_steps: dict = None,
                 file_with_kicks: str = None,
                 verbose: bool = False) -> dict:
    """
    Get optical functions, etc.

    :param structure: obtained from read_structure func
    :param imperfections_file: file name with imperfections in the madx format
    :param aligns: imperfections
    :param old_aligns: preexisted imperfections
    :param target_vars: variables to be used in minimization
    :param target_optical_funcs: desired target funcs for minimization
    :param elem_and_params_to_match: elements and parameters to vary
    :param param_steps: steps for param variations
    :param file_with_kicks: a file name with corrector kicks obtained from the madx orbit correction command
    :param verbose: whether to print debugging info to a console
    :return: dict with optical functions, orbits, etc.
    """
    madx = Madx(stdout=verbose)
    if not verbose:
        madx.input("option, echo=false, warn=false, info=false, twiss_print=false;")
    else:
        madx.input("debug=true;")
    collect_structure(structure, madx)
    if imperfections_file:
        madx.input(f"readtable, file={imperfections_file}, table=tabl;")
        madx.input("seterr, table=tabl;")
    else:
        Imperfections.add_to_model(madx, aligns)
        Imperfections.add_to_model(madx, old_aligns)

    if file_with_kicks:
        madx.input(f"call, file={file_with_kicks};")

    madx.input(f'match, sequence = {structure["sequence_div"]["name"]};')
    for idx in range(len(target_optical_funcs["betx"])):
        bpm = target_optical_funcs["name"][idx]
        goals = [f"{var}={target_optical_funcs[var][idx]}" for var in target_vars]
        goals = ", ".join(goals)
        madx.input(f"constraint, sequence={structure['sequence_div']['name']}, range={bpm}, {goals};")

    for elem, param in elem_and_params_to_match.items():
        madx.input(f"vary, name = {elem}->{param}, step = {param_steps[param]};")

    madx.input('lmdif, calls=3000, tolerance=1e-16;')
    madx.input('simplex, calls=1000, tolerance=1e-15;')
    madx.input('migrad, calls=10000, tolerance=1e-15, strategy=3;')
    madx.input('jacobian, calls=10000, tolerance=1e-15, repeat=3;')
    madx.input('endmatch;')

    for elem, param in elem_and_params_to_match.items():
        madx.input(f'value, {elem}->{param};')

    try:
        madx.select('FLAG = Twiss', 'class = monitor')
        madx.twiss(table='twiss', centre=True)
        madx.input('select, flag = twiss, clear;')
        res = {"x": madx.table.twiss.selection().x,
               "y": madx.table.twiss.selection().y,
               "betx": madx.table.twiss.selection().betx,
               "bety": madx.table.twiss.selection().bety,
               "dx": madx.table.twiss.selection().dx,
               "dy": madx.table.twiss.selection().dy,
               "s": madx.table.twiss.selection().s,
               "name": [name.split(":")[0] for name in madx.table.twiss.selection().name]}
    except TwissFailed:
        print("Twiss Failed!")
        res = None

    madx.quit()
    del madx

    return res


def correct_orbit(structure: dict,
                  imperfections_file: str = None,
                  aligns: dict = None,
                  old_aligns: dict = None,
                  plane: str = "x",
                  ncorrs: int = 0,
                  algorithm: str = "micado",
                  corrs_to_use: dict = None,
                  sngval: int = 2,
                  sngcut: int = 50,
                  verbose: bool = False) -> None:
    """
    Get optical functions, etc.

    :param structure: obtained from read_structure func
    :param imperfections_file: file name with imperfections in the madx format
    :param aligns: imperfections
    :param old_aligns: preexisted imperfections
    :param plane: coordinate for orbit correction
    :param ncorrs: number of correctors to use
    :param algorithm: method to solve an inverse problem
    :param corrs_to_use: desired correctors to be used in orbit correction
    :param sngval: threshold for singular values and redundant correctors
    :param sngcut: threshold for redundant correctors
    :param verbose: whether to print debugging info to a console
    :return: dict with optical functions, orbits, etc.
    """
    madx = Madx(stdout=verbose)
    if not verbose:
        madx.input("option, echo=false, warn=false, info=false, twiss_print=false;")
    else:
        madx.input("debug=true;")
    collect_structure(structure, madx)
    if imperfections_file:
        madx.input(f"readtable, file={imperfections_file}, table=tabl;")
        madx.input("seterr, table=tabl;")
    else:
        Imperfections.add_to_model(madx, aligns)
        Imperfections.add_to_model(madx, old_aligns)

    try:
        madx.select('FLAG = Twiss', 'class = monitor')
        madx.twiss(table='twiss', centre=True)
        madx.input('select, flag = twiss, clear;')

        if corrs_to_use:
            if plane == "x":
                corr_type = "hkicker"
            elif plane == "y":
                corr_type = "vkicker"
            for corr in structure["kick_total"][corr_type]:
                madx.input(f"usekick, status=off, sequence={structure['sequence_div']['name']}, pattern={corr};")
            for corr in corrs_to_use[corr_type]:
                madx.input(f"usekick, status=on, sequence={structure['sequence_div']['name']}, pattern={corr};")

        madx.input(f"correct, sequence={structure['sequence_div']['name']}, mode={algorithm}, plane={plane}, ncorr={ncorrs}, sngval={sngval}, sngcut={sngcut}, orbit=twiss, CLIST = corr.out, MLIST = mon.out, resout=1, error=1e-8;")
    except TwissFailed:
        print("Twiss Failed!")

    madx.quit()
    del madx
