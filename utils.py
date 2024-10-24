import sys
from typing import List, Union, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from cpymad.madx import Madx, TwissFailed

sys.path.append("../")
sys.path.append("../LOCOinPython/")
from LOCOinPython.src.numerical_model.imperfections import Imperfections
from LOCOinPython.src.file_loader.structure_loader import collect_structure


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
               initial_twiss: dict = None,
               imperfections_file: str = None,
               aligns: dict = None,
               old_aligns: dict = None,
               kicks_corrs: dict = None,
               closing: bool = False,
               elems_for_closing: dict = None,
               save_etable: bool = False,
               file_to_save: str = None,
               file_with_kicks: str = None,
               file_with_matching_results: str = None,
               saveseq: bool = False,
               seq_file: str = None,
               verbose: bool = False,
               radiate: bool = False) -> dict:
    """
    Get optical functions, etc.

    :param structure: obtained from read_structure func
    :param initial_twiss: initial values for optical function for twiss calculation
    :param imperfections_file: file name with imperfections in the madx format
    :param aligns: imperfections
    :param old_aligns: preexisted imperfections
    :param kicks_corrs: corr_name-kick_value dict
    :param closing: whether to perform closing
    :param elems_for_closing: dict with elems to use as knobs in closing
    :param save_etable: whether to save error table in the standard madx format
    :param file_to_save: a file name to save an error table to
    :param file_with_kicks: a file name with corrector kicks obtained from the madx orbit correction command
    :param file_with_matching_results: a file name with matching results
    :param saveseq: whether to save a sequence to a file
    :param seq_file: a file name with to save a sequence
    :param verbose: whether to print debugging info to a console
    :param radiate: whether to turn on radiation
    :return: dict with optical functions, orbits, etc.
    """
    madx = Madx(stdout=verbose)
    collect_structure(structure, madx, verbose=verbose, radiate=radiate)

    if file_with_matching_results:
        madx.input(f"call, file={file_with_matching_results};")
        madx.input(f"use, sequence={structure['sequence_div']['name']};")

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
                if "_x" in corr:
                    madx.elements[corr[:-2]].hkick += kick_val
                elif "_z" in corr:
                    madx.elements[corr[:-2]].vkick += kick_val
                else:
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
        if initial_twiss is None:
            initial_twiss = {}
        madx.select('FLAG = Twiss', 'class = monitor')
        madx.twiss(table='twiss', centre=True, **initial_twiss)
        madx.input('select, flag = twiss, clear;')

        madx.input("emit; ex = beam->ex; ey = beam->ey;")

        res = {"x": madx.table.twiss.selection().x,
               "y": madx.table.twiss.selection().y,
               "betx": madx.table.twiss.selection().betx,
               "bety": madx.table.twiss.selection().bety,
               "dx": madx.table.twiss.selection().dx,
               "dy": madx.table.twiss.selection().dy,
               "s": madx.table.twiss.selection().s,
               "name": [name.split(":")[0] for name in madx.table.twiss.selection().name],
               "qx": madx.table.summ.q1[0],
               "qy": madx.table.summ.q2[0],
               "betx_all": madx.table.twiss.betx,
               "bety_all": madx.table.twiss.bety,
               "dx_all": madx.table.twiss.dx,
               "dy_all": madx.table.twiss.dy,
               "s_all": madx.table.twiss.s,
               "x_all": madx.table.twiss.x,
               "y_all": madx.table.twiss.y,
               "name_all": madx.table.twiss.name,
               "ex": madx.globals["ex"],
               "ey": madx.globals["ey"],
               "coupling": 0 if madx.globals["ey"] == 1 and madx.globals["ex"] == 1 else madx.globals["ey"] / madx.globals["ex"]}

    except TwissFailed:
        print("Twiss Failed!")
        res = None

    if save_etable:
        madx.input("select, flag=error;")
        madx.input(f"esave, file = {file_to_save};")

    if saveseq:
        madx.input(f"save, sequence={structure['sequence_div']['name']}, file={seq_file}, noexpr=true, csave=true;")

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
                   verbose: bool = False,
                   radiate: bool = False) -> dict:
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
    :param radiate: whether to turn on radiation
    :return: dict with optical functions, orbits, etc.
    """
    madx = Madx(stdout=verbose)
    collect_structure(structure, madx, verbose=verbose, radiate=radiate)

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
        madx.input("select, flag=error;")
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

    if not len(list_df):
        raise ValueError("All particles are lost! Tracking data is empty")

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
    elif params_to_show == "beta_all":
        plt.plot(data["s_all"], data["betx_all"], label='betx')
        plt.plot(data["s_all"], data["bety_all"], label='bety')
        plt.xlabel("s [m]")
        plt.ylabel("Beta function [m]")
        plt.title(title)
        plt.legend()
    elif params_to_show == "disp":
        plt.plot(data["s"], data["dx"], label='dx')
        plt.plot(data["s"], data["dy"], label='dy')
        plt.xlabel("s [m]")
        plt.ylabel("Dispersion [m]")
        plt.title(title)
        plt.legend()
    elif params_to_show == "disp_all":
        plt.plot(data["s_all"], data["dx_all"], label='dx')
        plt.plot(data["s_all"], data["dy_all"], label='dy')
        plt.xlabel("s [m]")
        plt.ylabel("Dispersion [m]")
        plt.title(title)
        plt.legend()
    elif params_to_show == "orbit_all":
        plt.plot(data["s_all"], data["x_all"], label='x')
        plt.plot(data["s_all"], data["y_all"], label='y')
        plt.xlabel("s [m]")
        plt.ylabel("Orbit [m]")
        plt.title(title)
        plt.legend()
    else:
        raise ValueError(f"Unknown params to show: {params_to_show}")


def plot_dynap(data: dict, dependency_to_show: dict, title: str, plot_name: str, x_limits: list = None, y_limits: list = None):
    """
    Plot dynamic aperture.

    :param data: dict with values for particles and their 6D coordinates
    :param dependency_to_show: what dynap dependency to show
    :param title: overall plot name
    :param plot_name: plot name for a dependency
    :return: None
    """
    if x_limits is None:
        x_limits = []
    if y_limits is None:
        y_limits = []
    x, y = list(dependency_to_show.items())
    x, x_units = x[0], x[1]
    y, y_units = y[0], y[1]
    plt.scatter(data[x], data[y], 3, label=plot_name)
    plt.xlabel(x + f" [{x_units}]")
    plt.ylabel(y + f" [{y_units}]")
    plt.title(title)
    plt.xlim(*x_limits)
    plt.ylim(*y_limits)
    plt.legend()


def create_err_table(errors_and_sigmas: Union[List[str], Dict[str, float]], elems_with_errs: List[str], seed: int = 42) -> dict:
    """
    Create a table with errors normally distributed.

    :param errors_and_sigmas: error type and sigma for randomizing
    :param elems_with_errs: elems to add errors to
    :param seed: seed
    :return: a table with error types and values introduced to elements
    """
    if isinstance(errors_and_sigmas, list):
        tmp = {}
        for err in errors_and_sigmas:
            if err in ["dx", "dy", "ds"]:
                tmp[err] = 80e-6
            elif err in ["dphi", "dpsi", "dtheta"]:
                tmp[err] = 200e-6
            else:
                raise ValueError(f"Unknown error type: {err}")
        errors_and_sigmas = tmp
    np.random.seed(seed)

    align_errs = {}

    for err, sigma in errors_and_sigmas.items():
        err_type_to_add = {err: {"Knobs": {}}}
        for elem in elems_with_errs:
            err_type_to_add[err]["Knobs"][elem] = {"Elements": [elem]}
        err_type_to_add[err]["Values"] = (np.random.normal(scale=sigma, size=len(elems_with_errs))).tolist()
        align_errs.update(err_type_to_add)

    return align_errs


def _parse_multipole(definition: str) -> Tuple[str, List[str]]:
    """
    Parse multipole definition as defined in MAD-X file.
    It is needed to create a custom multipole and to vary its fields during matching.

    :param definition: string representation of multipole definition
    :return:
            string representation of multipole definition without initial field values
            list of initial field values in the string format
    """
    if "knl" in definition or "ksl" in definition:
        definition = definition.split(",")[0]  # Remove knl= or ksl=
    else:
        definition = definition.split(";")[0]
    values = ["0.0", "0.0"]
    return definition, values


def _make_knob_for_matching(madx: Madx, elem: str, param: str, structure: dict) -> str:
    """
    Make a knob for variation in the MAD-X matching. Knobs for multipoles are created and processed in a special way.

    :param madx: Madx instance
    :param elem: element name
    :param param: parameter name in terms of k1, k1s, kick, etc.
    :param structure: a structure object obtained from the read_structure func
    :return: string representation of the knob for MAD-X input
    """
    if structure["elements"][elem]["type"] == "multipole":
        multipole_definition, values = _parse_multipole(str(madx.elements[elem]))
        if param == "k1":
            p = elem + "_k1"
            madx.input(f"{p}={values[1]};")
            madx.input(f"{multipole_definition}, knl:={{{values[0]},{p}}};")
        elif param == "k1s":
            p = elem + "_k1s"
            madx.input(f"{p}={values[1]};")
            madx.input(f"{multipole_definition}, ksl:={{{values[0]},{p}}};")
        else:
            raise ValueError(f"Check params for variation: {param}")
    else:
        p = f"{elem}->{param}"

    return p


def match_optics(structure: dict,
                 imperfections_file: str = None,
                 aligns: dict = None,
                 old_aligns: dict = None,
                 target_vars_and_weights: Dict[str, float] = None,
                 target_optical_funcs: dict = None,
                 elem_and_params_to_match: dict = None,
                 param_steps: dict = None,
                 algorithms: Union[str, List[str]] = "lmdif",
                 file_with_kicks: str = None,
                 file_with_matching_results: str = None,
                 save_matching: bool = False,
                 file_to_save: str = None,
                 verbose: bool = False,
                 radiate: bool = False) -> Tuple[dict, str]:
    """
    Get optical functions, etc.

    :param structure: obtained from read_structure func
    :param imperfections_file: file name with imperfections in the madx format
    :param aligns: imperfections
    :param old_aligns: preexisted imperfections
    :param target_vars_and_weights: variables and weights to use in minimization
    :param target_optical_funcs: desired target funcs for minimization
    :param elem_and_params_to_match: elements and parameters to vary
    :param param_steps: steps for param variations
    :param algorithms: algorithms to use during optimization
    :param file_with_kicks: a file name with corrector kicks obtained from the madx orbit correction command
    :param file_with_matching_results: a file name with matching results
    :param verbose: whether to print debugging info to a console
    :param save_matching: whether to the results of matching
    :param file_to_save: the output file to save the results of matching
    :param radiate: whether to turn on radiation
    :return:
            dict with optical functions, orbits, etc.
            string representation of element definitions after matching
    """
    madx = Madx(stdout=verbose)
    collect_structure(structure, madx, verbose=verbose, radiate=radiate)

    if file_with_matching_results:
        madx.input(f"call, file={file_with_matching_results};")
        madx.input(f"use, sequence={structure['sequence_div']['name']};")

    knobs_for_matching = []
    for elem, param in elem_and_params_to_match.items():
        knob = _make_knob_for_matching(madx, elem, param, structure)
        knobs_for_matching.append((knob, param))
    madx.input(f"use, sequence={structure['sequence_div']['name']};")

    if imperfections_file:
        madx.input(f"readtable, file={imperfections_file}, table=tabl;")
        madx.input("seterr, table=tabl;")
    else:
        Imperfections.add_to_model(madx, aligns)
        Imperfections.add_to_model(madx, old_aligns)

    if file_with_kicks:
        madx.input(f"call, file={file_with_kicks};")

    if "coupling" in target_vars_and_weights or "ex" in target_vars_and_weights or "ey" in target_vars_and_weights:
        # Use macro to optimize emittances and coupling
        madx.input(f'match, use_macro, sequence = {structure["sequence_div"]["name"]};')
        macro = "mac: macro={twiss, centre=True;"
        constraint = ""
        if "coupling" in target_vars_and_weights:
            macro += "emit; MVAR1:=beam->ey/beam->ex;"
            constraint += f"constraint, weight={target_vars_and_weights['coupling']}, expr=MVAR1={target_optical_funcs['coupling']};"
        if "ex" in target_vars_and_weights:
            macro += "MVAR2:=beam->ex;"
            constraint += f"constraint, weight={target_vars_and_weights['ex']}, expr=MVAR2={target_optical_funcs['ex']};"
        if "ey" in target_vars_and_weights:
            macro += "MVAR3:=beam->ey;"
            constraint += f"constraint, weight={target_vars_and_weights['ey']}, expr=MVAR3={target_optical_funcs['ey']};"
        madx.input(macro + "};" + constraint)

        for idx in range(len(target_optical_funcs["betx"])):
            bpm = target_optical_funcs["name"][idx]
            for var, weight in target_vars_and_weights.items():
                if var not in ["qx", "qy", "coupling", "ex", "ey"]:
                    madx.input(f"constraint, weight={weight}, expr=table(twiss,{bpm},{var})={target_optical_funcs[var][idx]};")
        if "qx" in target_vars_and_weights:
            madx.input(f"constraint, weight={target_vars_and_weights['qx']}, expr=table(summ,q1)={target_optical_funcs['qx']};")
        if "qy" in target_vars_and_weights:
            madx.input(f"constraint, weight={target_vars_and_weights['qy']}, expr=table(summ,q2)={target_optical_funcs['qy']};")
    else:
        madx.input(f'match, sequence = {structure["sequence_div"]["name"]};')
        for idx in range(len(target_optical_funcs["betx"])):
            bpm = target_optical_funcs["name"][idx]
            goals = [f"{var}={target_optical_funcs[var][idx]}" for var in target_vars_and_weights.values() if var not in ["qx", "qy"]]
            goals = ", ".join(goals)
            madx.input(f"constraint, sequence={structure['sequence_div']['name']}, range={bpm}, {goals};")
        weights = ""
        if "qx" in target_vars_and_weights or "qy" in target_vars_and_weights:
            weights += f"gweight, q1={target_vars_and_weights['qx']}, q2={target_vars_and_weights['qy']};"
        else:
            weights += "weight," + ",".join([f"{var}={weight}" for var, weight in target_vars_and_weights.items() if var not in ["qx", "qy"]]) + ";"
        madx.input(weights)

    for knob, param in knobs_for_matching:
        madx.input(f"vary, name = {knob}, step = {param_steps[param]};")

    for algorithm in algorithms:
        if algorithm == "lmdif":
            madx.input('lmdif, calls=3000, tolerance=1e-16;')
        elif algorithm == "simplex":
            madx.input('simplex, calls=1000, tolerance=1e-15;')
        elif algorithm == "migrad":
            madx.input('migrad, calls=1000, tolerance=1e-15, strategy=3;')
        elif algorithm == "jacobian":
            madx.input('jacobian, calls=1000, tolerance=1e-15, repeat=3;')
        else:
            raise ValueError(f"Check the optimization algorithm: {algorithm}")
    madx.input('endmatch;')

    for knob, param in knobs_for_matching:
        madx.input(f'value, {knob};')

    try:
        madx.select('FLAG = Twiss', 'class = monitor')
        madx.twiss(table='twiss', centre=True)
        madx.input('select, flag = twiss, clear;')

        madx.input("emit; ex = beam->ex; ey = beam->ey;")

        res = {"x": madx.table.twiss.selection().x,
               "y": madx.table.twiss.selection().y,
               "betx": madx.table.twiss.selection().betx,
               "bety": madx.table.twiss.selection().bety,
               "dx": madx.table.twiss.selection().dx,
               "dy": madx.table.twiss.selection().dy,
               "s": madx.table.twiss.selection().s,
               "name": [name.split(":")[0] for name in madx.table.twiss.selection().name],
               "qx": madx.table.summ.q1[0],
               "qy": madx.table.summ.q2[0],
               "betx_all": madx.table.twiss.betx,
               "bety_all": madx.table.twiss.bety,
               "dx_all": madx.table.twiss.dx,
               "dy_all": madx.table.twiss.dy,
               "s_all": madx.table.twiss.s,
               "x_all": madx.table.twiss.x,
               "y_all": madx.table.twiss.y,
               "name_all": madx.table.twiss.name,
               "ex": madx.globals["ex"],
               "ey": madx.globals["ey"],
               "coupling": 0 if madx.globals["ey"] == 1 and madx.globals["ex"] == 1 else madx.globals["ey"] / madx.globals["ex"]}

    except TwissFailed:
        print("Twiss Failed!")
        res = None

    matching_results = []
    for elem in elem_and_params_to_match.keys():
        matching_results.append(str(madx.elements[elem]))
    matching_results = "\n".join(matching_results)

    if save_matching:
        with open(file_to_save, 'w') as f:
            f.write(matching_results)

    madx.quit()
    del madx

    return res, matching_results


def correct_orbit(structure: dict,
                  imperfections_file: str = None,
                  aligns: dict = None,
                  old_aligns: dict = None,
                  file_with_kicks: str = None,
                  planes: List[str] = "x",
                  ncorrs: int = 0,
                  algorithm: str = "micado",
                  corrs_to_use: dict = None,
                  target_orbit: str = None,
                  sngval: int = 2,
                  sngcut: int = 50,
                  verbose: bool = False,
                  radiate: bool = False) -> dict:
    """
    Get optical functions, etc.

    :param structure: obtained from read_structure func
    :param imperfections_file: file name with imperfections in the madx format
    :param aligns: imperfections
    :param old_aligns: preexisted imperfections file_with_kicks
    :param file_with_kicks: a file name with corrector kicks obtained from the madx orbit correction command
    :param planes: coordinates for orbit correction
    :param ncorrs: number of correctors to use
    :param algorithm: method to solve an inverse problem
    :param corrs_to_use: desired correctors to be used in orbit correction
    :param target_orbit: file in TWISS TSV format with target orbit
    :param sngval: threshold for singular values and redundant correctors
    :param sngcut: threshold for redundant correctors
    :param verbose: whether to print debugging info to a console
    :param radiate: whether to turn on radiation
    :return: dict with optical functions, orbits, etc.
    """
    madx = Madx(stdout=verbose)
    collect_structure(structure, madx, verbose=verbose, radiate=radiate)

    if imperfections_file:
        madx.input(f"readtable, file={imperfections_file}, table=tabl;")
        madx.input("seterr, table=tabl;")
    else:
        Imperfections.add_to_model(madx, aligns)
        Imperfections.add_to_model(madx, old_aligns)

    if file_with_kicks:
        madx.input(f"call, file={file_with_kicks};")

    try:
        madx.select('FLAG = Twiss', 'class = monitor')
        madx.twiss(table='twiss', centre=True)
        madx.input('select, flag = twiss, clear;')

        if not planes:
            raise ValueError(f"Empty planes parameter: {planes}")
        if isinstance(planes, str):
            planes = [planes]

        extern = False
        if target_orbit is not None:
            madx.input(f'readtable, file="{target_orbit}", table="twiss_bpm";')
            target_twiss_orbit = 'twiss_bpm'
            extern = True

        for plane in planes:
            if plane == "x":
                corr_type = "hkicker"
            elif plane == "y":
                corr_type = "vkicker"
            else:
                raise ValueError(f"Unknown plane for orbit correction: {plane}")

            if corrs_to_use:
                for corr in structure["kick_total"][corr_type]:
                    madx.input(f"usekick, status=off, sequence={structure['sequence_div']['name']}, pattern={corr};")
                for corr in corrs_to_use[corr_type]:
                    madx.input(f"usekick, status=on, sequence={structure['sequence_div']['name']}, pattern={corr};")

            if extern:
                madx.input(f"correct, sequence={structure['sequence_div']['name']}, mode={algorithm}, plane={plane}, ncorr={ncorrs}, sngval={sngval}, sngcut={sngcut}, orbit=twiss, extern={extern}, target={target_twiss_orbit}, CLIST = corr_{plane}.out, MLIST = mon_{plane}.out, resout=1, error=1e-15;")
            else:
                madx.input(f"correct, sequence={structure['sequence_div']['name']}, mode={algorithm}, plane={plane}, ncorr={ncorrs}, sngval={sngval}, sngcut={sngcut}, orbit=twiss, CLIST = corr_{plane}.out, MLIST = mon_{plane}.out, resout=1, error=1e-15;")

        madx.select('FLAG = Twiss', 'class = monitor')
        madx.twiss(table='twiss', centre=True)
        madx.input('select, flag = twiss, clear;')

        madx.input("emit; ex = beam->ex; ey = beam->ey;")

        res = {"x": madx.table.twiss.selection().x,
               "y": madx.table.twiss.selection().y,
               "betx": madx.table.twiss.selection().betx,
               "bety": madx.table.twiss.selection().bety,
               "dx": madx.table.twiss.selection().dx,
               "dy": madx.table.twiss.selection().dy,
               "s": madx.table.twiss.selection().s,
               "name": [name.split(":")[0] for name in madx.table.twiss.selection().name],
               "qx": madx.table.summ.q1[0],
               "qy": madx.table.summ.q2[0],
               "betx_all": madx.table.twiss.betx,
               "bety_all": madx.table.twiss.bety,
               "dx_all": madx.table.twiss.dx,
               "dy_all": madx.table.twiss.dy,
               "s_all": madx.table.twiss.s,
               "x_all": madx.table.twiss.x,
               "y_all": madx.table.twiss.y,
               "name_all": madx.table.twiss.name,
               "ex": madx.globals["ex"],
               "ey": madx.globals["ey"],
               "coupling": 0 if madx.globals["ey"] == 1 and madx.globals["ex"] == 1 else madx.globals["ey"] / madx.globals["ex"]}

    except TwissFailed:
        print("Twiss Failed!")
        res = None

    madx.quit()
    del madx

    return res
