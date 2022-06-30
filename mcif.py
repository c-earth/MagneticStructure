from structure import MagneticStructure
from pymatgen.io.cif import CifParser, CifFile
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.core.periodic_table import Element
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.core.lattice import Lattice
from itertools import groupby
from io import StringIO
import collections
from pathlib import Path
import numpy as np
import math
import re

class MCifParser(CifParser):
    '''
    MCIF version of CifParser
    '''
    
    def __init__(self, filename, occupancy_tolerance=1.0, site_tolerance=1e-4):
        
        self._occupancy_tolerance = occupancy_tolerance
        self._site_tolerance = site_tolerance
        if isinstance(filename, (str, Path)):
            self._cif = CifFile.from_file(filename)
        else:
            self._cif = CifFile.from_string(filename.read())
        
        self.warnings = []
        self.feature_flags = {}
        self.feature_flags['magcif'] = True
        
        def is_incommensurate():
            prefixes = ["_cell_modulation_dimension", "_cell_wave_vector"]
            for d in self._cif.data.values():
                for k in d.data.keys():
                    for prefix in prefixes:
                        if prefix in k:
                            return True
            return False
        
        self.feature_flags['incommensurate'] = is_incommensurate()
        
        for k in self._cif.data.keys():
            self._cif.data[k] = self._sanitize_data(self._cif.data[k])
    
    @staticmethod
    def from_string(cif_string, occupancy_tolerance=1.0):
    
        stream = StringIO(cif_string)
        return MCifParser(stream, occupancy_tolerance)
        
    def _sanitize_data(self, data):
        if "_atom_site_attached_hydrogens" in data.data.keys():
            attached_hydrogens = [str2float(x) for x in data.data["_atom_site_attached_hydrogens"] if str2float(x) != 0]
            if len(attached_hydrogens) > 0:
                self.warnings.append(
                    "Structure has implicit hydrogens defined, "
                    "parsed structure unlikely to be suitable for use "
                    "in calculations unless hydrogens added."
                )

        if "_atom_site_type_symbol" in data.data.keys():

            idxs_to_remove = []

            new_atom_site_label = []
            new_atom_site_type_symbol = []
            new_atom_site_occupancy = []
            new_fract_x = []
            new_fract_y = []
            new_fract_z = []

            for idx, el_row in enumerate(data["_atom_site_label"]):


                if len(data["_atom_site_type_symbol"][idx].split(" + ")) > len(
                    data["_atom_site_label"][idx].split(" + ")
                ):

                    els_occu = {}

                    symbol_str = data["_atom_site_type_symbol"][idx]
                    symbol_str_lst = symbol_str.split(" + ")
                    for elocc_idx, sym in enumerate(symbol_str_lst):
                        symbol_str_lst[elocc_idx] = re.sub(r"\([0-9]*\)", "", sym.strip())
                        els_occu[
                            str(re.findall(r"\D+", symbol_str_lst[elocc_idx].strip())[1]).replace("<sup>", "")
                        ] = float("0" + re.findall(r"\.?\d+", symbol_str_lst[elocc_idx].strip())[1])

                    x = str2float(data["_atom_site_fract_x"][idx])
                    y = str2float(data["_atom_site_fract_y"][idx])
                    z = str2float(data["_atom_site_fract_z"][idx])

                    for et, occu in els_occu.items():
                        new_atom_site_label.append(et + "_fix" + str(len(new_atom_site_label)))
                        new_atom_site_type_symbol.append(et)
                        new_atom_site_occupancy.append(str(occu))
                        new_fract_x.append(str(x))
                        new_fract_y.append(str(y))
                        new_fract_z.append(str(z))

                    idxs_to_remove.append(idx)

            for original_key in data.data:
                if isinstance(data.data[original_key], list):
                    for id in sorted(idxs_to_remove, reverse=True):
                        del data.data[original_key][id]

            if len(idxs_to_remove) > 0:
                self.warnings.append("Pauling file corrections applied.")

                data.data["_atom_site_label"] += new_atom_site_label
                data.data["_atom_site_type_symbol"] += new_atom_site_type_symbol
                data.data["_atom_site_occupancy"] += new_atom_site_occupancy
                data.data["_atom_site_fract_x"] += new_fract_x
                data.data["_atom_site_fract_y"] += new_fract_y
                data.data["_atom_site_fract_z"] += new_fract_z

        if self.feature_flags["magcif"]:

            correct_keys = [
                "_space_group_symop_magn_operation.xyz",
                "_space_group_symop_magn_centering.xyz",
                "_space_group_magn.name_BNS",
                "_space_group_magn.number_BNS",
                "_atom_site_moment_crystalaxis_x",
                "_atom_site_moment_crystalaxis_y",
                "_atom_site_moment_crystalaxis_z",
                "_atom_site_moment_label",
                "_atom_site_moment_Fourier_atom_site_label",
                "_atom_site_moment_Fourier_axis",
                "_atom_site_moment_Fourier_wave_vector_seq_id",
                "_atom_site_moment_Fourier_param_cos",
                "_atom_site_moment_Fourier_param_sin",
                "_atom_site_Fourier_wave_vector_seq_id",
                "_atom_site_Fourier_wave_vector_q1_coeff",
                "_atom_site_Fourier_wave_vector_q2_coeff"
            ]

            changes_to_make = {}

            for original_key in data.data:
                for correct_key in correct_keys:
                    # convert to all underscore
                    trial_key = "_".join(correct_key.split("."))
                    test_key = "_".join(original_key.split("."))
                    if trial_key == test_key:
                        changes_to_make[correct_key] = original_key

            # make changes
            for correct_key, original_key in changes_to_make.items():
                data.data[correct_key] = data.data[original_key]

            # renamed_keys maps interim_keys to final_keys
            renamed_keys = {
                "_magnetic_space_group.transform_to_standard_Pp_abc": "_space_group_magn.transform_BNS_Pp_abc"
            }
            changes_to_make = {}

            for interim_key, final_key in renamed_keys.items():
                if data.data.get(interim_key):
                    changes_to_make[final_key] = interim_key

            if len(changes_to_make) > 0:
                self.warnings.append("Keys changed to match new magCIF specification.")

            for final_key, interim_key in changes_to_make.items():
                data.data[final_key] = data.data[interim_key]

        # check for finite precision frac coordinates (e.g. 0.6667 instead of 0.6666666...7)
        # this can sometimes cause serious issues when applying symmetry operations
        important_fracs = (1 / 3.0, 2 / 3.0)
        fracs_to_change = {}
        for label in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"):
            if label in data.data.keys():
                for idx, frac in enumerate(data.data[label]):
                    try:
                        frac = str2float(frac)
                    except Exception:
                        # coordinate might not be defined e.g. '?'
                        continue
                    for comparison_frac in important_fracs:
                        if abs(1 - frac / comparison_frac) < 1e-4:
                            fracs_to_change[(label, idx)] = str(comparison_frac)
        if fracs_to_change:
            self.warnings.append(
                "Some fractional coordinates rounded to ideal values to avoid issues with finite precision."
            )
            for (label, idx), val in fracs_to_change.items():
                data.data[label][idx] = val

        return data
        
    def _unique_coords(self, coords_in, magmoms_in=None, modulation_in=None, lattice=None):
        coords = []
        if magmoms_in:
            magmoms = []
            modulations = []
            if len(magmoms_in) != len(coords_in):
                raise ValueError
            for tmp_coord, tmp_magmom, tmp_modulation in zip(coords_in, magmoms_in, modulation_in):
                for op in self.symmetry_operations:
                    coord = op.operate(tmp_coord)
                    coord = np.array([i - math.floor(i) for i in coord])
                    magmom = dict()
                    modulation = dict()
                    if isinstance(op, MagSymmOp):
                        for el in tmp_magmom.keys():
                            magmom[el] = Magmom.from_moment_relative_to_crystal_axes(
                                op.operate_magmom(tmp_magmom[el]), lattice=lattice)
                            modulation[el] = dict()
                            for vec in tmp_modulation[el].keys():
                                modulation[el][vec] = {'cos': Magmom.from_moment_relative_to_crystal_axes(
                                op.operate_magmom(tmp_modulation[el][vec]['cos']), lattice=lattice),
                                                        'sin': Magmom.from_moment_relative_to_crystal_axes(
                                op.operate_magmom(tmp_modulation[el][vec]['sin']), lattice=lattice)}
                    else:
                        for el in tmp_magmom.keys():
                            magmom[el] = Magmom(tmp_magmom[el])
                            modulation[el] = dict()
                            for vec in tmp_modulation.keys():
                                modulation[el][vec] = {'cos': Magmom(tmp_modulation[el][vec]['cos']),
                                                        'sin': Magmom(tmp_modulation[el][vec]['sin'])}
                    if not in_coord_list_pbc(coords, coord, atol=self._site_tolerance):
                        coords.append(coord)
                        magmoms.append(magmom)
                        modulations.append(modulation)
            return coords, magmoms, modulations
    
    def get_lattice(
        self,
        data,
        length_strings=("a", "b", "c"),
        angle_strings=("alpha", "beta", "gamma"),
        lattice_type=None,
    ):
        try:

            lengths = [str2float(data["_cell_length_" + i]) for i in length_strings]
            angles = [str2float(data["_cell_angle_" + i]) for i in angle_strings]
            if not lattice_type:
                return Lattice.from_parameters(*lengths, *angles)

            return getattr(Lattice, lattice_type)(*(lengths + angles))

        except KeyError:
            # Missing Key search for cell setting
            for lattice_lable in [
                "_symmetry_cell_setting",
                "_space_group_crystal_system",
            ]:
                if data.data.get(lattice_lable):
                    lattice_type = data.data.get(lattice_lable).lower()
                    try:

                        required_args = getargspec(getattr(Lattice, lattice_type)).args

                        lengths = (l for l in length_strings if l in required_args)
                        angles = (a for a in angle_strings if a in required_args)
                        return self.get_lattice(data, lengths, angles, lattice_type=lattice_type)
                    except AttributeError as exc:
                        self.warnings.append(str(exc))
                        warnings.warn(exc)

                else:
                    return None
        return None
        
    def parse_modulation(self, data):
        modulation = dict()
        for label in data['_atom_site_label']:
            modulation[label] = {i: {'cos': np.array([0, 0, 0]), 'sin': np.array([0, 0, 0])} for i in self.propvecs.keys()}
        if '_atom_site_moment_Fourier_atom_site_label' in data.data.keys():
            for label, vec, axis, cos, sin in zip(data['_atom_site_moment_Fourier_atom_site_label'],
                                                    data['_atom_site_moment_Fourier_wave_vector_seq_id'],
                                                    data['_atom_site_moment_Fourier_axis'],
                                                    data['_atom_site_moment_Fourier_param_cos'],
                                                    data['_atom_site_moment_Fourier_param_sin']):
                if vec not in self.propvecs.keys():
                    continue
                if axis == 'x':
                    modulation[label][vec]['cos'][0] = str2float(cos)
                    modulation[label][vec]['sin'][0] = str2float(sin)
                elif axis == 'y':
                    modulation[label][vec]['cos'][1] = str2float(cos)
                    modulation[label][vec]['sin'][1] = str2float(sin)
                elif axis == 'z':
                    modulation[label][vec]['cos'][2] = str2float(cos)
                    modulation[label][vec]['sin'][2] = str2float(sin)    
        return modulation
    
    def get_propvecs(self, data):
        propvecs = dict()
        if '_parent_propagation_vector.id' in data.data.keys():
            propvec_id = None
            propvec = None
            for front, back in zip(data['_parent_propagation_vector.id'], data['_parent_propagation_vector.kxkykz']):
                if front[0] in {'k', 'K'}:
                    propvec_id = front
                    propvec = back
                else:
                    propvec += ',' + front + ',' +back
                    propvecs[propvec_id[1:]] = np.array(eval(propvec))
        elif '_cell_wave_vector_seq_id' in data.data.keys():
            propvecs = {data['_cell_wave_vector_seq_id'][i]: np.array(
                                [
                                    str2float(data["_cell_wave_vector_x"][i]),
                                    str2float(data["_cell_wave_vector_y"][i]),
                                    str2float(data["_cell_wave_vector_z"][i]),
                                ]
                            )
                            for i in range(len(data['_cell_wave_vector_seq_id']))}
        
        return propvecs
        
           
    def _get_structure(self, data, primitive, symmetrized):
        def get_num_implicit_hydrogens(sym):
            num_h = {"Wat": 2, "wat": 2, "O-H": 1}
            return num_h.get(sym[:3], 0)
        lattice = self.get_lattice(data)
        self.symmetry_operations = self.get_magsymops(data)
        self.propvecs = self.get_propvecs(data)
        magmoms_ave = self.parse_magmoms(data, lattice=lattice)
        magmoms_mod = self.parse_modulation(data)
        oxi_states = self.parse_oxi_states(data)
        coord_to_species = {}
        coord_to_magmoms_ave = {}
        coord_to_magmoms_mod = {}
        def get_matching_coord(coord):
            keys = list(coord_to_species.keys())
            coords = np.array(keys)
            for op in self.symmetry_operations:
                c = op.operate(coord)
                inds = find_in_coord_list_pbc(coords, c, atol=self._site_tolerance)
                if len(inds):
                    return keys[inds[0]]
            return False
        for i in range(len(data["_atom_site_label"])):
            try:
                symbol = self._parse_symbol(data["_atom_site_type_symbol"][i])
                num_h = get_num_implicit_hydrogens(data["_atom_site_type_symbol"][i])
            except KeyError:
                symbol = self._parse_symbol(data["_atom_site_label"][i])
                num_h = get_num_implicit_hydrogens(data["_atom_site_label"][i])
            if not symbol:
                continue

            if oxi_states is not None:
                o_s = oxi_states.get(symbol, 0)
                if "_atom_site_type_symbol" in data.data.keys():
                    oxi_symbol = data["_atom_site_type_symbol"][i]
                    o_s = oxi_states.get(oxi_symbol, o_s)
                try:
                    el = Species(symbol, o_s)
                except Exception:
                    el = DummySpecies(symbol, o_s)
            else:
                el = get_el_sp(symbol)
            
            x = str2float(data["_atom_site_fract_x"][i])
            y = str2float(data["_atom_site_fract_y"][i])
            z = str2float(data["_atom_site_fract_z"][i])
            magmom_ave = magmoms_ave.get(data["_atom_site_label"][i], np.array([0, 0, 0]))
            magmom_mod = magmoms_mod[data["_atom_site_label"][i]]
            
            try:
                occu = str2float(data["_atom_site_occupancy"][i])
            except (KeyError, ValueError):
                occu = 1

            if occu > 0:
                coord = (x, y, z)
                match = get_matching_coord(coord)
                comp_d = {el: occu}
                magm_d = {el: magmom_ave}
                modu_d = {el: magmom_mod}
                if num_h > 0:
                    comp_d["H"] = num_h
                    self.warnings.append(
                        "Structure has implicit hydrogens defined, "
                        "parsed structure unlikely to be suitable for use "
                        "in calculations unless hydrogens added."
                    )
                comp = Composition(comp_d)
                if not match:
                    coord_to_species[coord] = comp
                    coord_to_magmoms_ave[coord] = magm_d
                    coord_to_magmoms_mod[coord] = modu_d
                else:
                    coord_to_species[match] += comp
                    coord_to_magmoms_ave[match][el] = magmom_ave
                    coord_to_magmoms_mod[match][el] = magmom_mod
        sum_occu = [
            sum(c.values()) for c in coord_to_species.values() if not set(c.elements) == {Element("O"), Element("H")}
        ]
        if any(o > 1.05 for o in sum_occu):
            msg = (
                "Some occupancies ({}) sum to > 1! If they are within "
                "the occupancy_tolerance, they will be rescaled. "
                "The current occupancy_tolerance is set to: {}".format(sum_occu, self._occupancy_tolerance)
            )
            warnings.warn(msg)
            self.warnings.append(msg)

        allspecies = []
        allcoords = []
        allmagmoms = []
        allmodulations = []
        allhydrogens = []
        equivalent_indices = []
        if coord_to_species.items():
            for idx, (comp, group) in enumerate(
                groupby(
                    sorted(list(coord_to_species.items()), key=lambda x: x[1]),
                    key=lambda x: x[1],
                )
            ):
                tmp_coords = [site[0] for site in group]
                tmp_magmom_ave = [coord_to_magmoms_ave[tmp_coord] for tmp_coord in tmp_coords]
                tmp_magmom_mod = [coord_to_magmoms_mod[tmp_coord] for tmp_coord in tmp_coords]
                coords, magmoms, modulations = self._unique_coords(tmp_coords, magmoms_in=tmp_magmom_ave, modulation_in=tmp_magmom_mod, lattice=lattice)
                if set(comp.elements) == {Element("O"), Element("H")}:
                    im_h = comp["H"]
                    species = Composition({"O": comp["O"]})
                else:
                    im_h = 0
                    species = comp

                equivalent_indices += len(coords) * [idx]

                allhydrogens.extend(len(coords) * [im_h])
                allcoords.extend(coords)
                allspecies.extend(len(coords) * [species])
                allmagmoms.extend(magmoms)
                allmodulations.extend(modulations)

            for i, species in enumerate(allspecies):
                totaloccu = sum(species.values())
                if 1 < totaloccu <= self._occupancy_tolerance:
                    allspecies[i] = species / totaloccu
        if allspecies and len(allspecies) == len(allcoords) and len(allspecies) == len(allmagmoms):
            site_properties = {}
            if any(allhydrogens):
                assert len(allhydrogens) == len(allcoords)
                site_properties["implicit_hydrogens"] = allhydrogens

            site_properties["magmom"] = allmagmoms
            site_properties["modulation"] = allmodulations

            if len(site_properties) == 0:
                site_properties = None

            struct = MagneticStructure(lattice, allspecies, allcoords, self.propvecs, self.feature_flags['incommensurate'], site_properties=site_properties)

            if symmetrized:

                wyckoffs = ["Not Parsed"] * len(struct)
                sg = SpacegroupOperations("Not Parsed", -1, self.symmetry_operations)

                return SymmetrizedStructure(struct, sg, equivalent_indices, wyckoffs)

            struct = struct.get_sorted_structure()

            if primitive:
                struct = struct.get_primitive_structure(use_site_props=True)

            return struct
            
def str2float(text):
    """
    Remove uncertainty brackets from strings and return the float.
    """

    try:
        # Note that the ending ) is sometimes missing. That is why the code has
        # been modified to treat it as optional. Same logic applies to lists.
        return float(re.sub(r"\(.+\)*", "", text))
    except TypeError:
        if isinstance(text, list) and len(text) == 1:
            return float(re.sub(r"\(.+\)*", "", text[0]))
    except ValueError as ex:
        if text.strip() == ".":
            return 0
        raise ex
    raise ValueError(f"{text} cannot be converted to float")
