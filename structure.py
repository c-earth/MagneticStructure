from pymatgen.core.structure import Structure
import numpy as np

class MagneticStructure(Structure):
    '''
    Magnetic version of pymatgen.core.structure.Structure.
    Properties:
        lattice                 : pymatgen.core.lattice.Lattice
            Pymatgen's Lattice instance representing structure's lattice parameters
        propvecs                : Nx3 array
            list of magnetic propagation vectors 
            (fraction of reciprocal basis vectors)
        coords                  : Nx3 array
            list of bases' coordinates
            (cartesian)
        site_occupancies        : list(dict(pymatgen.core.periodic_table.Element))
            list of bases' atomic occupancies
            (probability of site occupation)
        site_moments            : list(dict(pymatgen.core.periodic_table.Element))
            list of bases' magnetic moment
            (cartesian)
        site_modulations        : list(dict(pymatgen.core.periodic_table.Element))
            list of bases' fundamental Fourier components ('cos', 'sin') of magnetic modulation
            (cartesian)
        incommensurate       : bool
            whether the structure is incommensurate or not
        alloy                : bool
            whether there is any substitution/doping site
        magnetic_alloy       : bool
            whether there is any magnetic atom substitution/doping of magnetic site
    '''
    
    def __init__(self,
                lattice,
                species,
                coords,
                propvecs,
                site_properties,
                charge = None,
                validate_proximity = False,
                to_unit_cell = False,
                coords_are_cartesian = False
                ):
        '''
        Generate instance that store magnetic structure data.
        Arguments:
            lattice                 : pymatgen.core.lattice.Lattice
            species                 : list(dict(pymatgen.core.periodic_table.Element))
            coords                  : Nx3 array
            propvecs                : Nx3 array
            site_properties         : dict
            charge                  : float
            validate_proximity      : bool
            to_unit_cell            : bool
            coords_are_cartesian    : bool
        '''
        super().__init__(lattice,
                        species,
                        coords,
                        charge,
                        validate_proximity,
                        to_unit_cell,
                        coords_are_cartesian,
                        site_properties)
        self._lattice = lattice
        self._propvecs = propvecs
        self._coords = [i.coords for i in self]
        self._site_occupancies = [i._data for i in self.species_and_occu]
        self._site_moments = self.site_properties['magmom']
        self._site_modulations = self.site_properties['modulation']
        self._incommensurate = self.is_incommensurate()
        self._alloy = self.is_alloy()
        self._magnetic_alloy = self.is_magnetic_alloy()
            
    @classmethod
    def from_sites(cls,
                    sites,
                    propvecs,
                    charge,
                    validate_proximity: bool = False,
                    to_unit_cell: bool = False,
                    ):
        if len(sites) < 1:
            raise ValueError(f"You need at least one site to construct a {cls}")
        prop_keys = []
        props = {}
        lattice = sites[0].lattice
        for i, site in enumerate(sites):
            if site.lattice != lattice:
                raise ValueError("Sites must belong to the same lattice")
            for k, v in site.properties.items():
                if k not in prop_keys:
                    prop_keys.append(k)
                    props[k] = [None] * len(sites)
                props[k][i] = v
        return cls(
            lattice,
            [site.species for site in sites],
            [site.frac_coords for site in sites],
            propvecs,
            site_properties=props,
            charge=charge,
            validate_proximity=validate_proximity,
            to_unit_cell=to_unit_cell,
        )
    
    @classmethod
    def from_str(cls,
                input_string: str,
                primitive=False,
                sort=False,
                merge_tol=0.0,
                ):
        from mcif import MCifParser
        parser = MCifParser.from_string(input_string, occupancy_tolerance=1.05)
        s = parser.get_structures(primitive=primitive)[0]

        if sort:
            s = s.get_sorted_structure()
        if merge_tol:
            s.merge_sites(merge_tol)
        return cls.from_sites(s, s._propvecs, charge=s._charge)
        
    @classmethod
    def magnetic_only_structure(cls):
        magnetic_site_indexes = []
        for i in range(len(self._coords)):
            moments = self._site_moments[i]
            modulations = self._site_modulations[i]
            for moment in moments.values():
                if np.linalg.norm(moment.moment) > 0:
                    magnetic_site_indexes.append(i)
                    break
            for modulation in modulations.values():
                for mod in modulation.values():
                    if np.linalg.norm(mod['cos'].moment) + np.linalg.norm(mod['sin'].moment) > 0:
                        magnetic_site_indexes.append(i)
                        break
        return cls(self._lattice,
                    [self._site_occupancies[i] for i in magnetic_site_indexes],
                    [self._coords for i in magnetic_site_indexes],
                    self._propvecs,
                    {'magmom': [self._site_moments[i] for i in magnetic_site_indexes], 
                    'modulation': [self._site_modulations[i] for i in magnetic_site_indexes]},
                    charge = None,
                    validate_proximity = False,
                    to_unit_cell = False,
                    coords_are_cartesian = False)
                    
    def is_incommensurate(self):
        for i in range(len(self._coords)):
            modulations = self._site_modulations[i]
            for modulation in modulations.values():
                for mod in modulation.values():
                    if np.linalg.norm(mod['cos'].moment) + np.linalg.norm(mod['sin'].moment) > 0:
                        return True
        return False
        
    def is_alloy(self):
        for i in range(len(self._coords)):
            occupancies = self._site_occupancies[i]
            if np.sum(np.array(list(occupancies.values()))>0) > 1:
                return True
        return False
    
    def is_magnetic_alloy(self):
        for i in range(len(self._coords)):
            occupancies = self._site_occupancies[i]
            if np.sum(np.array(list(occupancies.values()))>0) > 1:
                moments = self._site_moments[i]
                for moment in moments.values():
                    if np.linalg.norm(moment.moment) > 0:
                        return True
        return False
    
    def get_sorted_structure(self, key = None, reverse = False):
        sites = sorted(self, key=key, reverse=reverse)
        return type(self).from_sites(sites, self._propvecs, charge=self._charge)
