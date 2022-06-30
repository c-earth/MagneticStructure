from pymatgen.core.structure import Structure
from MagneticStructure.mcif import MCifParser
import warnings

class MagneticStructure(Structure):
    '''
    Magnetic version of Pymatgen's Structure.
    '''
    def __init__(
        self,
        lattice,
        species,
        coords,
        propvecs,
        incommensurate,
        charge: float = None,
        validate_proximity: bool = False,
        to_unit_cell: bool = False,
        coords_are_cartesian: bool = False,
        site_properties: dict = None,
    ):

        super().__init__(
            lattice,
            species,
            coords,
            charge=charge,
            validate_proximity=validate_proximity,
            to_unit_cell=to_unit_cell,
            coords_are_cartesian=coords_are_cartesian,
            site_properties=site_properties,
        )
        self.incommensurate = incommensurate
        self.propvecs = propvecs
    
    @classmethod
    def from_sites(
        cls,
        sites,
        propvecs,
        incommensurate,
        charge,
        validate_proximity: bool = False,
        to_unit_cell: bool = False,
    ):
        if len(sites) < 1:
            raise ValueError(f"You need at least one site to construct a {cls}")
        prop_keys = []  # type: List[str]
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
            incommensurate,
            charge=charge,
            site_properties=props,
            validate_proximity=validate_proximity,
            to_unit_cell=to_unit_cell,
        )
    
    @classmethod
    def from_str(
        cls,
        input_string: str,
        primitive=False,
        sort=False,
        merge_tol=0.0,
    ):
        parser = MCifParser.from_string(input_string, occupancy_tolerance=1.05)
        s = parser.get_structures(primitive=primitive)[0]

        if sort:
            s = s.get_sorted_structure()
        if merge_tol:
            s.merge_sites(merge_tol)
        return cls.from_sites(s, s.propvecs, s.incommensurate, charge=s._charge)
    
    def get_sorted_structure(self, key = None, reverse = False):
        sites = sorted(self, key=key, reverse=reverse)
        return type(self).from_sites(sites, self.propvecs, self.incommensurate, charge=self._charge)
