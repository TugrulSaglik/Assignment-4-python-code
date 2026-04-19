"""
Module: frame_solver
Purpose: Object-Oriented structural hierarchy and analysis pipeline with strict input validation.
"""
from matrix_library import Matrix, SparseMatrix

# --- INDEPENDENT CLASSES ---
class Material:
    def __init__(self, mat_id, E, alpha=0.0):
        if E <= 0: raise ValueError(f"Material {mat_id}: E must be positive. Received {E}")
        self.mat_id = mat_id
        self.E = E
        self.alpha = alpha

class Section:
    def __init__(self, sec_id, A, I, depth=0.0):
        if A <= 0: raise ValueError(f"Section {sec_id}: A must be positive. Received {A}")
        if I <= 0: raise ValueError(f"Section {sec_id}: I must be positive. Received {I}")
        if depth < 0: raise ValueError(f'Section {sec_id}: Depth cannot be negative. Received {depth}')
        self.sec_id = sec_id
        self.A = A
        self.I = I
        self.depth = depth

# --- SUPPORTS & LOADS ---
class Support:
    def __init__(self, dx=0.0, dy=0.0, rz=0.0):
        self.rx, self.ry, self.rz = 0, 0, 0
        self.settlement_dx = dx
        self.settlement_dy = dy
        self.settlement_rz = rz

class Fixed(Support):
    def __init__(self, dx=0.0, dy=0.0, rz=0.0):
        super().__init__(dx, dy, rz)
        self.rx, self.ry, self.rz = 1, 1, 1

class Pin(Support):
    def __init__(self, dx=0.0, dy=0.0, rz=0.0):
        super().__init__(dx, dy, rz)
        self.rx, self.ry, self.rz = 1, 1, 0

class Roller(Support):
    def __init__(self, dx=0.0, dy=0.0, rz=0.0):
        super().__init__(dx, dy, rz)
        self.rx, self.ry, self.rz = 0, 1, 0

class FixedRoller(Support):
    def __init__(self, dx=0.0, dy=0.0, rz=0.0):
        super().__init__(dx, dy, rz)
        self.rx, self.ry, self.rz = 0, 1, 1

class Load:
    def __init__(self, load_id): self.load_id = load_id

class NodalLoad(Load):
    def __init__(self, load_id, fx=0.0, fy=0.0, mz=0.0):
        if fx == 0.0 and fy == 0.0 and mz == 0.0:
            raise ValueError(f"NodalLoad {load_id}: Must have non-zero force.")
        super().__init__(load_id)
        self.fx, self.fy, self.mz = fx, fy, mz

class MemberLoad(Load):
    def __init__(self, load_id): super().__init__(load_id)

class PointLoad(MemberLoad):
    def __init__(self, load_id, magnitude, location_ratio):
        if magnitude == 0.0: raise ValueError("Magnitude cannot be zero.")
        if not (0.0 <= location_ratio <= 1.0): raise ValueError("Location ratio out of bounds.")
        super().__init__(load_id)
        self.magnitude, self.location_ratio = magnitude, location_ratio

class UniformlyDistributedLoad(MemberLoad):
    def __init__(self, load_id, magnitude):
        if magnitude == 0.0: raise ValueError("Magnitude cannot be zero.")
        super().__init__(load_id)
        self.magnitude = magnitude

class TemperatureLoad(MemberLoad):
    def __init__(self, load_id, t_top, t_bottom):
        super().__init__(load_id)
        self.t_top = t_top
        self.t_bottom = t_bottom

# --- NODES & ELEMENTS ---
class Node:
    def __init__(self, node_id, x, y):
        self.node_id, self.x, self.y = node_id, x, y
        self.support = None
        self.nodal_loads, self.connected_elements = [], []

    def assign_support(self, support): self.support = support
    def assign_load(self, load):
        if isinstance(load, NodalLoad): self.nodal_loads.append(load)

class Element:
    def __init__(self, elem_id): self.elem_id = elem_id

class Spring(Element):
    def __init__(self, elem_id, node, stiffness):
        if stiffness <= 0: raise ValueError("Stiffness must be positive.")
        super().__init__(elem_id)
        self.node, self.stiffness = node, stiffness
        node.connected_elements.append(self)

class Member(Element):
    def __init__(self, elem_id, start_node, end_node, material, section, offset_start_y=0.0, offset_end_y=0.0):
        if start_node.node_id == end_node.node_id: 
            raise ValueError("Identical nodes.")
        
        # ADDED BACK: Catch overlapping coordinates immediately upon element creation
        if start_node.x == end_node.x and start_node.y == end_node.y: 
            raise ValueError("Member start and end nodes have identical coordinates (Zero length).")
            
        super().__init__(elem_id)
        self.start_node, self.end_node = start_node, end_node
        self.material, self.section = material, section
        self.offset_start_y = offset_start_y
        self.offset_end_y = offset_end_y
        self.member_loads = []
        start_node.connected_elements.append(self)
        end_node.connected_elements.append(self)

    def assign_load(self, load):
        if isinstance(load, MemberLoad): self.member_loads.append(load)
        
    def get_length_and_angles(self):
        # Calculate physical geometry based on the offsets
        x1 = self.start_node.x
        y1 = self.start_node.y + self.offset_start_y
        x2 = self.end_node.x
        y2 = self.end_node.y + self.offset_end_y
        
        dx = x2 - x1; dy = y2 - y1
        L = (dx**2 + dy**2)**0.5
        if L == 0: raise ValueError("Zero physical length.")
        return L, dx/L, dy/L

    def get_rotation_matrix(self):
        _, c, s = self.get_length_and_angles()
        R = Matrix(6, 6)
        R.set_val(0, 0, c); R.set_val(1, 1, c); R.set_val(3, 3, c); R.set_val(4, 4, c)
        R.set_val(0, 1, s); R.set_val(3, 4, s)
        R.set_val(1, 0, -s); R.set_val(4, 3, -s)
        R.set_val(2, 2, 1); R.set_val(5, 5, 1)
        return R

    def get_global_offset_transformation(self):
        """Matrix T that transforms Global Nodal DOFs to Global Physical DOFs."""
        T = Matrix(6, 6)
        for i in range(6): T.set_val(i, i, 1.0)
        T.set_val(0, 2, -self.offset_start_y)
        T.set_val(3, 5, -self.offset_end_y)
        return T

class TrussElement(Member):
    def __init__(self, elem_id, start_node, end_node, material, section, offset_start_y=0.0, offset_end_y=0.0):
        super().__init__(elem_id, start_node, end_node, material, section, offset_start_y, offset_end_y)

    def get_local_stiffness(self):
        L, _, _ = self.get_length_and_angles()
        EA_L = (self.material.E * self.section.A) / L
        k_loc = Matrix(6, 6)
        k_loc.set_val(0, 0, EA_L); k_loc.set_val(3, 3, EA_L)
        k_loc.set_val(0, 3, -EA_L); k_loc.set_val(3, 0, -EA_L)
        return k_loc

    def get_fixed_end_forces(self):
        fef = [0.0] * 6
        for load in self.member_loads:
            if isinstance(load, TemperatureLoad):
                T_avg = (load.t_top + load.t_bottom) / 2.0
                axial_fef = self.material.E * self.section.A * self.material.alpha * T_avg
                fef[0] += axial_fef; fef[3] -= axial_fef
        return fef

class FrameElement(Member):
    def __init__(self, elem_id, start_node, end_node, material, section, release_start=False, release_end=False, offset_start_y=0.0, offset_end_y=0.0):
        super().__init__(elem_id, start_node, end_node, material, section, offset_start_y, offset_end_y)
        self.release_start = release_start
        self.release_end = release_end

    def get_local_stiffness(self):
        L, _, _ = self.get_length_and_angles()
        E, A, I = self.material.E, self.section.A, self.section.I
        k_loc = Matrix(6, 6)
        
        EA_L = E * A / L
        k_loc.set_val(0, 0, EA_L); k_loc.set_val(3, 3, EA_L)
        k_loc.set_val(0, 3, -EA_L); k_loc.set_val(3, 0, -EA_L)

        if not self.release_start and not self.release_end:
            k_loc.set_val(1, 1, 12*E*I/(L**3)); k_loc.set_val(4, 4, 12*E*I/(L**3))
            k_loc.set_val(1, 4, -12*E*I/(L**3)); k_loc.set_val(4, 1, -12*E*I/(L**3))
            k_loc.set_val(1, 2, 6*E*I/(L**2)); k_loc.set_val(2, 1, 6*E*I/(L**2))
            k_loc.set_val(1, 5, 6*E*I/(L**2)); k_loc.set_val(5, 1, 6*E*I/(L**2))
            k_loc.set_val(4, 2, -6*E*I/(L**2)); k_loc.set_val(2, 4, -6*E*I/(L**2))
            k_loc.set_val(4, 5, -6*E*I/(L**2)); k_loc.set_val(5, 4, -6*E*I/(L**2))
            k_loc.set_val(2, 2, 4*E*I/L); k_loc.set_val(5, 5, 4*E*I/L)
            k_loc.set_val(2, 5, 2*E*I/L); k_loc.set_val(5, 2, 2*E*I/L)
        elif self.release_start and not self.release_end: 
            k_loc.set_val(1, 1, 3*E*I/(L**3)); k_loc.set_val(4, 4, 3*E*I/(L**3))
            k_loc.set_val(1, 4, -3*E*I/(L**3)); k_loc.set_val(4, 1, -3*E*I/(L**3))
            k_loc.set_val(1, 5, 3*E*I/(L**2)); k_loc.set_val(5, 1, 3*E*I/(L**2))
            k_loc.set_val(4, 5, -3*E*I/(L**2)); k_loc.set_val(5, 4, -3*E*I/(L**2))
            k_loc.set_val(5, 5, 3*E*I/L)
        elif not self.release_start and self.release_end: 
            k_loc.set_val(1, 1, 3*E*I/(L**3)); k_loc.set_val(4, 4, 3*E*I/(L**3))
            k_loc.set_val(1, 4, -3*E*I/(L**3)); k_loc.set_val(4, 1, -3*E*I/(L**3))
            k_loc.set_val(1, 2, 3*E*I/(L**2)); k_loc.set_val(2, 1, 3*E*I/(L**2))
            k_loc.set_val(4, 2, -3*E*I/(L**2)); k_loc.set_val(2, 4, -3*E*I/(L**2))
            k_loc.set_val(2, 2, 3*E*I/L)
        return k_loc

    def get_fixed_end_forces(self):
        L, _, _ = self.get_length_and_angles()
        E, A, I = self.material.E, self.section.A, self.section.I
        alpha, h = self.material.alpha, self.section.depth
        fef = [0.0] * 6

        for load in self.member_loads:
            if isinstance(load, UniformlyDistributedLoad):
                w = load.magnitude
                fef[1] -= w * L / 2; fef[2] -= (w * L**2) / 12
                fef[4] -= w * L / 2; fef[5] += (w * L**2) / 12
            
            elif isinstance(load, PointLoad):
                P = load.magnitude
                a = load.location_ratio * L; b = L - a
                fef[1] -= P * (b**2) * (3*a + b) / (L**3)
                fef[2] -= P * a * (b**2) / (L**2)
                fef[4] -= P * (a**2) * (a + 3*b) / (L**3)
                fef[5] += P * (a**2) * b / (L**2)
            
            elif isinstance(load, TemperatureLoad):
                T_avg = (load.t_top + load.t_bottom) / 2.0
                axial_fef = E * A * alpha * T_avg
                fef[0] += axial_fef; fef[3] -= axial_fef   
                if h > 0:
                    # FIXED: Swapped to t_bottom - t_top
                    M_temp = (E * I * alpha * (load.t_bottom - load.t_top)) / h
                    fef[2] += M_temp; fef[5] -= M_temp

        if self.release_start and not self.release_end:
            M1 = fef[2]; fef[2] = 0.0
            fef[5] -= 0.5 * M1; fef[1] -= 1.5 * M1 / L; fef[4] += 1.5 * M1 / L
        elif not self.release_start and self.release_end:
            M2 = fef[5]; fef[5] = 0.0
            fef[2] -= 0.5 * M2; fef[1] += 1.5 * M2 / L; fef[4] -= 1.5 * M2 / L
        elif self.release_start and self.release_end:
            M1, M2 = fef[2], fef[5]; fef[2] = 0.0; fef[5] = 0.0
            fef[1] -= (M1 + M2) / L; fef[4] += (M1 + M2) / L
        return fef
    
# --- STRUCTURAL MODEL ---
class StructuralModel:
    def __init__(self):
        self.materials, self.sections, self.nodes, self.elements = {}, {}, {}, {}
        self.equation_map = {}
        self.num_equations = 0
        self.K_global_struct = None
        self.load_vector, self.displacements = [], []
        self.member_forces = {}

    def validate_model(self):
        if not self.nodes: raise ValueError("Model contains no nodes.")
        if not self.elements: raise ValueError("Model contains no elements.")

        coords = set()
        for n_id, node in self.nodes.items():
            coord = (node.x, node.y)
            if coord in coords: raise ValueError(f"Node {n_id} overlaps an existing node.")
            coords.add(coord)
            if len(node.connected_elements) == 0:
                raise ValueError(f"Node {n_id} is floating.")

        total_restraints = sum(n.support.rx + n.support.ry + n.support.rz for n in self.nodes.values() if n.support)
        if total_restraints < 3:
            raise ValueError("Kinematically unstable: Requires at least 3 restrained degrees of freedom.")

        visited_nodes = set()
        stack = [next(iter(self.nodes.values()))]
        while stack:
            curr_node = stack.pop()
            if curr_node.node_id not in visited_nodes:
                visited_nodes.add(curr_node.node_id)
                for elem in curr_node.connected_elements:
                    if hasattr(elem, 'start_node'):
                        adj = elem.end_node if elem.start_node == curr_node else elem.start_node
                        stack.append(adj)
        if len(visited_nodes) < len(self.nodes):
            raise ValueError("Disconnected sub-structures (floating members) detected.")

    def process_equations(self):
        self.validate_model()
        self.equation_map = {n_id: [0, 0, 0] for n_id in self.nodes.keys()}
        
        for n_id, node in self.nodes.items():
            rx, ry, rz = 0, 0, 0
            if node.support:
                rx, ry, rz = node.support.rx, node.support.ry, node.support.rz
            
            # LOCAL INSTABILITY FIX: If a node is free to rotate but ONLY connects to 
            # trusses, we must lock its rotation DOF to prevent a singular matrix.
            if rz == 0 and len(node.connected_elements) > 0:
                only_trusses = all(isinstance(e, TrussElement) for e in node.connected_elements)
                if only_trusses:
                    rz = 1 
                    
            self.equation_map[n_id] = [rx, ry, rz]
            
        eq_num = 1
        for n_id in sorted(self.nodes.keys()):
            for dof in range(3):
                if self.equation_map[n_id][dof] == 0:  
                    self.equation_map[n_id][dof] = eq_num
                    eq_num += 1
                else:  
                    self.equation_map[n_id][dof] = 0
        self.num_equations = eq_num - 1

    def assemble_matrices(self):
        self.K_global_struct = SparseMatrix(self.num_equations)
        for elem_id, elem in self.elements.items():
            if isinstance(elem, Member):
                k_loc = elem.get_local_stiffness()
                R = elem.get_rotation_matrix()
                T = elem.get_global_offset_transformation()
                
                # K_node = T^T * R^T * K_loc * R * T
                k_glob_phys = R.transpose().multiply(k_loc).multiply(R)
                k_node = T.transpose().multiply(k_glob_phys).multiply(T)
                
                G = self.equation_map[elem.start_node.node_id] + self.equation_map[elem.end_node.node_id]
                for p in range(6):
                    for q in range(6):
                        P, Q = G[p], G[q]
                        if P != 0 and Q != 0:  
                            self.K_global_struct.add_val(P - 1, Q - 1, k_node.get_val(p, q))

    def solve_system(self):
        self.load_vector = [0.0] * self.num_equations
        
        for n_id, node in self.nodes.items():
            for load in node.nodal_loads:
                dofs = self.equation_map[n_id] 
                forces = [load.fx, load.fy, load.mz]
                for q in range(3):
                    if dofs[q] != 0: self.load_vector[dofs[q] - 1] += forces[q]
                    
        for elem in self.elements.values():
            if isinstance(elem, Member) and elem.member_loads:
                fef_local = elem.get_fixed_end_forces()
                enl_local = [-f for f in fef_local]
                
                R = elem.get_rotation_matrix()
                T = elem.get_global_offset_transformation()
                
                # Convert to physical global, then to nodal global
                enl_glob_phys = R.transpose().multiply(enl_local)
                enl_node = T.transpose().multiply(enl_glob_phys)
                
                G = self.equation_map[elem.start_node.node_id] + self.equation_map[elem.end_node.node_id]
                for q in range(6):
                    if G[q] != 0: self.load_vector[G[q] - 1] += enl_node[q]

        for elem in self.elements.values():
            if isinstance(elem, Member):
                d_known_global = [0.0] * 6
                sn, en = elem.start_node, elem.end_node
                
                if sn.support:
                    d_known_global[0] = sn.support.settlement_dx if sn.support.rx else 0.0
                    d_known_global[1] = sn.support.settlement_dy if sn.support.ry else 0.0
                    d_known_global[2] = sn.support.settlement_rz if sn.support.rz else 0.0
                if en.support:
                    d_known_global[3] = en.support.settlement_dx if en.support.rx else 0.0
                    d_known_global[4] = en.support.settlement_dy if en.support.ry else 0.0
                    d_known_global[5] = en.support.settlement_rz if en.support.rz else 0.0

                if any(d != 0.0 for d in d_known_global):
                    R = elem.get_rotation_matrix()
                    T = elem.get_global_offset_transformation()
                    k_loc = elem.get_local_stiffness()
                    
                    # f_settlement = K_node * d_known
                    k_node = T.transpose().multiply(R.transpose().multiply(k_loc).multiply(R)).multiply(T)
                    f_settlement_node = k_node.multiply(d_known_global)

                    G = self.equation_map[sn.node_id] + self.equation_map[en.node_id]
                    for q in range(6):
                        if G[q] != 0: self.load_vector[G[q] - 1] -= f_settlement_node[q]
        try:
            self.displacements = self.K_global_struct.solve(self.load_vector)
        except ValueError as e:
            if "Zero pivot" in str(e): raise ValueError("Structural validation failed: Kinematic instability or collapse mechanism detected.")
            else: raise e

    def calculate_internal_forces(self):
        for elem_id, elem in self.elements.items():
            if isinstance(elem, Member):
                G = self.equation_map[elem.start_node.node_id] + self.equation_map[elem.end_node.node_id]
                d_global_node = [0.0] * 6
                
                nodes = [elem.start_node, elem.start_node, elem.start_node, elem.end_node, elem.end_node, elem.end_node]
                dofs = [0, 1, 2, 0, 1, 2]
                
                for i in range(6):
                    if G[i] != 0:
                        d_global_node[i] = self.displacements[G[i] - 1]
                    else:
                        support = nodes[i].support
                        if support:
                            if dofs[i] == 0: d_global_node[i] = support.settlement_dx
                            elif dofs[i] == 1: d_global_node[i] = support.settlement_dy
                            elif dofs[i] == 2: d_global_node[i] = support.settlement_rz
                
                R = elem.get_rotation_matrix()
                T = elem.get_global_offset_transformation()
                
                # Transform Nodal Disp -> Physical Disp -> Local Disp
                d_phys_global = T.multiply(d_global_node)
                d_local = R.multiply(d_phys_global)
                
                k_loc = elem.get_local_stiffness()
                forces = k_loc.multiply(d_local)
                fef = elem.get_fixed_end_forces()
                
                forces = [forces[i] + fef[i] for i in range(6)]
                self.member_forces[elem_id] = forces

    def calculate_reactions(self):
        self.reactions = {n_id: [0.0, 0.0, 0.0] for n_id, n in self.nodes.items() if n.support}
        
        for elem_id, elem in self.elements.items():
            if isinstance(elem, Member):
                R = elem.get_rotation_matrix()
                T = elem.get_global_offset_transformation()
                
                # Transform internal forces to Physical Global, then to Nodal Global
                f_global_phys = R.transpose().multiply(self.member_forces[elem_id])
                f_global_node = T.transpose().multiply(f_global_phys)
                
                sn_id = elem.start_node.node_id
                en_id = elem.end_node.node_id
                
                if sn_id in self.reactions:
                    self.reactions[sn_id][0] += f_global_node[0]
                    self.reactions[sn_id][1] += f_global_node[1]
                    self.reactions[sn_id][2] += f_global_node[2]
                if en_id in self.reactions:
                    self.reactions[en_id][0] += f_global_node[3]
                    self.reactions[en_id][1] += f_global_node[4]
                    self.reactions[en_id][2] += f_global_node[5]