from sys import has_accelerator
from sys.info import sizeof
from gpu import block_dim, block_idx, thread_idx, grid_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace, load
from math import ceildiv, sin, cos, sqrt
from time import monotonic
from memory import stack_allocation
from collections import List
from python import Python
from utils.numerics import max_finite

alias NUM_ITER = 10
alias NUM_POSES = 65536
alias WG_SIZE = 64      # Work group size
alias PPWI = 4          # Poses per work item

alias Zero = 0.0
alias Quarter = 0.25
alias Half = 0.5
alias One = 1.0
alias Two = 2.0
alias Four = 4.0
alias Cnstnt = 45.0

alias HBTYPE_F = 70
alias HBTYPE_E = 69
alias HARDNESS = 38.0
alias NPNPDIST = 5.5
alias NPPDIST = 1.0

alias dtype = DType.float32
alias FloatMax = max_finite[dtype]()

struct Vec3f32(Copyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32

    fn __init__(out self, x: Float32, y:Float32, z:Float32):
        self.x = x
        self.y = y
        self.z = z

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y
        self.z = other.z

    fn __moveinit__(out self, owned existing: Self):
        self.x = existing.x
        self.y = existing.y
        self.z = existing.z

struct Vec4f32(Copyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    fn __init__(out self, x: Float32, y:Float32, z:Float32, w:Float32):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.w = other.w

    fn __moveinit__(out self, owned existing: Self):
        self.x = existing.x
        self.y = existing.y
        self.z = existing.z
        self.w = existing.w

struct Atom(Copyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var type: Int32

    fn __init__(out self, x: Float32, y: Float32, z:Float32, type: Int32):
        self.x = x
        self.y = y
        self.z = z
        self.type = type

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.type = other.type

    fn __moveinit__(out self, owned existing: Self):
        self.x = existing.x
        self.y = existing.y
        self.z = existing.z
        self.type = existing.type

    @staticmethod
    fn read_atoms(path: String) raises -> List[Atom]:
        var file = open(path, "rb")
        var bytes = file.read_bytes()
        file.close()

        var ptr = bytes.unsafe_ptr().bitcast[UInt8]()
        var atoms = List[Atom]()
        var atom_size = 16      # 3 Float32s + 1 Int = 16 bytes

        var byte_count = len(bytes)
        var total = byte_count // atom_size

        for i in range(total):
            var offset = i * atom_size
            var x = (ptr + offset + 0).bitcast[Float32]()[0]
            var y = (ptr + offset + 4).bitcast[Float32]()[0]
            var z = (ptr + offset + 8).bitcast[Float32]()[0]
            var t = (ptr + offset + 12).bitcast[Int32]()[0]
            atoms.append(Atom(x, y, z, t))
        return atoms

struct FFParams(Copyable, Movable):
    var hbtype: Int32
    var radius: Float32
    var hphb: Float32
    var elsc: Float32

    fn __init__(out self, hbtype: Int32, radius: Float32, hphb: Float32, elsc: Float32):
        self.hbtype = hbtype
        self.radius = radius
        self.hphb = hphb
        self.elsc = elsc

    fn __copyinit__(out self, other: Self):
        self.hbtype = other.hbtype
        self.radius = other.radius
        self.hphb = other.hphb
        self.elsc = other.elsc

    fn __moveinit__(out self, owned existing: Self):
        self.hbtype = existing.hbtype
        self.radius = existing.radius
        self.hphb = existing.hphb
        self.elsc = existing.elsc

    @staticmethod
    fn read_ffparams(path: String) raises -> List[FFParams]:
        var file = open(path, "rb")
        var bytes = file.read_bytes()
        file.close()

        var ptr = bytes.unsafe_ptr().bitcast[UInt8]()
        var ffparams = List[FFParams]()
        var struct_size = 16

        var byte_count = len(bytes)
        var total = byte_count // struct_size

        for i in range(total):
            var offset = i * struct_size
            var hbtype = (ptr + offset + 0).bitcast[Int32]()[0]
            var radius = (ptr + offset + 4).bitcast[Float32]()[0]
            var hphb = (ptr + offset + 8).bitcast[Float32]()[0]
            var elsc = (ptr + offset + 12).bitcast[Float32]()[0]
            ffparams.append(FFParams(hbtype, radius, hphb, elsc))
        return ffparams

fn read_poses(path: String) raises -> List[List[Float32]]:
    var file = open(path, "rb")
    var bytes = file.read_bytes()
    file.close()

    var ptr = bytes.unsafe_ptr().bitcast[Float32]()
    var total_floats = len(bytes) // 4
    if total_floats % 6 != 0:
        raise Error("Pose size (", total_floats, ") not divisible by 6")

    var num_poses = total_floats // 6
    if not num_poses == NUM_POSES:
        raise Error("Number of poses", num_poses, "doesn't match the expected:", NUM_POSES)

    var poses = List[List[Float32]]()
    for i in range(6):
        var component = List[Float32](capacity=NUM_POSES)
        for j in range(NUM_POSES):
            component.append(ptr[i * num_poses + j])
        poses.append(component)
    return poses

struct Params:
    var num_poses: Int
    var iterations: Int
    var wgsize: Int
    var ppwi: Int
    var deck: String

    fn __init__(out self,
                num_poses: Int = NUM_POSES,
                iterations: Int = NUM_ITER,
                wgsize: Int = WG_SIZE,
                ppwi: Int = PPWI,
                deck: String = "data"):
        self.num_poses = num_poses
        self.iterations = iterations
        self.wgsize = wgsize
        self.ppwi = ppwi
        self.deck = deck

@fieldwise_init
struct Deck:
    var protein: List[Atom]
    var ligand: List[Atom]
    var forcefield: List[FFParams]
    var poses: List[List[Float32]]

fn fasten_kernel[PPWI: Int](natlig: Int, natpro: Int,
                            protein_molecule: UnsafePointer[Float32],
                            ligand_molecule: UnsafePointer[Float32],
                            transforms_0: UnsafePointer[Float32],
                            transforms_1: UnsafePointer[Float32],
                            transforms_2: UnsafePointer[Float32],
                            transforms_3: UnsafePointer[Float32],
                            transforms_4: UnsafePointer[Float32],
                            transforms_5: UnsafePointer[Float32],
                            etotals: UnsafePointer[Float32],
                            global_forcefield: UnsafePointer[Float32],
                            num_transforms: Int):
    var ix = block_idx.x * block_dim.x * PPWI + thread_idx.x
    if ix >= num_transforms:
        ix = num_transforms - PPWI

    # Compute transformation matrix to private memory
    var etot = SIMD[dtype, PPWI]()
    var transform = InlineArray[Vec4f32, PPWI * 3](uninitialized=True)

    var lsz = block_dim.x
    for i in range(PPWI):
        var index = ix + i * lsz

        sx: Float32 = sin(transforms_0[index])
        cx: Float32 = cos(transforms_0[index])
        sy: Float32 = sin(transforms_1[index])
        cy: Float32 = cos(transforms_1[index])
        sz: Float32 = sin(transforms_2[index])
        cz: Float32 = cos(transforms_2[index])

        transform[i * 3].x = cy * cz
        transform[i * 3].y = sx * sy * cz - cx * sz
        transform[i * 3].z = cx * sy * cz + sx * sz
        transform[i * 3].w = transforms_3[index]
        transform[i * 3 + 1].x = cy * sz
        transform[i * 3 + 1].y = sx * sy * sz + cx * cz
        transform[i * 3 + 1].z = cx * sy * sz - sx * cz
        transform[i * 3 + 1].w = transforms_4[index]
        transform[i * 3 + 2].x = -sy
        transform[i * 3 + 2].y = sx * cy
        transform[i * 3 + 2].z = cx * cy
        transform[i * 3 + 2].w = transforms_5[index]

        etot[i] = Zero

    # Loop over ligand atoms
    il = 0
    while (True):
        # Load ligand atom data
        var l_atom_x = ligand_molecule[il * 4]
        var l_atom_y = ligand_molecule[il * 4 + 1]
        var l_atom_z = ligand_molecule[il * 4 + 2]
        var l_atom_type = Int32(ligand_molecule[il * 4 + 3])

        var l_offset = l_atom_type * 4
        var l_params_hbtype = Int32(global_forcefield[l_offset])
        var l_params_radius = global_forcefield[l_offset + 1]
        var l_params_hphb = global_forcefield[l_offset + 2]
        var l_params_elsc = global_forcefield[l_offset + 3]

        var lhphb_ltz = l_params_hphb < Zero
        var lhphb_gtz = l_params_hphb > Zero

        var lpos = InlineArray[Vec3f32, PPWI](uninitialized=True)
        var linitpos = Vec4f32(l_atom_x, l_atom_y, l_atom_z, One)
        for i in range(PPWI):
            t0 = transform[i * 3]
            t1 = transform[i * 3 + 1]
            t2 = transform[i * 3 + 2]
            lpos[i].x = t0.w + linitpos.x * t0.x + linitpos.y * t0.y + linitpos.z * t0.z
            lpos[i].y = t1.w + linitpos.x * t1.x + linitpos.y * t1.y + linitpos.z * t1.z
            lpos[i].z = t2.w + linitpos.x * t2.x + linitpos.y * t2.y + linitpos.z * t2.z

        # Loop over protein atoms
        ip = 0
        while True:
            # Load protein atom data
            var p_atom_x = protein_molecule[ip * 4]
            var p_atom_y = protein_molecule[ip * 4 + 1]
            var p_atom_z = protein_molecule[ip * 4 + 2]
            var p_atom_type = Int32(protein_molecule[ip * 4 + 3])

            var p_offset = p_atom_type * 4
            var p_params_hbtype = Int32(global_forcefield[p_offset])
            var p_params_radius = global_forcefield[p_offset + 1]
            var p_params_hphb = global_forcefield[p_offset + 2]
            var p_params_elsc = global_forcefield[p_offset + 3]

            var radij = p_params_radius + l_params_radius
            var r_radij = 1.0 / radij

            var elcdst: Float32
            if p_params_hbtype == HBTYPE_F and l_params_hbtype == HBTYPE_F:
                elcdst = Four
            else:
                elcdst = Two

            var elcdst1: Float32
            if p_params_hbtype == HBTYPE_F and l_params_hbtype == HBTYPE_F:
                elcdst1 = Quarter
            else:
                elcdst1 = Half

            var type_E = p_params_hbtype == HBTYPE_E or l_params_hbtype == HBTYPE_E

            var phphb_ltz = p_params_hphb < Zero
            var phphb_gtz = p_params_hphb > Zero
            var phphb_nz = p_params_hphb != Zero

            var p_hphb = p_params_hphb
            if phphb_ltz and lhphb_gtz:
                p_hphb *= -One
            else:
                p_hphb *= One

            var l_hphb = l_params_hphb
            if phphb_gtz and lhphb_ltz:
                p_hphb *= -One
            else:
                p_hphb *= One

            var distdslv: Float32
            if phphb_ltz:
                if lhphb_ltz:
                    distdslv = NPNPDIST
                else:
                    distdslv = NPPDIST
            else:
                if lhphb_ltz:
                    distdslv = NPPDIST
                else:
                    distdslv = -FloatMax

            var r_distdslv = 1.0 / distdslv
            var chrg_init = l_params_elsc * p_params_elsc
            var dslv_init = p_hphb + l_hphb

            for i in range(PPWI):
                var x = lpos[i].x - p_atom_x
                var y = lpos[i].y - p_atom_y
                var z = lpos[i].z - p_atom_z
                var distij = sqrt(x * x + y * y + z * z)
                print(distij)

                # Calculate the sum of the sphere radii
                var distbb = distij - radij
                var zone1 = distbb < Zero

                # Calculate steric energy
                tmp = One - distij * r_radij
                if zone1:
                    tmp *= Two * HARDNESS
                else:
                    tmp *= Zero
                etot[i] += tmp

                # Calculate formal and dipole charge interactions
                var f1: Float32
                if zone1:
                    f1 = One
                else:
                    f1 = One - distbb * elcdst1
                var f2: Float32
                if distbb < elcdst:
                    f2 = One
                else:
                    f2 = Zero
                var chrg_e = chrg_init * f1 * f2
                var neg_chrg_e = -abs(chrg_e)
                if type_E:
                    chrg_e = neg_chrg_e
                else:
                    chrg_e = chrg_e
                etot[i] += chrg_e * Cnstnt

                # Calculate the two cases for Nonpolar-Polar repulsive interactions
                var coeff = One - distbb * r_distdslv
                var dslv_e = dslv_init
                if distbb < distdslv and phphb_nz:
                    dslv_e *= One
                else:
                    dslv_e *= Zero

                if zone1:
                    dslv_e *= One
                else:
                    dslv_e *= coeff

                etot[i] += dslv_e

            ip += 1
            if ip >= natpro:
                break
        il += 1
        if il >= natlig:
            break

    # Write results
    var td_base = block_idx.x * block_dim.x * PPWI + thread_idx.x
    if td_base < num_transforms:
        for i in range(PPWI):
            etotals[td_base + i * block_dim.x] = etot[i] * Half
            # print(etot[i])


def main():
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("GPU:", ctx.name())

        var params = Params()

        var protein = Atom.read_atoms(params.deck + "/protein.in")
        var ligand = Atom.read_atoms(params.deck + "/ligand.in")
        var forcefield = FFParams.read_ffparams(params.deck + "/forcefield.in")
        var poses = read_poses(params.deck + "/poses.in")

        var deck = Deck(protein, ligand, forcefield, poses)

        print("Poses     : ", len(deck.poses[0]))
        print("Iterations: ", params.iterations)
        print("Ligands   : ", len(deck.ligand))
        print("Protein   : ", len(deck.protein))
        print("Forcefield: ", len(deck.forcefield))
        print("Deck      : ", params.deck)
        print("WGsize    : ", params.wgsize)
        print("PPWI      : ", params.ppwi)

        d_etotals = ctx.enqueue_create_buffer[dtype](len(deck.poses[1]))

        protein_flat = ctx.enqueue_create_host_buffer[dtype](len(protein) * 4)
        var elements_per_atom = 4
        for i in range(len(protein)):
            protein_flat[i * elements_per_atom] = protein[i].x
            protein_flat[i * elements_per_atom + 1] = protein[i].y
            protein_flat[i * elements_per_atom + 2] = protein[i].z
            protein_flat[i * elements_per_atom + 3] = Float32(protein[i].type)
        d_protein = ctx.enqueue_create_buffer[dtype](len(protein_flat))
        ctx.enqueue_copy(dst_buf=d_protein, src_buf=protein_flat)

        ligand_flat = ctx.enqueue_create_host_buffer[dtype](len(ligand) * 4)
        for i in range(len(ligand)):
            ligand_flat[i * elements_per_atom] = ligand[i].x
            ligand_flat[i * elements_per_atom + 1] = ligand[i].y
            ligand_flat[i * elements_per_atom + 2] = ligand[i].z
            ligand_flat[i * elements_per_atom + 3] = Float32(ligand[i].type)
        d_ligand = ctx.enqueue_create_buffer[dtype](len(ligand_flat))
        ctx.enqueue_copy(dst_buf=d_ligand, src_buf=ligand_flat)
        # print(ligand_flat[0], ligand_flat[4], ligand_flat[8])

        forcefield_flat = ctx.enqueue_create_host_buffer[dtype](len(forcefield) * 4)
        for i in range(len(forcefield)):
            forcefield_flat[i * sizeof[FFParams]()] = Float32(forcefield[i].hbtype)
            forcefield_flat[i * sizeof[FFParams]() + 1] = forcefield[i].radius
            forcefield_flat[i * sizeof[FFParams]() + 2] = forcefield[i].hphb
            forcefield_flat[i * sizeof[FFParams]() + 3] = forcefield[i].elsc
        d_forcefield = ctx.enqueue_create_buffer[dtype](len(forcefield_flat))
        ctx.enqueue_copy(dst_buf=d_forcefield, src_buf=forcefield_flat)

        transforms = ctx.enqueue_create_host_buffer[dtype](NUM_POSES * 6)
        for i in range(6):
            for j in range(NUM_POSES):
                transforms[NUM_POSES * i + j] = deck.poses[i][j]
        transforms_0 = transforms.unsafe_ptr()
        transforms_1 = transforms_0 + NUM_POSES
        transforms_2 = transforms_1 + NUM_POSES
        transforms_3 = transforms_2 + NUM_POSES
        transforms_4 = transforms_3 + NUM_POSES
        transforms_5 = transforms_4 + NUM_POSES

        var block_size = params.wgsize
        var num_blocks = ceildiv(params.num_poses, params.ppwi)
        num_blocks = ceildiv(num_blocks, block_size)

        ctx.enqueue_function[fasten_kernel[PPWI]](len(deck.ligand), len(deck.protein),
                                                  d_protein.unsafe_ptr(),
                                                  d_ligand.unsafe_ptr(),
                                                  transforms_0,
                                                  transforms_1,
                                                  transforms_2,
                                                  transforms_3,
                                                  transforms_4,
                                                  transforms_5,
                                                  d_etotals.unsafe_ptr(),
                                                  d_forcefield.unsafe_ptr(),
                                                  params.num_poses,
                                                  grid_dim = (num_blocks, 1, 1),
                                                  block_dim = (block_size, 1, 1))
        ctx.synchronize()
        # with d_etotals.map_to_host() as result:
        #     for i in range(len(result)):
        #         print(result[i])