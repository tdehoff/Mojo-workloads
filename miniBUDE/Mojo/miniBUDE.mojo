from std.sys import has_accelerator, argv
from std.collections import List
from std.math import ceildiv, sin, cos, sqrt
from std.time import monotonic
from std.memory import stack_allocation
from std.utils.numerics import max_finite

from std.gpu import block_dim, block_idx, thread_idx, grid_dim
from std.gpu.host import DeviceContext

comptime NUM_ITER = 100
comptime NUM_POSES = 65536
comptime WG_SIZE = 64      # Work group size
# DEFAULT_PPWI 1, 2, 4, 8, 16, 32, 64, 128
comptime PPWI = 4         # Poses per work item
comptime DIFF_TOLERANCE_PCT = 0.025

comptime Zero = 0.0
comptime Quarter = 0.25
comptime Half = 0.5
comptime One = 1.0
comptime Two = 2.0
comptime Four = 4.0
comptime Cnstnt = 45.0

comptime HBTYPE_F = 70
comptime HBTYPE_E = 69
comptime HARDNESS = 38.0
comptime NPNPDIST = 5.5
comptime NPPDIST = 1.0

comptime dtype = DType.float32
comptime FloatMax = max_finite[dtype]()


struct Vec3f32(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32

    def __init__(out self, x: Float32, y: Float32, z: Float32):
        self.x = x
        self.y = y
        self.z = z

struct Vec4f32(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32

    def __init__(out self, x: Float32, y: Float32, z: Float32, w: Float32):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

struct Atom(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    var type: Int32

    def __init__(out self, x: Float32, y: Float32, z: Float32, type: Int32):
        self.x = x
        self.y = y
        self.z = z
        self.type = type

    @staticmethod
    def read_atoms(path: String) raises -> List[Atom]:
        var file = open(path, "r")
        var bytes = file.read_bytes()
        file.close()

        var ptr = bytes.unsafe_ptr().bitcast[UInt8]()
        var atoms = List[Atom]()
        var atom_size = 16      # Float32 x,y,z + Int32 type = 16 bytes

        var byte_count = len(bytes)
        var total = byte_count // atom_size

        for i in range(total):
            var offset = i * atom_size
            var x = (ptr + offset +  0).bitcast[Float32]()[0]
            var y = (ptr + offset +  4).bitcast[Float32]()[0]
            var z = (ptr + offset +  8).bitcast[Float32]()[0]
            var t = (ptr + offset + 12).bitcast[Int32]()[0]
            atoms.append(Atom(x, y, z, t))
        return atoms^

struct FFParams(ImplicitlyCopyable, Movable):
    var hbtype: Int32
    var radius: Float32
    var hphb: Float32
    var elsc: Float32

    def __init__(out self, hbtype: Int32, radius: Float32, hphb: Float32, elsc: Float32):
        self.hbtype = hbtype
        self.radius = radius
        self.hphb = hphb
        self.elsc = elsc

    @staticmethod
    def read_ffparams(path: String) raises -> List[FFParams]:
        var file = open(path, "r")
        var bytes = file.read_bytes()
        file.close()

        var ptr = bytes.unsafe_ptr().bitcast[UInt8]()
        var ffparams = List[FFParams]()
        var atom_size = 16      # 3 Float32s + 1 Int = 16 bytes

        var byte_count = len(bytes)
        var total = byte_count // struct_size

        for i in range(total):
            var offset = i * struct_size
            var hbtype = (ptr + offset +  0).bitcast[Int32]()[0]
            var radius = (ptr + offset +  4).bitcast[Float32]()[0]
            var hphb   = (ptr + offset +  8).bitcast[Float32]()[0]
            var elsc   = (ptr + offset + 12).bitcast[Float32]()[0]
            ffparams.append(FFParams(hbtype, radius, hphb, elsc))
        return ffparams^

def read_poses(path: String) raises -> List[List[Float32]]:
    var file = open(path, "r")
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
        poses.append(component^)
    return poses^

struct Params:
    var num_poses: Int
    var iterations: Int
    var wgsize: Int
    var ppwi: Int
    var deck: String

    def __init__(out self,
                num_poses: Int = NUM_POSES,
                iterations: Int = NUM_ITER,
                wgsize: Int = WG_SIZE,
                ppwi: Int = PPWI,
                deck: String = "../data/bm1"):
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

def fasten_kernel[PPWI: Int](natlig: Int, natpro: Int,
                             protein_molecule:  UnsafePointer[Atom,     MutAnyOrigin],
                             ligand_molecule:   UnsafePointer[Atom,     MutAnyOrigin],
                             transforms_0:      UnsafePointer[Float32,  MutAnyOrigin],
                             transforms_1:      UnsafePointer[Float32,  MutAnyOrigin],
                             transforms_2:      UnsafePointer[Float32,  MutAnyOrigin],
                             transforms_3:      UnsafePointer[Float32,  MutAnyOrigin],
                             transforms_4:      UnsafePointer[Float32,  MutAnyOrigin],
                             transforms_5:      UnsafePointer[Float32,  MutAnyOrigin],
                             etotals:           UnsafePointer[Float32,  MutAnyOrigin],
                             global_forcefield: UnsafePointer[FFParams, MutAnyOrigin],
                             num_transforms: Int):
    var ix = block_idx.x * block_dim.x * PPWI + thread_idx.x
    if ix >= num_transforms:
        ix = num_transforms - PPWI

    # Compute transformation matrix to private memory
    var etot = InlineArray[Float32, PPWI](fill=0)
    var transform = InlineArray[Vec4f32, PPWI * 3](uninitialized=True)

    var lsz = block_dim.x
    for i in range(PPWI):
        var index = ix + i * lsz

        var sx: Float32 = sin(transforms_0[index])
        var cx: Float32 = cos(transforms_0[index])
        var sy: Float32 = sin(transforms_1[index])
        var cy: Float32 = cos(transforms_1[index])
        var sz: Float32 = sin(transforms_2[index])
        var cz: Float32 = cos(transforms_2[index])

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
    var il = 0
    while True:
        var l_atom   = ligand_molecule[il]
        var l_params = global_forcefield[Int(l_atom.type)]
        var lhphb_ltz = l_params.hphb < Zero
        var lhphb_gtz = l_params.hphb > Zero

        var lpos = InlineArray[Vec3f32, PPWI](uninitialized=True)
        var linitpos = Vec4f32(l_atom.x, l_atom.y, l_atom.z, One)
        for i in range(PPWI):
            var t0 = transform[i * 3]
            var t1 = transform[i * 3 + 1]
            var t2 = transform[i * 3 + 2]
            lpos[i].x = t0.w + linitpos.x * t0.x + linitpos.y * t0.y + linitpos.z * t0.z
            lpos[i].y = t1.w + linitpos.x * t1.x + linitpos.y * t1.y + linitpos.z * t1.z
            lpos[i].z = t2.w + linitpos.x * t2.x + linitpos.y * t2.y + linitpos.z * t2.z

        # Loop over protein atoms
        var ip = 0
        while True:
            var p_atom   = protein_molecule[ip]
            var p_params = global_forcefield[Int(p_atom.type)]

            var radij   = p_params.radius + l_params.radius
            var r_radij = 1.0 / radij

            var elcdst: Float32
            if p_params.hbtype == HBTYPE_F and l_params.hbtype == HBTYPE_F:
                elcdst = Four
            else:
                elcdst = Two

            var elcdst1: Float32
            if p_params.hbtype == HBTYPE_F and l_params.hbtype == HBTYPE_F:
                elcdst1 = Quarter
            else:
                elcdst1 = Half

            var type_E    = p_params.hbtype == HBTYPE_E or l_params.hbtype == HBTYPE_E
            var phphb_ltz = p_params.hphb < Zero
            var phphb_gtz = p_params.hphb > Zero
            var phphb_nz  = p_params.hphb != Zero

            var p_hphb = p_params.hphb
            if phphb_ltz and lhphb_gtz:
                p_hphb *= -One
            else:
                p_hphb *= One

            var l_hphb = l_params.hphb
            if phphb_gtz and lhphb_ltz:
                l_hphb *= -One
            else:
                l_hphb *= One

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
            var chrg_init  = l_params.elsc * p_params.elsc
            var dslv_init  = p_hphb + l_hphb

            for i in range(PPWI):
                var x      = lpos[i].x - p_atom.x
                var y      = lpos[i].y - p_atom.y
                var z      = lpos[i].z - p_atom.z
                var distij = sqrt(x * x + y * y + z * z)
                var distbb = distij - radij
                var zone1  = distbb < Zero

                # Calculate steric energy
                var tmp = One - distij * r_radij
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
                var chrg_e     = chrg_init * f1 * f2
                var neg_chrg_e = -abs(chrg_e)
                if type_E:
                    chrg_e = neg_chrg_e
                else:
                    chrg_e = chrg_e
                etot[i] += chrg_e * Cnstnt

                # Calculate the two cases for Nonpolar-Polar repulsive interactions
                var coeff  = One - distbb * r_distdslv
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

def main() raises:
    args = argv()
    csv_output = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--csv":
            csv_output = True
        i += 1

    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()

        if not csv_output:
            print("GPU:", ctx.name())
            print("Driver:", ctx.get_api_version())

        var params = Params()

        var protein    = Atom.read_atoms(params.deck + "/protein.in")
        var ligand     = Atom.read_atoms(params.deck + "/ligand.in")
        var forcefield = FFParams.read_ffparams(params.deck + "/forcefield.in")
        var poses      = read_poses(params.deck + "/poses.in")

        var deck = Deck(protein^, ligand^, forcefield^, poses^)
        if not csv_output:
            print("Poses     : ", len(deck.poses[0]))
            print("Iterations: ", params.iterations)
            print("Ligands   : ", len(deck.ligand))
            print("Protein   : ", len(deck.protein))
            print("Forcefield: ", len(deck.forcefield))
            print("Deck      : ", params.deck)
            print("WGsize    : ", params.wgsize)
            print("PPWI      : ", params.ppwi)
            print("")

        d_etotals = ctx.enqueue_create_buffer[dtype](len(deck.poses[0]))

        protein_h = ctx.enqueue_create_host_buffer[dtype](len(deck.protein) * 4)
        var protein_h_ptr = protein_h.unsafe_ptr().bitcast[Atom]()
        for i in range(len(deck.protein)):
            protein_h_ptr[i] = deck.protein[i]
        d_protein_buf = ctx.enqueue_create_buffer[dtype](len(deck.protein) * 4)
        ctx.enqueue_copy(dst_buf=d_protein_buf, src_buf=protein_h)
        var d_protein = d_protein_buf.unsafe_ptr().bitcast[Atom]()

        ligand_h = ctx.enqueue_create_host_buffer[dtype](len(deck.ligand) * 4)
        var ligand_h_ptr = ligand_h.unsafe_ptr().bitcast[Atom]()
        for i in range(len(deck.ligand)):
            ligand_h_ptr[i] = deck.ligand[i]
        d_ligand_buf = ctx.enqueue_create_buffer[dtype](len(deck.ligand) * 4)
        ctx.enqueue_copy(dst_buf=d_ligand_buf, src_buf=ligand_h)
        var d_ligand = d_ligand_buf.unsafe_ptr().bitcast[Atom]()

        forcefield_h = ctx.enqueue_create_host_buffer[dtype](len(deck.forcefield) * 4)
        var ff_h_ptr = forcefield_h.unsafe_ptr().bitcast[FFParams]()
        for i in range(len(deck.forcefield)):
            ff_h_ptr[i] = deck.forcefield[i]
        d_ff_buf = ctx.enqueue_create_buffer[dtype](len(deck.forcefield) * 4)
        ctx.enqueue_copy(dst_buf=d_ff_buf, src_buf=forcefield_h)
        var d_forcefield = d_ff_buf.unsafe_ptr().bitcast[FFParams]()

        transforms_h = ctx.enqueue_create_host_buffer[dtype](NUM_POSES * 6)
        for i in range(6):
            for j in range(NUM_POSES):
                transforms_h[NUM_POSES * i + j] = deck.poses[i][j]
        d_transforms = ctx.enqueue_create_buffer[dtype](NUM_POSES * 6)
        ctx.enqueue_copy(dst_buf=d_transforms, src_buf=transforms_h)
        transforms_0 = d_transforms.unsafe_ptr()
        transforms_1 = transforms_0 + NUM_POSES
        transforms_2 = transforms_1 + NUM_POSES
        transforms_3 = transforms_2 + NUM_POSES
        transforms_4 = transforms_3 + NUM_POSES
        transforms_5 = transforms_4 + NUM_POSES

        var block_size = params.wgsize
        var num_blocks = ceildiv(params.num_poses, params.ppwi)
        num_blocks = ceildiv(num_blocks, block_size)
        # Warmup call
        comptime kernel = fasten_kernel[PPWI]
        ctx.enqueue_function[kernel, kernel](len(deck.ligand), len(deck.protein),
                                             d_protein, d_ligand,
                                             transforms_0, transforms_1, transforms_2,
                                             transforms_3, transforms_4, transforms_5,
                                             d_etotals.unsafe_ptr(), d_forcefield,
                                             params.num_poses,
                                             grid_dim = (num_blocks, 1, 1),
                                             block_dim = (block_size, 1, 1))
        ctx.synchronize()

        # Timing
        var kernel_times = List[Float64]()
        var total_elapsed: Float64 = 0.0

        for i in range(NUM_ITER):
            start = monotonic()
            ctx.enqueue_function[kernel, kernel](len(deck.ligand), len(deck.protein),
                                                 d_protein, d_ligand,
                                                 transforms_0, transforms_1, transforms_2,
                                                 transforms_3, transforms_4, transforms_5,
                                                 d_etotals.unsafe_ptr(), d_forcefield,
                                                 params.num_poses,
                                                 grid_dim = (num_blocks, 1, 1),
                                                 block_dim = (block_size, 1, 1))
            ctx.synchronize()
            end = monotonic()
            var elapsed = Float64(end - start)
            kernel_times.append(elapsed)
            total_elapsed += elapsed

        # Validate results
        etotals_h = ctx.enqueue_create_host_buffer[dtype](params.num_poses)
        ctx.enqueue_copy(dst_buf=etotals_h, src_buf=d_etotals)
        ctx.synchronize()

        # Load reference energies
        var ref_energies = List[Float32]()
        var ref_file = open(params.deck + "/ref_energies.out", "r")
        var ref_content = ref_file.read()
        ref_file.close()
        var ref_lines = ref_content.split("\n")
        for j in range(len(ref_lines)):
            if len(ref_energies) >= params.num_poses:
                break
            var line = ref_lines[j].strip()
            if line.byte_length() > 0:
                ref_energies.append(Float32(atof(line)))

        # Verify correctness
        var max_diff_pct: Float64 = 0.0
        var num_failed: Int = 0
        for i in range(params.num_poses):
            var ref_val = Float64(ref_energies[i])
            var com_val = Float64(etotals_h[i])
            # don't verify anything less than one
            if abs(ref_val) < 1.0 and abs(com_val) < 1.0:
                continue
            var diff_pct = abs(ref_val - com_val) / abs(ref_val) * 100.0
            if diff_pct > max_diff_pct:
                max_diff_pct = diff_pct
            if diff_pct > DIFF_TOLERANCE_PCT:
                num_failed += 1

        var valid = max_diff_pct < DIFF_TOLERANCE_PCT
        if not csv_output:
            if valid:
                print("Validation: PASS (max_diff_%:", max_diff_pct, ")")
            else:
                print("Validation: FAIL (max_diff_%:", max_diff_pct,
                      "failed:", num_failed, "/", params.num_poses, ")")

        if csv_output:
            # print("backend,GPU,ppwi,wgsize,sum_ms,avg_ms,min_ms,max_ms,stddev_ms,gflops/s")

            # Average time per iteration
            var ns      = total_elapsed / Float64(NUM_ITER)
            var runtime = ns * 1e-9

            # Compute FLOP/s
            var ops_per_wg = UInt32(PPWI * 27 + len(deck.ligand) * (2 + PPWI * 18 + len(deck.protein) * (10 + PPWI * 30)) + PPWI)
            var total_ops = Float64(ops_per_wg) * (Float64(NUM_POSES) / Float64(PPWI))
            var flops  = total_ops / runtime
            var gflops = flops / 1e9

            # Compute timing stats in ms
            var sum_ms: Float64 = 0.0
            var min_ms: Float64 = kernel_times[0] * 1e-6
            var max_ms: Float64 = kernel_times[0] * 1e-6
            for i in range(NUM_ITER):
                var t_ms = kernel_times[i] * 1e-6
                sum_ms += t_ms
                if t_ms < min_ms:
                    min_ms = t_ms
                if t_ms > max_ms:
                    max_ms = t_ms
            var avg_ms = sum_ms / Float64(NUM_ITER)
            var variance: Float64 = 0.0
            for i in range(NUM_ITER):
                var d = kernel_times[i] * 1e-6 - avg_ms
                variance += d * d
            variance /= Float64(NUM_ITER)
            var stddev_ms = sqrt(variance)

            print("Mojo,", ctx.name(), ",", PPWI, ",", WG_SIZE, ",",
                  sum_ms, ",", avg_ms, ",", min_ms, ",", max_ms, ",",
                  stddev_ms, ",", gflops)
        # else:
        #     # Print result
        #     with d_etotals.map_to_host() as result:
        #          for i in range(len(result)):
        #              print(result[i])
