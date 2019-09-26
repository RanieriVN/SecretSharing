from ncpol2sdpa import *
from qutip import *
import numpy as np
import itertools as it
import time
import os
from datetime import datetime as dt

################################################################################################################################
# Identifying the current run in the day
cur_date = dt.today()
cur_run = 1

val = True
new_dir = False
while not new_dir:
    out_folder = os.path.abspath(os.path.join("Results","SingleIn",cur_date.strftime("%m%d") + "_Run" + str(cur_run)))

    if not os.path.exists(out_folder):
        new_dir = True
    else:
        cur_run +=1

################################################################################################################################
# Parameters for the run and the problem
NPA_level = 2
verbose = 0
cur_solver = "mosek"

in_list  = q_in_Alice, q_in_Bob, q_in_Charlie = 2, 2, 2
out_list = q_out_Eve, q_out_Alice, q_out_Bob, q_out_Charlie = 2,2,2,2

################################################################################################################################
# Functions
def genSeq(num_list_in):
    """Generates the sequence itertools.product(range(num_list_in[0]),range(num_list_in[1]),...)"""
    return it.product(*[range(n) for n in num_list_in])

def scn_in(q_inputs,parties=['Eve','Alice','Bob','Charlie'],x=0,y=0,z=0):
    """Generates the sequence of inputs pertaining to the parties specified"""
    part_dict = {'Eve':0, 'Alice':1, 'Bob':2, 'Charlie':3}
    p_seq = list(set([part_dict[p] for p in parties]))
    p_seq.sort()
    
    Yv = (x,x,x+q_inputs[0]*y,x+q_inputs[0]*z)
    return [Yv[p] for p in p_seq]

def genCorrelatorIneq(corr_seq = [1,1,1,1,1,1,1,1]):
    Bell = np.ones((2,2,2,2,2,2))

    Bell[1,:,:,:,:,:] *= -1
    Bell[:,1,:,:,:,:] *= -1
    Bell[:,:,1,:,:,:] *= -1
    
    for i,m in enumerate(corr_seq):
        if not m==1:
            b_z = i%2
            b_y = (i//2)%2
            b_x = (i//2)//2
            
            # Varying z quickest and x slowest
            Bell[:,:,:,b_x,b_y,b_z] *= m
    return Bell

def get_basic_monomials(ops_list,q_rem):
    q_parties = len(ops_list)

    ops_out = []
    if q_rem == 0:
        for j in range(q_parties):
            for ops in ops_list[j]:
                ops_out.append(ops)
        return ops_out        
    
    for j in range(q_parties):
        for ops in ops_list[j]:
            if ops_list[j+1:] == []:
                break
            else:
                ops_sub = get_basic_monomials(ops_list[j+1:],q_rem-1)
                for ops2 in ops_sub:
                    ops_out.append(ops*ops2)
    
    return ops_out

################################################################################################################################
# Setting up the scenario:
# Parties are Alice, Bob and Charlie. Alice is subdivided into two parts where one behaves as Eve
# Unbeknownst to the other parties, Alice's input interferes with Bob's and Charlie's devices

Eve_set     = [q_out_Eve]*q_in_Alice
Alice_set   = [q_out_Alice]*q_in_Alice
Bob_set     = [q_out_Bob]*q_in_Alice*q_in_Bob
Charlie_set = [q_out_Charlie]*q_in_Alice*q_in_Charlie

ProbQ = Probability(Eve_set, Alice_set, Bob_set, Charlie_set)

# Coefficients for Svetlichny's inequality
coeff_corr = [-1,1,1,1,1,1,1,-1]
ineq_str = "Svetlichny's"
Bell = genCorrelatorIneq(coeff_corr)

#===============================================================================================================================
# Objective function: Guessing probability
#===============================================================================================================================

obj_fun = 0
for e in range(q_out_Eve):
    obj_fun -= ProbQ([e,e],scn_in(in_list,parties=['Eve','Charlie'],x=0,z=0),('A','D'))

#===============================================================================================================================
# Equality constraints
#===============================================================================================================================
# Fine-tuned no-signaling constraints: 
# This is a special constraint particular to the problem at hand; the observable distribution (w/o Eve) must seem like a secure, 
# undisturbed no-signaling distribution, therefore tracing out Eve and Alice should leave the distribution independent of x. 
#
# Since only distributions of the form P(b,c|(x,y),(x,z)) are accessible in this scenario, the restriction is applied to this
# marginal and not to P(a,b,c|x,(x',y),(x'',z))
eqs_ns = []
for b,c,y,z in genSeq([q_out_Bob, q_out_Charlie, q_in_Bob, q_in_Charlie]):
    # Reference value for the marginal
    PMarg_ref = ProbQ([b,c],scn_in(in_list,parties=['Bob','Charlie'],x=0,y=y,z=z),('C','D'))
    
    for x in range(q_in_Alice):
        # Marginal distribution for other values of x on Bob and Charlie's side
        PMarg = ProbQ([b,c],scn_in(in_list,parties=['Bob','Charlie'],x=x,y=y,z=z),('C','D'))
        eqs_ns += [PMarg - PMarg_ref]

# Svetlichny's expression evaluated on the probability, later set to specific values of violation
ineq_viol = 0
for a,b,c in genSeq(out_list[1:]):
    for x,y,z in genSeq(in_list):
        ineq_viol += Bell[a,b,c,x,y,z]*ProbQ([a,b,c],scn_in(in_list,parties=['Alice','Bob','Charlie'],x=x,y=y,z=z),('B','C','D'))

# Secret sharing condition
eq_ss = []
for a,b,c in genSeq(out_list[1:]):
    if (a+b+c)%2 == 0:
        eq_ss += [ProbQ([a,b,c],[0,0,0],('B','C','D'))]

# Symmetric reduced distribution
eq_sym = []
for a,b,c,x,y,z in genSeq(out_list[1:]+in_list):
    if not ((x==y) and (a==b)):
        eq_sym += [ProbQ([a,b,c],[x,x+q_in_Alice*y,x+q_in_Alice*z],('B','C','D'))-ProbQ([b,a,c],[y,y+q_in_Alice*x,y+q_in_Alice*z],('B','C','D'))]
        
        
#===============================================================================================================================
# Inequality constraints
#===============================================================================================================================
# Nonnegativity has to be enforced by hand here
trivial_ineqs = []
for e,a,b,c,xe,xa,big_y,big_z in genSeq(out_list+(q_in_Alice,q_in_Alice,q_in_Alice*q_in_Bob,q_in_Alice*q_in_Charlie)):
    trivial_ineqs += [ProbQ([e,a,b,c],[xe,xa,big_y,big_z])]


################################################################################################################################
# SDP BLOCK

# Preparing the output directory
if not dt.today().strftime("%m%d") == cur_date.strftime("%m%d"):
    cur_date = dt.today()
    cur_run = 1

new_dir = False
while not new_dir:
    out_folder = os.path.abspath(os.path.join("Results","SingleIn",cur_date.strftime("%m%d") + "_Run" + str(cur_run)))

    try:
        os.makedirs(out_folder)
        new_dir = True
    except FileExistsError:
        cur_run += 1
        print("Directory {} already exists.".format(os.path.normpath(out_folder)))
        new_dir = False

################################################################################################################################
# Running the sdp

# Local parameters for the run
q_pts_sub = 5
q_pts_sup = 16
q_pts = q_pts_sub + 2*q_pts_sup

viol = np.hstack([np.linspace(0,4.75,q_pts_sub),np.linspace(4.751,5.25,q_pts_sup),np.linspace(5.251,5.4,q_pts_sup)])

# Observations passed to the output file
obs_str = ["Using secret-sharing condition"]
obs_str += ["Enforcing trivial nonnegativity inequalities"]
obs_str += ["Using moment equalities instead of trivial equalities for no-signaling constraints and secret sharing"]
obs_str += ["Computing relaxation in parallel, using extra monomials ABC, ABD, BCD and ABCD"]

# Initiate the SDP for different values of the violation
sdp_guess = [SdpRelaxation(ProbQ.get_all_operators(),verbose=verbose,parallel=True) for i in range(q_pts)]

###############################################################################################################################
p_opt = np.zeros((q_pts,2))
st_out = []

relax_time = np.zeros((q_pts,))
solve_time = np.zeros((q_pts,))

BigTic = time.time()
for j in range(q_pts):
    
    tic = time.perf_counter()
    
    eq_viol = [ineq_viol - viol[j]]
    
    sdp_guess[j].get_relaxation(NPA_level, objective = obj_fun,\
                                equalities = eq_viol,\
                                momentequalities = eqs_ns+eq_ss,\
                                inequalities = trivial_ineqs,\
                                substitutions = ProbQ.substitutions,\
                                extramonomials = ProbQ.get_extra_monomials('ABC','ABD','BCD','ABCD'))
    
    
    tac = time.perf_counter()
    relax_time[j] = tac-tic
    
    #########
    out_str = os.path.join(out_folder,"pt{0}_Relax_lv{1}_viol{2:.3f}.dat-s".format(j,NPA_level,viol[j]))
    sdp_guess[j].write_to_file(out_str)
    
    #########
    tic = time.perf_counter()
    sdp_guess[j].solve(cur_solver)
    tac = time.perf_counter()
    
    solve_time[j] = tac-tic

    #########
    sol_out = {}
    for k,xval in enumerate(sdp_guess[j].x_mat):
        sol_out['Primal_'+str(k)] = xval
        
    for k,xval in enumerate(sdp_guess[j].y_mat):
        sol_out['Dual_'+str(k)] = xval
    
    out_str = os.path.join(out_folder,"pt{0}_viol{1:.3f}_Solution".format(j,viol[j]))        
    np.savez(out_str, **sol_out)

    
    p_opt[j,0] = viol[j]
    p_opt[j,1] = -sdp_guess[j].primal
    st_out.append(sdp_guess[j].status)
    if not sdp_guess[j].status == "optimal":
        print(sdp_guess[j].status)
    
#########

BigTac = time.time()
total_time = BigTac - BigTic

################################################################################################################################
# Writing to the output file

with open(os.path.join(out_folder,"output_details.txt"),"w") as log:
    log.write("Solver used: {}\n".format(cur_solver))
    log.write("Tested {0} inequality for violations in the range {1}-{2}\n".format(ineq_str, viol[0], viol[-1]))

    log.write("Solver status for each objective:\n")
    for j,stats in enumerate(st_out):
        log.write("\tPoint x={0} - Violation {1:.3f}:\n".format(j,viol[j]))
        log.write("\t\tStatus: {}\n".format(stats))
        log.write("\t\tRelaxation time: {0:.2f} s\n".format(relax_time[j]))
        log.write("\t\tSolver time: {0:.2f} s\n\n".format(solve_time[j]))

    log.write("Total time elapsed: {0:.2f} s\n\n".format(total_time))

    try:
        if obs_str != []:
            log.write("Observations:\n")
            for obs in obs_str:
                log.write("\t" + obs + "\n")
    except NameError:
        obs_str = []

np.save(os.path.join(out_folder,"optim_result"),p_opt)
np.savetxt(os.path.join(out_folder,"optim_result.txt"), p_opt)

cur_run +=1


################################################################################################################################
# Retrieving the probabilities
ops = ProbQ.get_all_operators()

ops_A = [ops[0],ops[1]]
ops_B = [ops[2],ops[3]]
ops_C = [op for op in ops[4:8]]
ops_D = [op for op in ops[8:]]

ops_list = [ops_A, ops_B, ops_C, ops_D]

basic_monoms = []
for pty in range(NPA_level-1,-1,-1):
    basic_monoms += get_basic_monomials(ops_list,pty)

basic_monoms_row = get_basic_monomials(ops_list[:2],1)+get_basic_monomials(ops_list[:2],0)
basic_monoms_col = get_basic_monomials(ops_list[2:],1)+get_basic_monomials(ops_list[2:],0)

ProbFull_dict = {}

for mon in basic_monoms:
    m_idx = sdp_guess[0].monomial_index[mon]
    ProbFull_dict[mon] = [sdp_guess[j].x_mat[0][m_idx,0] for j in range(q_pts)]

for mon in basic_monoms_row:
    for mon2 in basic_monoms_col:
        m_idx = sdp_guess[0].monomial_index[mon]
        mc_idx = sdp_guess[0].monomial_index[mon2]
        ProbFull_dict[mon*mon2] = [sdp_guess[j].x_mat[0][m_idx,mc_idx] for j in range(q_pts)]

ProbFull = np.zeros(out_list+(q_in_Alice,q_in_Alice,q_in_Alice*q_in_Bob,q_in_Alice*q_in_Charlie,q_pts))

# I'm really not proud of what follows, but this is much faster than using sdp_guess[j][ProbQ(...)]
for xe,xa,xy,xz,j in genSeq([q_in_Alice,q_in_Alice,q_in_Alice*q_in_Bob,q_in_Alice*q_in_Charlie,q_pts]):
    ProbFull[0,0,0,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]
    
    # (1-A)BCD = BCD - ABCD; A(1-B)CD = ACD - ABCD; ...
    ProbFull[1,0,0,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0,0],[xa,xy,xz],('B','C','D'))][j] - ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]
    ProbFull[0,1,0,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0,0],[xe,xy,xz],('A','C','D'))][j] - ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]
    ProbFull[0,0,1,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0,0],[xe,xa,xz],('A','B','D'))][j] - ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]
    ProbFull[0,0,0,1,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0,0],[xe,xa,xy],('A','B','C'))][j] - ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]

    # (1-A)(1-B)CD = CD - ACD - BCD + ABCD
    ProbFull[1,1,0,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0],[xy,xz],('C','D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xy,xz],('A','C','D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xa,xy,xz],('B','C','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]

    # (1-A)B(1-C)D = BD - ABD - BCD + ABCD
    ProbFull[1,0,1,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0],[xa,xz],('B','D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xa,xz],('A','B','D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xa,xy,xz],('B','C','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]

    # (1-A)BC(1-D) = BC - ABC - BCD + ABCD
    ProbFull[1,0,0,1,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0],[xa,xy],('B','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xa,xy],('A','B','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xa,xy,xz],('B','C','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]

    # A(1-B)(1-C)D = AD - ABD - ACD + ABCD
    ProbFull[0,1,1,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0],[xe,xz],('A','D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xa,xz],('A','B','D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xy,xz],('A','C','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]

    # A(1-B)C(1-D) = AC - ABC - ACD  + ABCD
    ProbFull[0,1,0,1,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0],[xe,xy],('A','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xa,xy],('A','B','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xy,xz],('A','C','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]

    # AB(1-C)(1-D) = AB - ABC - ABD  + ABCD
    ProbFull[0,0,1,1,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0,0],[xe,xa],('A','B'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xa,xy],('A','B','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0,0],[xe,xa,xz],('A','B','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0,0],[xe,xa,xy,xz])][j]

    # (1-A)(1-B)(1-C)D = D - BD - CD + BCD - A(1-B)(1-C)D
    ProbFull[1,1,1,0,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0],[xz],('D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xa,xz],('B','D'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xy,xz],('C','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0],[xa,xy,xz],('B','C','D'))][j]\
                                      - ProbFull[0,1,1,0,xe,xa,xy,xz,j]
    
    # (1-A)(1-B)C(1-D) = C - BC - CD + BCD - A(1-B)C(1-D)
    ProbFull[1,1,0,1,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0],[xy],('C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xa,xy],('B','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xy,xz],('C','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0],[xa,xy,xz],('B','C','D'))][j]\
                                      - ProbFull[0,1,0,1,xe,xa,xy,xz,j]
    
    # (1-A)B(1-C)(1-D) = B - BC - BD + BCD - AB(1-C)(1-D)
    ProbFull[1,0,1,1,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0],[xa],('B'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xa,xy],('B','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xa,xz],('B','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0],[xa,xy,xz],('B','C','D'))][j]\
                                      - ProbFull[0,0,1,1,xe,xa,xy,xz,j]
    
    # A(1-B)(1-C)(1-D) = A(1-C)(1-D) - AB(1-C)(1-D) = A - AC - AD + ACD - AB(1-C)(1-D)
    ProbFull[0,1,1,1,xe,xa,xy,xz,j] = ProbFull_dict[ProbQ([0],[xe],('A'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xe,xy],('A','C'))][j]\
                                      - ProbFull_dict[ProbQ([0,0],[xe,xz],('A','D'))][j]\
                                      + ProbFull_dict[ProbQ([0,0,0],[xe,xy,xz],('A','C','D'))][j]\
                                      - ProbFull[0,0,1,1,xe,xa,xy,xz,j]

    ProbFull[1,1,1,1,xe,xa,xy,xz,j] = 1 - np.sum(ProbFull[:,:,:,:,xe,xa,xy,xz,j],(0,1,2,3))

np.save(os.path.join(out_folder,"Probabilities"),ProbFull)




