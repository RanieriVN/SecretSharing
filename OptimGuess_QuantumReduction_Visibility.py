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

new_dir = False
while not new_dir:
    out_folder = os.path.abspath(os.path.join("Results","SingleIn",cur_date.strftime("%m%d") + "_Run" + str(cur_run)))

    if not os.path.exists(out_folder):
        new_dir = True
    else:
        cur_run +=1

cur_solver = "mosek"

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

def ProbRed(vis):
    # Variables used in the "self-test reduction"
    GHZ = ket2dm((tensor(basis(2,0),basis(2,0),basis(2,0)) + tensor(basis(2,1),basis(2,1),basis(2,1))).unit())
    Rho = vis*GHZ + (1-vis)*0.125*tensor(qeye(2),qeye(2),qeye(2))
    
    MA = [[0.5*(qeye(2)-a*(sigmax()+sigmay())/np.sqrt(2)) for a in (1,-1)],\
          [0.5*(qeye(2)+a*(sigmax()-sigmay())/np.sqrt(2)) for a in (1,-1)]]
    MB = [[0.5*(qeye(2)+b*sigmax()) for b in (1,-1)],\
          [0.5*(qeye(2)+b*sigmay()) for b in (1,-1)]]
    MC = [[0.5*(qeye(2)+b*sigmax()) for b in (1,-1)],\
          [0.5*(qeye(2)+b*sigmay()) for b in (1,-1)]]
    
    if vis<0:
        vis=0
    elif vis>1:
        vis=1
    
    ProbRedOut = np.zeros([2,2,2,2,2,2])
    for a,b,c,x,y,z in genSeq([2,2,2,2,2,2]):
        ProbRedOut[a,b,c,x,y,z] = (Rho*tensor(MA[x][a],MB[y][b],MC[z][c])).tr()
    
    return ProbRedOut

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
# Parameters for the run
NPA_level = 2
in_list  = q_in_Alice, q_in_Bob, q_in_Charlie = 2, 2, 2
out_list = q_out_Eve, q_out_Alice, q_out_Bob, q_out_Charlie = 2,2,2,2
verbose = 0

q_pts_A = 32
q_pts_B = 64
q_pts_C = 32

vis_list = np.hstack([np.linspace(0,0.7,q_pts_A),np.linspace(0.71,0.95,q_pts_B),np.linspace(0.951,1,q_pts_C)])

q_pts = q_pts_A + q_pts_B + q_pts_C

obs_str = ["Not using secret-sharing condition"]
obs_str += ['Enforcing trivial inequalities']
obs_str += ['Considering GHZ state with optimal measurements as the reduced distribution']
obs_str += ["Guessing Charlie's output for x=0,z=0, varying visibilities"]
obs_str += ["Using extra monomials ABC and ABCD"]

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
# Insurance
trivial_ineqs = []
for e,a,b,c,xe,xa,big_y,big_z in genSeq(out_list+(q_in_Alice,q_in_Alice,q_in_Alice*q_in_Bob,q_in_Alice*q_in_Charlie)):
    trivial_ineqs += [ProbQ([e,a,b,c],[xe,xa,big_y,big_z])]


################################################################################################################################
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

sdp_guess = [SdpRelaxation(ProbQ.get_all_operators(),verbose=verbose) for j in range(q_pts)]

p_opt = np.zeros((q_pts,2))
st_out = []

relax_time = np.zeros((q_pts,))
solve_time = np.zeros((q_pts,))

BigTic = time.time()
for j in range(q_pts):

    ProbST = ProbRed(vis_list[j])
    # Assuming that the restriction to an "Eveless scenario" equals to a reduced-visibility GHZ distribution
    eqs_GHZreduct = []
    for a,b,c,x,y,z in genSeq(out_list[1:]+in_list):
        eqs_GHZreduct += [ProbQ([a,b,c],scn_in(in_list,parties=['Alice','Bob','Charlie'],x=x,y=y,z=z),('B','C','D')) - ProbST[a,b,c,x,y,z]]
    
    tic = time.perf_counter()
    
    sdp_guess[j].get_relaxation(NPA_level, objective = obj_fun,\
                                momentequalities = eqs_GHZreduct,\
                                inequalities = trivial_ineqs,\
                                extramonomials = ProbQ.get_extra_monomials('ABC','ABCD'),\
                                substitutions = ProbQ.substitutions)
        
    tac = time.perf_counter()
    relax_time[j] = tac-tic
    
    #########
    
    out_str = os.path.join(out_folder,"pt{0}_vis{1:.3f}_SdpRelax_lv{2}.dat-s".format(j,vis_list[j],NPA_level))
    sdp_guess[j].write_to_file(out_str)    
    
    #########
    
    tic = time.perf_counter()
    sdp_guess[j].solve(cur_solver)
    tac = time.perf_counter()
    
    solve_time[j] = tac-tic
    
    #########
    
    sol_out = {}
    for k, xelm in enumerate(sdp_guess[j].x_mat):
        sol_out['Primal_'+str(k)] = xelm
    
    for k, xelm in enumerate(sdp_guess[j].y_mat):
        sol_out['Dual_'+str(k)] = xelm
        
    out_str = os.path.join(out_folder,"pt{0}_vis{1:.3f}_Solutions".format(j,vis_list[j]))
    np.savez(out_str,**sol_out)

    #########
    
    p_opt[j,0] = vis_list[j]
    p_opt[j,1] = -sdp_guess[j].primal
    st_out.append(sdp_guess[j].status)
    
#########

BigTac = time.time()
total_time = BigTac - BigTic

################################################################################################################################
# Writing to the output file
with open(os.path.join(out_folder,"output_details.txt"),"w") as log:
    log.write("Solver used: {}\n".format(cur_solver))
    log.write("NPA hierarchy level used: {}\n".format(NPA_level))

    log.write("Solver status for each objective:\n")
    for j,stats in enumerate(st_out):
        log.write("\tPoint {0} - Visibility {1:.3f}: {2}".format(j,vis_list[j],stats) + "\n")
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

# I'm really not proud of what follows. Sorry
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

################################################################################################################################
# Checking the restrictions

pos_tol = 1e-8
norm_tol = 1e-14
ghz_tol = 1e-8

ProbOut  = np.zeros(out_list+in_list+(q_pts,))
for e,a,b,c,x,y,z,j in genSeq(out_list+in_list+(q_pts,)):
    ProbOut[e,a,b,c,x,y,z,j] = ProbFull[e,a,b,c,x,x,x+q_in_Alice*y,x+q_in_Alice*z,j]

with open(os.path.join(out_folder,'Checklist.txt'),'w') as o_fl:
    o_fl.write('Nonnegativity tolerance: ' + str(pos_tol) +'\n')
    o_fl.write('Normalization tolerance: ' + str(norm_tol)+'\n')
    o_fl.write('Comparison to reduced distribution tolerance: ' + str(ghz_tol)+'\n\n')
    
    for j in range(q_pts):

        o_fl.write('Visibility {}'.format(vis_list[j])+'\n')

        o_fl.write('Nonnegativity -- general estimation: '+str(not np.any(ProbFull<-pos_tol))+'\n')
        o_fl.write('Normalization: ' + str(not np.any(np.abs(np.sum(ProbFull,(0,1,2,3))-1)>norm_tol)) + '\n')

        ProbGHZ = ProbRed(vis_list[j])

        comp = True
        for a,b,c,x,y,z in genSeq(out_list[1:]+in_list):
            comp &= not (np.abs(np.sum(ProbOut[:,a,b,c,x,y,z,j],0)-ProbGHZ[a,b,c,x,y,z])>ghz_tol)
            if not comp:
                break

        o_fl.write('Comparison with the quantum probability: '+str(comp)+'\n')
        o_fl.write("Violations of Svetlichny's inequality: " + str(np.sum(np.reshape(np.sum(ProbOut[:,:,:,:,:,:,:,j],0)*Bell,[-1,1])))+'\n')

        o_fl.write("Guessing probability: " + str(np.sum(ProbOut[0,:,:,0,0,0,0,j]+ProbOut[1,:,:,1,0,0,0,j],(0,1)))+'\n\n')




