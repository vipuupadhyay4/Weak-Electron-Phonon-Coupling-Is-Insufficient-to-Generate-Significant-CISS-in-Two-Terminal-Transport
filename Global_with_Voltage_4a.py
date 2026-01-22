import numpy as np
import cmath
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.integrate import quad
import random
from scipy.signal import hilbert

# Pauli Matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
W = np.array([[1, 0], [0, 1]])

# System Parameters
t0 = 0.4
epsilon0 = -6*t0
t1 =0.1*t0
lambda0 = t0/40
lambda1 =0.1*lambda0
nl = 4
Mt =2
Nf = Mt * nl
eta1=0 
a = 1
c = a
k1 = t0 / 4
omega0 = 0.1 * t0
T = 0.25
tole=0.00005 
# Vectors defining helicity
def phi(m):
    return 2 * np.pi * (m - 1) * Mt / (Nf - 1)

def rvec(m, a, c):
    return np.array([a * np.cos(phi(m)), 
                    1* a * np.sin(phi(m)), 
                     (m - 1) * c / (Nf - 1)])
    
def d(m, s):
        vec_diff = rvec(m, a, c) - rvec(m + s, a, c)
        return vec_diff / np.linalg.norm(vec_diff)

def v(m, s):
    return np.cross(d(m, s), d(m+s,s))



# Pauli Matrices List
Vpau = [X, Y, Z]

def dot(A, V):
    return A[0] * V[0] + A[1] * V[1] + A[2] * V[2]

def vpm(m):
    return dot(v(m, 1), Vpau)




#################################################################################################
# Hamiltonian Matrix
H = np.zeros((2 * Nf, 2 * Nf), dtype=complex)
for i in range(0, Nf - 2):
 
     #On Site energies
     H[2*i,2*i]= epsilon0;
     H[2*i+1,2*i+1]= epsilon0;
     
     #Nearest Neighbor hopping between same spins
     H[2*i,2*(i+1)]= -t0;
     H[2*(i+1),2*i]= -t0;
     H[2*i+1,2*(i+1)+1]= -t0;
     H[2*(i+1)+1,2*i+1]= -t0;
    
     
     #Next Nearest Hopping between same spins
     H[2*i,2*(i+2)]= lambda0 *1j *vpm(i+1)[0,0];
     H[2*(i+2),2*i]= np.conj(lambda0 *1j *vpm(i+1)[0,0]);
     H[2*i+1,2*(i+2)+1]= lambda0 *1j *vpm(i+1)[1,1];
     H[2*(i+2)+1,2*i+1]= np.conj(lambda0 *1j *vpm(i+1)[1,1]);
     
     
     #Next Nearest Hopping between different spins
     H[2*i,2*(i+2)+1]= lambda0* 1j *vpm(i+1)[0,1];
     H[2*(i+2)+1,2*i]=  np.conj(lambda0*1j *vpm(i+1)[0,1]);
     H[2*i+1,2*(i+2)]= lambda0 *1j *vpm(i+1)[1,0];
     H[2*(i+2),2*i+1]= np.conj(lambda0 *1j *vpm(i+1)[1,0]);
    
#Second last 
#On Site energies
H[2*(Nf-2),2*(Nf-2)]= epsilon0;
H[2*(Nf-2)+1,2*(Nf-2)+1]= epsilon0;

#Nearest Neighbor hopping between same spins
H[2*(Nf-2),2*((Nf-2)+1)]= -t0;
H[2*((Nf-2)+1),2*(Nf-2)]= -t0;
H[2*(Nf-2)+1,2*((Nf-2)+1)+1]= -t0;
H[2*((Nf-2)+1)+1,2*(Nf-2)+1]= -t0;

#Final on site energies
H[2*(Nf-1),2*(Nf-1)]= epsilon0;
H[2*(Nf-1)+1,2*(Nf-1)+1]= epsilon0;
#################################################################################################

#Matrix formation for electron phonon self energy
# Hamiltonian Matrix
EPH = np.zeros((2 * Nf, 2 * Nf), dtype=complex)
for i in range(0, Nf - 2):
  
     #Nearest Neighbor hopping between same spins
     EPH[2*i,2*(i+1)]= -t1;
     EPH[2*(i+1),2*i]= -t1;
     EPH[2*i+1,2*(i+1)+1]= -t1;
     EPH[2*(i+1)+1,2*i+1]= -t1;
    
     
     #Next Nearest Hopping between same spins
     EPH[2*i,2*(i+2)]= lambda1 *1j *vpm(i+1)[0,0];
     EPH[2*(i+2),2*i]= lambda1 *np.conj(1j *vpm(i+1)[0,0]);
     EPH[2*i+1,2*(i+2)+1]= lambda1 *1j *vpm(i+1)[1,1];
     EPH[2*(i+2)+1,2*i+1]= lambda1 *np.conj(1j *vpm(i+1)[1,1]);
     
     
     #Next Nearest Hopping between different spins
     EPH[2*i,2*(i+2)+1]= lambda1*1j*vpm(i+1)[0,1];
     EPH[2*(i+2)+1,2*i]= lambda1*np.conj(1j *vpm(i+1)[0,1]);
     EPH[2*i+1,2*(i+2)]= lambda1*1j*vpm(i+1)[1,0];
     EPH[2*(i+2),2*i+1]= lambda1*np.conj(1j *vpm(i+1)[1,0]);
    
#Nearest Neighbor hopping between same spins
EPH[2*(Nf-2),2*((Nf-2)+1)]= -t1;
EPH[2*((Nf-2)+1),2*(Nf-2)]= -t1;
EPH[2*(Nf-2)+1,2*((Nf-2)+1)+1]= -t1;
EPH[2*((Nf-2)+1)+1,2*(Nf-2)+1]= -t1;

print(np.allclose(EPH, EPH.conj().T, atol=1e-10))




########################## Up Lead Magnetisation ##############################################################################################
Jnet1=[] 
Jnet2=[]
for V in np.arange (0*t0,25*t0,1*t0):
     
    def NwB(omega0, T):
        # Example function for NwB (replace with your actual formula)
        return 1 / (np.exp(omega0 / T) -1)
    
    dab=NwB(omega0, T)
    dem=1+dab    
    p=0.5
    EP = np.zeros((2 * Nf, 2 * Nf), dtype=complex)
    Gin=np.zeros((2 * Nf, 2 * Nf), dtype=complex)
    Gout=np.zeros((2 * Nf, 2 * Nf), dtype=complex)
    EP[0,0]=-k1*1j/2*(1+p);
    EP[1,1]=-k1*1j/2*(1-p);
    EP[2*(Nf-1),2*(Nf-1)]=-k1*1j/2;
    EP[2*(Nf-1)+1,2*(Nf-1)+1]=-k1*1j/2;
    Gin[0,0]=k1*(1+p);
    Gin[1,1]=k1*(1-p);
    Gout[2*(Nf-1),2*(Nf-1)]=k1;
    Gout[2*(Nf-1)+1,2*(Nf-1)+1]=k1;
    
        
    
      # Define the Fermi-Dirac distribution function Nw
    def Nw(w, mu, T):
        return 1 / (np.exp((w - mu) / T) + 1)
    
    mu1=V/2
    mu2=-V/2
    
    def f1(En):
            return Nw(En, mu1, T)
    def f2(En):
            return Nw(En, mu2, T)
   
       
    Jl=[]
    Dl=[]
    for x in range(0,10):
        # We will define the energy mesh 
        Nhis=7 #Order of Pulay Mixings
        NE = 400;
        delE = omega0;
        E0 =-4+x*omega0/(10);
        EnG = np.array([E0 + n * delE for n in range(NE)])
        
        
            #Initial Guesses based on previous iterates       
        GR = np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        GA = np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        GP= np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        GN= np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        A= np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        for n in range(1, NE-1):    
            z = EnG[n]+1j*eta1
            GR[n] = np.linalg.inv(z*np.eye(2*Nf) - H - EP)
            GA[n] = np.conj(GR[n].T)
            GN[n] = f1(z)*GR[n] @ Gin @ GA[n] + f2(z)*GR[n] @ Gout @ GA[n]
            A[n]   = 1j*(GR[n] - GA[n])
            GP[n] = A[n] - GN[n]

        G0 = np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        EIN0 = np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        A = np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        
            
                
            
        
        
        #Defining the matrices we will be needing in the analysis
        GRN = np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        GAN = np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        GPN= np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
        GNN= np.zeros((NE, 2 * Nf, 2 * Nf), dtype=complex)
       
         
         
         #History storing matrices, HGR stores GReen function 
         #difference, EGR stores differences of Residues
        HGR= np.zeros((Nhis,NE,2*Nf,2 * Nf), dtype=complex)
        EGR= np.zeros((Nhis,NE*2 * Nf* 2 * Nf), dtype=complex)
     
         
                    
        #Going towards the Loop now, have to be very careful what is what
        cnt=0
        
        with open("hilbert_errors.txt", "w") as f:           
            #The loop will find x_k+1, given x_k and evaluate in a loop (ok)
            while cnt <= 30000:
            
                #Functions evaluation for initial Guess
                for n in range(1, NE-1):
                    GNprev, GPprev = GN[n-1], GP[n-1]
                    GNnext, GPnext = GN[n+1], GP[n+1]
                    G0[n]   = EPH @ (dab*GNprev + dem*GPprev + dem*GNnext + dab*GPnext) @ EPH
                    EIN0[n] = EPH @ (dab*GNprev + dem*GNnext) @ EPH
            
            # ===== Exact principal-value Hilbert =====
                def hilbert_pv_trapz(f, E):
                    N = E.size
                    dE = E[1] - E[0]
                    diff = E[:, None] - E[None, :]
                    np.fill_diagonal(diff, np.inf)
                    kernel = 1.0 / diff
                    integral = kernel @ (f * dE)
                    return integral / np.pi
                
                gamma_exact = np.zeros_like(G0, dtype=complex)
                for i in range(2*Nf):
                    for j in range(2*Nf):
                        gamma_exact[:, i, j] = hilbert_pv_trapz(G0[:, i, j], EnG)
                ERR = 0.5 * (gamma_exact - 1j * G0)
                
                #Functions evaluation for initial Guess
                for n in range(1, NE-1):    
                    z = EnG[n]+1j*eta1
                    GRN[n] = np.linalg.inv(z*np.eye(2*Nf) - H - EP -ERR[n])
                    GAN[n] = np.conj(GRN[n].T)
                    GNN[n] = f1(z)*GRN[n] @ Gin @ GAN[n] + f2(z)*GRN[n] @ Gout @ GAN[n] + GRN[n] @ EIN0[n] @ GAN[n]
                    A[n]   = 1j*(GRN[n] - GAN[n])
                    GPN[n] = A[n] - GNN[n]
                            
                GR_old=GR.copy()
                R_new=(GRN-GR).flatten()
                err1=np.linalg.norm(R_new)/np.max(np.abs(GRN))
                errA=np.linalg.norm(R_new)

            
            
            
                print(f"Iteration {cnt}, Residual = {err1},Absolute={errA}")
                f.write(f"{cnt}\t{err1}\t{errA}\n") 
                f.flush()
                
                
                if err1 < tole:
                    AF,GRF, GAF, GNF, GPF = A.copy(),GRN.copy(), GAN.copy(), GNN.copy(), GPN.copy()
                    for n in range(1, NE-1):
                        GNprev, GPprev = GNF[n-1], GPF[n-1]
                        GNnext, GPnext = GNF[n+1], GPF[n+1]
                        G0[n]   = EPH @ (dab*GNprev + dem*GPprev + dem*GNnext + dab*GPnext) @ EPH
                        EIN0[n] = EPH @ (dab*GNprev + dem*GNnext) @ EPH
                    break
                
                
                idx = cnt % Nhis
                HGR[idx] = GR.copy()
                EGR[idx] = R_new.copy()
                    
                if cnt >= 4*(Nhis-1):
                    # Build the residual overlap matrix B
                    B = EGR[:Nhis] @ EGR[:Nhis].conj().T
                    # Build the augmented matrix M for the linear system
                    M = np.zeros((Nhis + 1, Nhis + 1), dtype=complex)
                    M[:Nhis, :Nhis] = B
                    M[:Nhis, Nhis] = 1
                    M[Nhis, :Nhis] = 1
                
                    # Build the right-hand side vector
                    rhs = np.zeros(Nhis + 1)
                    rhs[Nhis] = 1
                
                    
                    # Solve for the coefficients c
                    try:
                        sol = np.linalg.solve(M, rhs)
                        c = sol[:Nhis]
                    except np.linalg.LinAlgError:
                        # If the matrix is singular, fall back to simple mixing
                        c = np.ones(Nhis) / Nhis
                        print("Warning: Singularity detected. Reverting to simple mixing.")
                    beta = 0.3 # default damping               
                    beta1=0
                    if err2<err1:
                        beta=0.1
                    
                    if err1<0.8:
                        beta1=0.6
                    
                    if err1<0.004:
                        beta1=0.8
                                        
                                                                    
                    GR_pulay = sum(c[i] * HGR[i] for i in range(Nhis))
                    GR_next = beta1*GR_old+(1-beta1)*(beta * GR_pulay + (1 - beta) * GRN)

                
                    
                    
                    #Functions evaluation for initial Guess
                    for n in range(1, NE-1):    
                        z = EnG[n]+1j*eta1
                        GA_next[n] = np.conj(GR_next[n].T)
                        GN_next[n] = f1(z)*GR_next[n] @ Gin @ GA_next[n] + f2(z)*GR_next[n] @ Gout @ GA_next[n] + GR_next[n] @ EIN0[n] @ GA_next[n]
                        A[n]   = 1j*(GR_next[n] - GA_next[n])
                        GP_next[n] = A[n] - GN_next[n]
                            

                

                
                                        
                else: 
                    
                    GR_next = GRN.copy()
                    GA_next = GAN.copy()
                    GP_next = GPN.copy()
                    GN_next = GNN.copy()
            
                
                    
                # This should be GRN because, it is now the old Guess, this is x_k
                GR, GA, GN, GP = GR_next.copy(), GA_next.copy(), GN_next.copy(), GP_next.copy()
                err2=err1.copy()
                        
                cnt += 1

        
        for i in range(1, NE-1):
            z=EnG[i]+1j*eta1
            
            
            ILD=np.trace(Gin @ GNF[i])-f1(z)*np.trace(Gin @ AF[i])
          
            IRD=np.trace(Gout @ GNF[i])-f2(z)*np.trace(Gout @ AF[i])
            
            IPD=np.trace(G0[i] @ GNF[i])-np.trace(EIN0[i] @ AF[i])
                
                                       
            Jl.append((z,ILD.real,IRD.real,IPD.real,ILD.real+IRD.real+IPD.real))
            
    
    
    Jls1= sorted(Jl, key=lambda x: x[0])
    
    Jl= np.array(Jls1, dtype=float)  # Convert to NumPy array with float values
    
    # Extract columns properly
    x1 = Jl[:, 0].flatten()  # Ensure 1D array
    IflU=Jl[:, 1].flatten()
    IflD=Jl[:, 2].flatten()
    IfpD=Jl[:,3].flatten()
    IfpT=Jl[:,4].flatten()
        
    
    # Numerical integration
    Ilu = simpson(IflU,x1)
    Ild = simpson(IflD,x1)
    Ilp = simpson(IfpD,x1)
    IlT = simpson(IfpT,x1)
    
    Jl1=Jl.copy()
       
    
    # Append to Jnet
    Jnet1.append((V/t0, Ilu, Ild,Ilp,IlT))
    
 

print(Jnet1)

real_data = np.real(Jnet1)
# Save as a space-separated text file
np.savetxt("Currents_with_voltage_0p1.txt", real_data, fmt="%.9f")  # Adjust precision as needed

np.save("GR_final.npy", GRF)
np.save("GA_final.npy", GAF)
np.save("GN_final.npy", GNF)
np.save("GP_final.npy", GPF)



    
