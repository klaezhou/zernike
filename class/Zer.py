import numpy as np
from astropy import units as u
from pyoof  import zernike # calling the sub-package
import math
class Zer:
    def __init__(self,order,num_points,rho,theta):
        """
        Parameters
        ----------
        order: Zernike polynomials up to order^th order  total :(order+1)*(order+2)/2 ( positive int)
        num_points: number of points (int)
        rho: radius (`~numpy.ndarray)
        theta: theta angle (`~numpy.ndarray)  given by \\theta = \\mathrm{arctan}(y / x)
        """
        self.num_poly=int((order+1)*(order+2)/2)
        self.Zernike =np.zeros((num_points,self.num_poly))
        self.num_points=num_points
        assert len(rho) == num_points, "len(rho) must equals to num_points"
        assert len(theta) == num_points, "len(theta) must equals to num_points"
        self.rho = rho
        self.theta =theta
        self.order=order
        self.beta= np.zeros((self.num_poly,1))
        self.alpha=None
        self.M=None
        self.Ox,self.Oy,self.Oz=6,6,8
        self.dZ_dx,self.dZ_dy=None,None
        self.C=None
        self.Z_star=None
        
    
    def Construct_Zernike(self)->np.ndarray:
        loc=0
        order=self.order
        for n in range(order + 1):
            m_values = []
            for m in range(-n, n + 1):
                if (n - m) % 2 == 0:
                    m_values.append(m)
            for i in range(len(m_values)):
                self.Zernike[:,loc]=zernike.U(n=n, l=m_values[i], rho=self.rho / self.rho.max(),theta=self.theta)
                loc+=1
        return self.Zernike
    
    def Construct_M(self,times,Z_star,dZ_dx=None,dZ_dy=None,Ox=None,Oy=None,Oz=None)->np.ndarray:
        """
        Notes : get dZ_dx and dZ_dy from dZ_dr and dZ_dtheta
        Formula: 
        \frac{dZ}{dx} = \frac{\partial Z}{\partial \rho} \cos \theta - \frac{\partial Z}{\partial \theta} \frac{\sin \theta}{\rho}
        \frac{dZ}{dy} = \frac{\partial Z}{\partial \rho} \sin \theta + \frac{\partial Z}{\partial \theta} \frac{\cos \theta}{\rho}

        Parameters
        ----------
        dZ_dx : np.ndarray length ==N same rules to dZ_dy  
    
        times : list{np.ndarray} with length ==Z_star   

        """
        def diagonal_stack(*matrices):
            
            total_rows = sum(mat.shape[0] for mat in matrices)
            total_cols = sum(mat.shape[1] for mat in matrices)
            result = np.zeros((total_rows, total_cols))
            current_row = 0
            current_col = 0
            for mat in matrices:
                result[current_row:current_row + mat.shape[0], current_col:current_col + mat.shape[1]] = mat
                current_row += mat.shape[0]
                current_col += mat.shape[1]
            
            return result
        if dZ_dx is None: dZ_dx=self.dZ_dx
        else: self.dZ_dx=dZ_dx
        if dZ_dy is None: dZ_dy=self.dZ_dy
        else: self.dZ_dy=dZ_dy
        if Ox is None: Ox=self.Ox 
        else: self.Ox=Ox
        if Oy is None: Oy=self.Oy 
        else: self.Oy=Oy
        if Oz is None: Oz=self.Oz 
        else: self.Oz=Oz
        assert len(times)==Z_star, "times must have the same length as Z_star"
        # caculate the mean time 
        self.Z_star=Z_star
        mean_times_array = np.array([np.mean(t) for t in times])
        M_i=[]
        loc=0
        for i in range(Z_star):
            Ni=len(times[i])
            mx=np.zeros((Ni,Ox+1))
            my=np.zeros((Ni,Oy+1))
            mz=np.zeros((Ni,Oz+1))
            for m in range(Ox+1):
               mx[:,m]=dZ_dx[loc:loc+Ni]*(times[i]-mean_times_array[i])**m 
            for m in range(Oy+1):   
               my[:,m]=dZ_dy[loc:loc+Ni]*(times[i]-mean_times_array[i])**m 
            for m in range(Oz+1):     
               mz[:,m]=(times[i]-mean_times_array[i])**m 
            Mi=np.hstack((mx,my,mz))
            M_i.append(Mi)
            loc+=Ni
                
        
        M=diagonal_stack(*M_i)
        self.M=M
        return self.M
    
    def Construct_C(self,times,Z_star,sreg=10000,dZ_dx=None,dZ_dy=None,Ox=None,Oy=None,Oz=None):
        if dZ_dx is None: dZ_dx=self.dZ_dx
        else: self.dZ_dx=dZ_dx
        if dZ_dy is None: dZ_dy=self.dZ_dy
        else: self.dZ_dy=dZ_dy
        if Ox is None: Ox=self.Ox 
        else: self.Ox=Ox
        if Oy is None: Oy=self.Oy 
        else: self.Oy=Oy
        if Oz is None: Oz=self.Oz
        else : self.Oz=Oz
        self.C=np.zeros((6*(Z_star-1)+3,(Ox+Oy+Oz+3)*Z_star))
        mean_times_array = np.array([np.mean(t) for t in times])
        def Rt(t):
            def Rn(t, order):
                Rn = np.zeros((2, order + 1))
                Rn[0, :] = [t**i for i in range(order + 1)]
                Rn[1, 1:] = [i * t**(i-1) for i in range(1, order + 1)]
                return Rn
            Rx = Rn(t, Ox)
            Ry = Rn(t, Oy)
            Rz = Rn(t, Oz)

            Rt = np.zeros((6, Ox + Oy + Oz + 3))

            Rt[0:2, 0:Ox + 1] = Rx
            Rt[2:4, Ox + 1:Ox + Oy + 2] = Ry
            Rt[4:6, Ox + Oy + 2:Ox + Oy + Oz + 3] = Rz

            return Rt
        def S(Ox, Oy):
            columns = Ox + Oy + 3
            S = np.zeros((3, columns))
            S[0,0]=1
            S[1,Ox+1]=1
            S[2,Ox+Oy+2]=1
            return S
        locx=0
        locy=0
        lenR=(Ox+Oy+Oz+3)
        for i in range(Z_star-1):
            self.C[locx:locx+6,locy:locy+lenR]=Rt(times[i+1][0]-mean_times_array[i])
            locy+=lenR
            self.C[locx:locx+6,locy:locy+lenR]=-Rt(times[i+1][0]-mean_times_array[i+1])
            locx+=6
        self.C[-3:,0:lenR-Oz]=S(Ox,Oy)
        self.C=sreg*self.C
        return self.C
    
    def diff_trans(self,dZ_dr,dZ_dtheta)->tuple:
        """
        Returns
        -------
        tuple
            dZ_dx,dZ_dy
        """
        dZ_dx = dZ_dr * np.cos(self.theta) - dZ_dtheta * np.sin(self.theta) / self.rho
        dZ_dy = dZ_dr * np.sin(self.theta) + dZ_dtheta * np.cos(self.theta) / self.rho
        self.dZ_dx,self.dZ_dy=dZ_dx,dZ_dy
        return dZ_dx,dZ_dy
    
    def initial_beta(self,beta=None):
        if beta is not None:
            self.beta=beta
        return self.beta
    
    def initial_alpha(self,alpha=None):
        if alpha is not None:
            self.alpha=alpha
        else:
            self.alpha=np.zeros(((self.Ox+self.Oy+self.Oz+3)*self.Z_star,1))
        return self.alpha
    

    def diff_zer(x,y,n,m):
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        def diffr_R(n, m, rho):
            sum = np.zeros_like(rho)
            for k in range((n - abs(m)) // 2 + 1):
                # print("m=", m, "n=", n,"k=",k)
                sum += ((-1) ** k * math.factorial(n - k) / (
                            math.factorial(k) * math.factorial((n + abs(m)) // 2 - k) * math.factorial(
                        (n - abs(m)) // 2 - k))) * rho ** (n - 2 * k-1)*( n - 2*k )
            return sum

        def diffx_r(x,y):
            diff=2*x/math.sqrt(x**2+y**2)
            return diff

        def difftheta_R(n,m,rho,theta):
            diff=m*zernike_radial(n, m, rho)*np.sin(theta)
            return diff

        def diffx_theta(x,y):
            diff=-1*y/x**2+y**2
            return diff

        def diffy_r(x,y):
            diff=2*y/math.sqrt(x**2+y**2)
            return diff

        def diffy_theta(x,y):
            diff=-x/x**2+y**2
            return diff

        diffx_zer=diffr_R(n,m,rho)*diffx_r(x,y)+difftheta_R(n,m,rho,theta)*diffx_theta(x,y)
        diffy_zer=diffr_R(n,m,rho)*diffy_r(x,y)+difftheta_R(n,m,rho,theta)*diffy_theta(x,y)

        return diffx_zer,diffy_zer
        
            
        


