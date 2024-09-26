import numpy as np
from astropy import units as u
from pyoof  import zernike # calling the sub-package

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
        self.M=None
        self.Ox,self.Oy,self.Oz=6,6,8
        self.dZ_dx,self.dZ_dy=None,None
        
        
    
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
        dZ_dx : np.ndarray length ==N similar to dZ_dy  
    
        times : list{np.ndarray} with length ==Z_star   

        """
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
                
        
        M=self.diagonal_stack(M_i)
        self.M=M
        return M
    

    
    def diff_trans(self,dZ_dr,dZ_dtheta):
        dZ_dx = dZ_dr * np.cos(self.theta) - dZ_dtheta * np.sin(self.theta) / self.rho
        dZ_dy = dZ_dr * np.sin(self.theta) + dZ_dtheta * np.cos(self.theta) / self.rho
        self.dZ_dx,self.dZ_dy=dZ_dx,dZ_dy
        return dZ_dx,dZ_dy
    

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
    
        
    

