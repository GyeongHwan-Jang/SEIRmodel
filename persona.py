import math
import numpy as np

class Persona:
    def __init__(self,i, posx, posy, objx, objy, v, t_contagiado, fijo):
        # movement speed
        self.v=v
        # target position
        self.objx=objx
        self.objy=objy
        #ID and name
        self.indice=i
        self.nombre="Persona "+str(i)
        #State: Susceptible, Infected or Retired
        self.infectado = False
        self.suceptible = True
        self.retirado = False
        #Current position
        self.posx = posx
        self.posy=posy
        #is it fixed (in quarantine)?
        self.fijo = fijo

        # displacement per iteration
        if self.fijo:

            self.deltax = 0
            self.deltay = 0
        else:
            self.deltax = (self.objx - self.posx) / self.v
            self.deltay = (self.objy - self.posy) / self.v
        #time in which the person was infected
        self.i_contagio=-1
        #time that the infection lasts, recover time
        self.t_contagiado = t_contagiado


    def __str__(self):
        return self.nombre+" en posicin "+str(self.posx)+", "+str(self.posy)

    def infectar(self,i):
        #infect
        self.infectado=True
        self.suceptible=False
        self.retirado = False
        self.i_contagio=i

    def retirar(self):
        #heal
        self.retirado=True
        self.suceptible=False
        self.infectado=False

    def set_objetivo(self,objx,objy):
        #this function is used to create a new target position
        self.objx=objx
        self.objy=objy
        if self.fijo:
            self.deltax = 0
            self.deltay=0
        else:
            self.deltax = (self.objx - self.posx) / self.v
            self.deltay = (self.objy - self.posy) / self.v
        print("Nuevo OBJ   ", self.objx,self.objy,"  ",self.indice)

    def check_contagio(self,i):
        #this function is used to heal the person if the established infection time has passed
        if self.i_contagio>-1:
            if i-self.i_contagio>self.t_contagiado:
                self.retirar()


    def update_pos(self, n_posx, n_posy):
        #this funcion animates the movement
        if(n_posx==0 and n_posy==0):
            self.posx=self.posx+self.deltax
            self.posy=self.posy+self.deltay
        else:
            self.posx=n_posx
            self.posy=n_posy

        if abs(self.posx-self.objx)<3 and abs(self.posy-self.objy)<3:
            self.set_objetivo(np.random.random()*100, np.random.random()*100)
        if self.posx>100:
            self.posx=100
        if self.posy>100:
            self.posy=100
        if self.posx<0:
            self.posx=0
        if self.posy<0:
            self.posy=0

    def get_color(self):
        if self.infectado:
            return 'red'
        if self.suceptible:
            return 'blue'
        if self.retirado:
            return 'gray'

    def get_pos(self):
        return (self.posx,self.posy)

    def get_dist(self,x,y):
        #this funcion calculates the distance between this person an another.
        return math.sqrt((self.posx-x)**2+(self.posy-y)**2)