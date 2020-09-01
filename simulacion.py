from persona import Persona
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#SIMULATION PARAMETERS
n=350  #number of individuals
p_infectadas = 1  #percentage of infected people at the beginning of the simulation (0-100%)
r_contagio=2  #radius of transmission in pixels (0-100)
p_contagio=4  #probability of transmission in percentage (0-100%)
p_aislamiento = 70  #percentage of the people in quarantine (0-100%)
t_contagiado=100   #time taken to recover in number of frames (0-infinity)


contagiados=0
personas=[]

#creating all the individuals in random positions. Infecting some of them
for i in range(n):
    p = Persona(i,np.random.random()*100, np.random.random()*100,
                np.random.random() * 100, np.random.random() * 100,
                (np.random.random()+0.5)*100,t_contagiado, False)

    if np.random.random()<p_infectadas/100:
        p.infectar(0)
        contagiados=contagiados+1
    if np.random.random()<p_aislamiento/100:
        p.fijo=True

    personas.append(p)


#this creates all the graphics
fig = plt.figure(figsize=(18,9))
ax = fig.add_subplot(1,2,1)
cx = fig.add_subplot(1,2,2)
ax.axis('off')
cx.axis([0,1000,0,n])
scatt=ax.scatter([p.posx for p in personas],
           [p.posy for p in personas],c='blue',s=8)
caja = plt.Rectangle((0,0),100,100,fill=False)
ax.add_patch(caja)
cvst,=cx.plot(contagiados,color="red",label="Currently infected")
rvst,=cx.plot(contagiados,color="gray",label="Recovered")
cx.legend(handles=[rvst,cvst])
cx.set_xlabel("Time")
cx.set_ylabel("People")


ct=[contagiados]
rt=[0]
t=[0]



#function excecuted frame by frame
def update(frame,rt,ct,t):
    contciclo = 0
    recuciclo = 0
    colores = []
    sizes = [8 for p in personas]
    for p in personas:
        #check how much time the person has been sick
        p.check_contagio(frame)
        #animate the movement of each person
        p.update_pos(0,0)
        if p.retirado:
            recuciclo+=1 #count the amount of recovered
        if p.infectado:
            contciclo=contciclo+1 #count the amount of infected
            #check for people around the sick individual and infect the ones within the
            # transmission radius given the probability
            for per in personas:
                if per.indice==p.indice or per.infectado or per.retirado:
                    pass
                else:
                    d=p.get_dist(per.posx,per.posy)
                    if d<r_contagio:
                        if np.random.random() < p_contagio / 100:
                            per.infectar(frame)
                            sizes[per.indice]=80


        colores.append(p.get_color()) #change dot color according to the person's status

    #update the plotting data
    ct.append(contciclo)
    rt.append(recuciclo)
    t.append(frame)


    #tramsfer de data to the matplotlib graphics
    offsets=np.array([[p.posx for p in personas],
                     [p.posy for p in personas]])
    scatt.set_offsets(np.ndarray.transpose(offsets))
    scatt.set_color(colores)
    scatt.set_sizes(sizes)
    cvst.set_data(t,ct)
    rvst.set_data(t,rt)
    return scatt,cvst,rvst

#run the animation indefinitely
animation = FuncAnimation(fig, update, interval=25,fargs=(rt,ct,t),blit=True)
plt.show()