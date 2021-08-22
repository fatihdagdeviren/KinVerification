#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
# http://deap.readthedocs.io/en/master/examples/ga_knapsack.html
# http://deap.readthedocs.io/en/master/api/algo.html
# http://deap.readthedocs.io/en/master/api/tools.html#deap.gp.cxOnePoint

import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
import KinVerification as kv
# Basarisli 166,175,122,125,126,127,128,129,130,133,134,136,142
# basarisiz 121,123,124,131,132,135,137,138,139,140,141
pathForChild = "../KinFace_V2/01/142.jpg"
kv.main(pathForChild)
child = kv.landmarks_children[0]
i=0
yakinParentSayisi = 15
sonucList=[]

def parentleriGetir(deger):
    global i
    i=deger

max_benzerlik = 0.7
IND_INIT_SIZE = len(child) # child vektor uzunlugunda random bir vektor olusturucagim.
NBR_ITEMS = 2 # 0,1 iceren 0 lar icin youngtan almayacagim, 1 ler icin alicam.

# To assure reproductibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
random.seed(64)
creator.create("Fitness", base.Fitness, weights=(-1.0,1.0))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_item", random.randrange,NBR_ITEMS)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalKnapsack(individual):   
    valueList = []        
    for item in range(0,len(individual)):
        deger = individual[item]        
        if deger == 0: # young parenttan almayacagim
            karsilastirilacakDeger = kv.landmarks_parents[i][1][item][2] # oldparent
        else:
            karsilastirilacakDeger = kv.landmarks_parents[i][0][item][2]  #youngParent
        
        lbpChild = child[item][2]
        #dist = numpy.linalg.norm(lbpChild-karsilastirilacakDeger)
        dist = numpy.corrcoef(lbpChild,karsilastirilacakDeger)
        valueList.append(dist[0][1])
    value = numpy.average(valueList)
    if value<max_benzerlik:
        return -1,-1             # Ensure overweighted bags are dominated
    return value,value


def onePointCrossOver(ind1,ind2):
    # Tek nokta caprazlama y�ntemini gerceklestiren y�ntemdir.
    size = min(len(ind1), len(ind2))
    if size>1:
        cxpoint = random.randint(1, size - 1)
    else:
        cxpoint = 1
    temp = toolbox.clone(ind1)
    ind1[cxpoint:] = ind2[cxpoint:]
    ind2[cxpoint:] = temp[cxpoint:]       
    del ind1.fitness.values  
    del ind2.fitness.values    
    return ind1,ind2

def cutAndAddCrossOver(ind1,ind2):
    # kes ve ekle �aprazlama y�ntemini ger�ekle�tiren metot.
    if len(ind1)>0 and len(ind2)>0:
        cxPoint1 = random.randint(1,len(ind1))
        cxPoint2 = random.randint(1,len(ind2))
        temp = toolbox.clone(ind1)
        ind1[cxPoint1:]  = ind2[cxPoint2:]
        ind2[cxPoint2:]  = temp[cxPoint1:]
    return ind1,ind2                            
    
    
   
def mutationComp(individual):
    " Herhangi bir elementin complementini alicam"
    x = random.randrange(len(individual))
    individual[x] = (individual[x]+1) % NBR_ITEMS
    return individual,


def main():       
    
    NGEN = 20 #The number of generation.
    MU = 12 #The number of individuals to select for the next generation.
    LAMBDA = 100 #The number of children to produce at each generation.
    CXPB = 0.7
    MUTPB = 0.2 # bireylerdeki genler mutasyon olasiligina bagli olarak degitiriolior.      
  
    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", onePointCrossOver)
    toolbox.register("mutate", mutationComp)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    # hoftaki en sonuncu bana en yakin sonucu vericek onu gostermem lazım ekranda.
    enYakinDeger = hof[-1]
    sonucList.append([i,enYakinDeger])
    return pop, stats, hof
                 
if __name__ == "__main__":
    for x in range(0,len(kv.landmarks_parents)):
        print("\n islenen parent ({})".format(kv.landmarks_parents[x][2]))
        if len(kv.landmarks_parents[x][0])>0 and len(kv.landmarks_parents[x][1])>0:
            parentleriGetir(x)
            main()
    sonucList=sorted(sonucList,key=lambda x:x[1].fitness.values[1],reverse=True)
    enYakinParent = sonucList[0:yakinParentSayisi] # en yakin p parent
    kv.ekrandaGoster(pathForChild,enYakinParent)
    deneme= 2
