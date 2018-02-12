from pylab import *

class GA:
    def __init__(self,
                 pop_size,DNA_size, DNA_fitness,
                 cross_rate,mutate_rate,noise_rate,noise_amp
                ):
        self.pop_size    = pop_size
        
        self.DNA_size    = DNA_size
        self.DNA_fitness = DNA_fitness
        
        self.cross_rate  = cross_rate
        self.mutate_rate = mutate_rate
        self.noise_rate  = noise_rate
        self.noise_amp   = noise_amp
        
        
        #self.pop = np.random.uniform(0, 1, size = (self.pop_size, self.DNA_size))
        self.pop = 0.05*np.ones( [self.pop_size, self.DNA_size])

    def get_pop_fitness(self):    
        pop_fitness = []
        # calculate fitness for each DNA in population
        for i in range(self.pop_size): 
            DNA = self.pop[i,:] 
            fit = self.DNA_fitness(DNA)
            pop_fitness.append(fit)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness

    def select(self):
        fitness = self.get_pop_fitness()
        idx = np.random.choice(np.arange(self.pop_size), 
                               size = self.pop_size,
                               replace = True, 
                               p = fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, partner):
        if np.random.rand() < self.cross_rate:
            # choose crossover points
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   
            # mating and produce one child
            parent[cross_points] = partner[cross_points]                            
        return parent

    def mutate(self, child):
        addnoise  = np.random.uniform(0,1,size=self.DNA_size) < self.noise_rate
        addmutate = np.random.uniform(0,1,size=self.DNA_size) < self.mutate_rate
        child[addnoise] += np.random.normal(0,self.noise_amp,size=self.DNA_size)[addnoise]
        child[addmutate] = np.random.uniform(0,1,size=self.DNA_size)[addmutate]
        
        for point in range(self.DNA_size):
            if child[point]<0:
                child[point]=0
            if child[point]>1:
                child[point]=1  
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            # select another individual from pop_copy
            ipartner  = np.random.randint(0, self.pop_size, size=1)
            partner   = pop_copy[ipartner][0]
            child     = self.crossover(parent, partner)         
            child     = self.mutate(child)
            parent[:] = child
        self.pop = pop



