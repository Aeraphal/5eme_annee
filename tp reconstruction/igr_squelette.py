
import numpy as np
import torch
import mcubes
from torch import nn
from scipy import spatial
from matplotlib import pyplot as plt
import torch.nn.functional as F

class GeomNet(nn.Module):
    def __init__(self, nlayers, nneurons):
        super(GeomNet, self).__init__()
        ##TODO
        self.layers = nn.ModuleList([])
        for i in range(nlayers-1):
            self.layers.append(nn.Linear(self.layers[-1],nneurons))

    def forward(self,x):
        ##TODO
        x = F.relu(self.layers(x))
        return x

#Cette fonction calcule le gradient de la sortie par rapport à l 'entrée (utile pour la contrainte eikonale)    
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


#fonction qui calcule la loss d'alignement des normales
def sdf_loss_align(grad, normals):
    return (1-nn.functional.cosine_similarity(grad, normals, dim = 1)).mean()


def evaluate_loss(geomnet, p, device, lpc, leik, lambda_pc, lambda_eik, batch_size=2000, pc_batch_size=2000):
    pts_random = torch.rand((batch_size, 3), device = device)*2-1
    pts_random.requires_grad = True
  
    sample = torch.randint(p.shape[0], (pc_batch_size,))

    sample_pc = p[sample,0:3]
    sample_nc = p[sample,3:]

    #TODO: get the sdf
#    sdf_pc = 
#    sdf_random = 

    #TODO: get the gradient
    grad_pc = gradient(sdf_pc, sample_pc)
    grad_random = gradient(sdf_random, pts_random)
  
    # TODO compute and store the losses
    loss_pc = sdf_loss_align(grad_pc,sample_nc)
    loss_eik = sdf_loss_align(grad_random,sample_nc)
    
    # append all the losses
    lpc.append(float(loss_pc))
    leik.append(float(loss_eik))
  
    # sum the losses
    loss = lambda_pc*loss_pc + lambda_eik*loss_eik

    return loss



def main() :
    p = np.loadtxt('armadillo_sub.xyz')
    
    #TODO calcul de la boite englobante
    kdtree = spatial.KDTree(p)

    #TODO normaliser la forme
    p[:,0:3] = p[:,0:3]-p[:,0:3]

    nlayers = 10
    nneurons = 10
    geomnet = GeomNet(nlayers, nneurons) #TODO: créer le réseau
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geomnet.to(device)
    points = torch.from_numpy(p).float().to(device)
    points.requires_grad = True

    lpc, leik = [], []
    lambda_pc = 1
    lambda_eik = 1

    optim = torch.optim.Adam(params = geomnet.parameters(), lr=1e-3)

    nepochs=5000
    
    for epoch in range(nepochs):
        #calculer la loss
        loss = evaluate_loss(geomnet,p,device,lpc,leik,lambda_pc,lambda_eik)

        #calculer les gradients
        gradients = gradient()

        #faire un pas d'optimisation


    #TODO creer une grille, et s'en servir pour le marching cubes (similaire à la méthode "classique")
    X, Y, Z = np.mgrid[np.min(p[:,0]):np.max(p[:,0]),np.min(p[:,1]):np.max(p[:,1]),np.min(p[:,2]):np.max(p[:,2])]
    u = np.zeros_like(X)

    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Z)):
                query = [X[i,j,k],Y[i,j,k],Z[i,j,k]]
                dist, idx = kdtree.query(query)
                
                sign = np.sign(np.dot(query - p[:,0:3][idx], p[:,3:6][idx]))
                u[i,j,k] = sign * dist

    vertices, triangles = mcubes.marching_cubes(u,0)

    mcubes.export_obj(vertices, triangles, 'result.obj')

    # display the loss
    plt.figure(figsize=(6,4))
    plt.yscale('log')
    plt.plot(lpc, label = 'Point cloud loss ({:.2f})'.format(lpc[-1]))
    plt.plot(leik, label = 'Eikonal loss ({:.2f})'.format(leik[-1]))
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("loss.pdf")
    plt.close()

main()
        
