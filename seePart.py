# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:43:17 2018

@author: Qing Yan
"""
def compare(i):
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(p[idx,:,:].detach().numpy())
    #for i in range(16):
    aff_matrix = theta.view(-1,16,2,3)[idx,i,:,:]
    angle = torch.atan(aff_matrix[1,0]/aff_matrix[0,0])
    angle = angle.detach().numpy()
    posi = np.unravel_index(torch.argmax(KM[idx,i,:,:]),(28,28))
    rect = patches.Rectangle((posi[1]-5*(np.cos(angle)-np.sin(angle)),posi[0]+5*(-np.cos(angle)-np.sin(angle))),
                             10,10,angle/3.14*180,linewidth=1,edgecolor='r',facecolor='none')
    ax[0].add_patch(rect)
    ax[1].imshow((x*KM)[idx,i,:,:].detach().numpy())
    plt.show()