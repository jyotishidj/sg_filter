def sg_filter(sig, order = 3, Lmax = 65):

    # This is an implementation of the following paper:
    #"Krishnan, Sunder Ram, and Chandra Sekhar Seelamantula. "On the selection of optimum Savitzky-Golay filters."
    #IEEE transactions on signal processing 61.2 (2012): 380-391."


    import numpy as np

    Lmin=order+2

    filtered=np.array([])
    length=np.array([])
    for i in range(10,len(sig)-10):
        loss=np.array([])
        estmA=np.array([])
        for l in range(Lmin,min(Lmax,min(i,len(sig)-i)),2):
            
            n=np.ones((l,1))
            for j in range(1,order+1):
                n=np.hstack((n,np.power(np.arange(-int(l/2),int(l/2)+1).reshape(-1,1),j)))
                
            A=np.matmul(np.matmul(np.linalg.inv(np.matmul(n.T,n)),n.T),sig[i-int(l/2):i+1+int(l/2)])
            estmA=np.append(estmA,A[0])
            
            estim=np.array([])
            for j in range(-int(l/2),int(l/2)+1):
                estim=np.append(estim,np.sum(A*np.power(j,np.arange(order+1))))
            
            sigma=np.median(sig[i-int(l/2)+1:i+1+int(l/2)]-sig[i-int(l/2):i+int(l/2)])/0.6745
            dfdx=(estim[1:]-estim[:-1])/(sig[i-int(l/2)+1:i+1+int(l/2)]-sig[i-int(l/2):i+int(l/2)])
            loss=np.append(loss,np.sum(estim**2)-2*np.sum(estim*sig[i-int(l/2):i+1+int(l/2)])+2*(sigma**2)*np.sum(dfdx)+1.2*(sigma**2)*np.sum(dfdx**2))
        
        filtered=np.append(filtered,estmA[np.argmin(loss)])
        length=np.append(length,np.arange(Lmin,min(Lmax,min(i,len(sig)-i)),2)[np.argmin(loss)])
      
    filtered_signal=np.append(np.append(sig[:10],filtered),sig[-10:])

    return filtered_signal