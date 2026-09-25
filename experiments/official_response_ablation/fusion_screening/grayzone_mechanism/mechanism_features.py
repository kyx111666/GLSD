"""Read sealed intermediate arrays; reconstruct, never replace, default evidence."""
import numpy as np

RADII=(.5,1.,2.,3.)
GAMMAS=(.25,.5,.75,1.)


def scale_details(feature,a0,rho,gamma):
    peaks,expected=feature.evidence(a0,rho)
    window=max(1,round(rho*feature.k));tol=max(1,round(gamma*feature.k))
    widths=list(feature.effective_scales)
    aligned=np.zeros((len(peaks),len(widths)))
    matched=np.zeros(aligned.shape,bool)
    distance=np.full(aligned.shape,-1,int)
    nearest=np.full(aligned.shape,-1,int)
    for j,width in enumerate(widths):
        other=feature.effective_scales[width][0]
        values=feature.local_cache[width,window]
        for i,p in enumerate(peaks):
            d=np.abs(other-p)
            if len(d):nearest[i,j]=int(d.min())
            eligible=np.flatnonzero(d<=tol)
            if len(eligible):
                ix=min(eligible,key=lambda ix:(int(d[ix]),-values[ix]))
                aligned[i,j]=values[ix];distance[i,j]=d[ix];matched[i,j]=True
    med=np.median(aligned,axis=1)
    if gamma==.5:
        np.testing.assert_allclose(med,expected[:,1],rtol=1e-13,atol=1e-14)
    present=np.array([np.median(v[m]) if m.any() else 0. for v,m in zip(aligned,matched)])
    return dict(widths=widths,tolerance=tol,window=window,aligned=aligned.tolist(),
        matched=matched.tolist(),distance=distance.tolist(),nearest_distance=nearest.tolist(),
        median=med.tolist(),mean=aligned.mean(1).tolist(),matched_only_median=present.tolist())


def candidate_rows(feature,a0,rho):
    peaks,values=feature.evidence(a0,rho)
    details={str(g):scale_details(feature,a0,rho,g) for g in GAMMAS}
    return dict(a0=a0,rho=rho,k=feature.k,peaks=peaks.tolist(),G=values[:,0].tolist(),
                L=values[:,1].tolist(),scales=details)


def binary_logloss(y,p):
    p=np.clip(p,1e-12,1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def fit_predict(xtrain,ytrain,train_subjects,xtest):
    """Fixed ridge logistic diagnostic; training-only normalization, equal subject weight."""
    xtrain=np.asarray(xtrain,float);xtest=np.asarray(xtest,float);y=np.asarray(ytrain,float)
    subjects,counts=np.unique(train_subjects,return_counts=True)
    sizes=dict(zip(subjects,counts))
    w=np.array([1/sizes[s] for s in train_subjects]);w/=w.sum()
    mu=(xtrain*w[:,None]).sum(0)
    sd=np.sqrt(((xtrain-mu)**2*w[:,None]).sum(0));sd=np.maximum(sd,1e-8)
    x=np.column_stack((np.ones(len(xtrain)),(xtrain-mu)/sd))
    t=np.column_stack((np.ones(len(xtest)),(xtest-mu)/sd))
    beta=np.zeros(x.shape[1]);prior=np.clip(np.dot(w,y),1e-6,1-1e-6)
    beta[0]=np.log(prior/(1-prior));pen=np.diag([0]+[1e-3]*(x.shape[1]-1))
    def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-40,40)))
    def objective(b):return np.dot(w,binary_logloss(y,sigmoid(x@b)))+.5*b@pen@b
    for _ in range(60):
        p=sigmoid(x@beta);grad=x.T@(w*(p-y))+pen@beta
        h=x.T@((w*p*(1-p))[:,None]*x)+pen+np.eye(x.shape[1])*1e-10
        step=np.linalg.solve(h,grad);rate=1.;current=objective(beta)
        while rate>1e-6 and objective(beta-rate*step)>current:rate*=.5
        beta-=rate*step
        if np.max(np.abs(rate*step))<1e-7:break
    return sigmoid(t@beta),dict(coefficients=beta.tolist(),train_mean=mu.tolist(),train_sd=sd.tolist(),ridge=1e-3)


def logit_scores(x):
    p=np.clip(np.asarray(x,float),1e-4,1-1e-4)
    return np.log(p/(1-p))
