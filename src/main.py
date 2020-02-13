import torch
import cv2
import numpy as np
import math

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture
from sklearn import mixture

import torch.nn as nn

from utils.img import Crop
from utils.debugger import Debugger
from utils.hmParser import parseHeatmap
from utils.horn87 import horn87

def draw_ellipse(pt_2d, cov_2d, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    """

    # print("debug: pos {}".format(pt_2d))
    # print("debug: covariance {}".format(cov_2d))

    eigval, eigvec = np.linalg.eig(cov_2d)

    try:
        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0
    except:
        print("INFO: no eigval")

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # avoid sqrt of small number
    tol = 1e-3
    eigval[np.abs(eigval) < tol] = 0

    try:
        a = math.sqrt(eigval[bigind])
        b = math.sqrt(eigval[smallind])

        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
        R = np.matrix([[math.cos(angle), math.sin(angle)],
                       [-math.sin(angle), math.cos(angle)]])
        fx = R * np.matrix([x, y])

        px = np.array(fx[0, :] + pt_2d[0]).flatten()
        py = np.array(fx[1, :] + pt_2d[1]).flatten()
    except:
        px = pt_2d[0]
        py = pt_2d[1]
    finally:
        ax.plot([pt_2d[0]], [pt_2d[1]], "o", color='red', markersize=10)
        ax.plot(px, py, "--g")

def set_bn_to_eval(layer):
    if type(layer) == nn.modules.batchnorm.BatchNorm2d:
        # print(type(layer))
        layer.training = False

def set_dropout_to_train(layer):
    if type(layer) == nn.modules.dropout.Dropout:
        #print(type(layer))
        layer.training = True

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    """

    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    # for nsig in range(1, 4):
    #     ax.add_patch(Ellipse(position, nsig * width, nsig * height,
    #                          angle, **kwargs))

    #draw 3 sigma error ellipse
    nsig = 3
    ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                         angle, **kwargs, zorder=1))

    ax.scatter(position[0], position[1], c="red", s=50, zorder=2)

def plot_ellipses(means, covars, ax, weights):
    eig_vals, eig_vecs = np.linalg.eigh(covars)
    unit_eig_vec = eig_vecs / np.linalg.norm(eig_vecs)
    angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
    # Ellipse needs degrees
    angle = 180 * angle / np.pi
    # eigenvector normalization
    eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
    ax.add_patch(Ellipse(means, eig_vals[0], eig_vals[1],
                              180 + angle, edgecolor='b',
                              lw=4, fill=True, alpha=weights))

def plot_gmm(gmm, X, label=True, ax=None):

    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    
    # if label:
    #     ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # else:
    #     ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    # ax.axis('equal')

    mc_dropout_trial_num = 100.0

    confidence_max = 0

    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        #prune redundant component
        clust_id = gmm.predict(np.reshape(pos, (1, -1)))
        # print("INFO: cluster id: {}".format(clust_id))
        cluster_samples = np.where(labels == clust_id)[0]
        cluster_num = len(cluster_samples)
        #w_factor = 1 / gmm.weights_.max()
        w_factor = cluster_num / mc_dropout_trial_num
        #print("INFO: detection rate: {}".format(w_factor*w))
        if np.sum(w) > 0.01 and w_factor > 0.2:
            confidence = w_factor*w
            if confidence > confidence_max:
                confidence_max = confidence

    #w_factor = 1 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        #prune redundant component
        clust_id = gmm.predict(np.reshape(pos, (1, -1)))
        # print("INFO: cluster id: {}".format(clust_id))
        cluster_samples = np.where(labels == clust_id)[0]
        cluster_num = len(cluster_samples)
        #w_factor = 1 / gmm.weights_.max()
        w_factor = cluster_num / mc_dropout_trial_num
        #print("INFO: detection rate: {}".format(w_factor*w))
        if np.sum(w) > 0.01 and w_factor > 0.2:
            alpha = w_factor*w / confidence_max
            draw_ellipse(pos, covar, edgecolor='b', lw=4, fill=True, alpha=alpha)
            #draw_ellipse(pos, covar, edgecolor='b', lw=4, fill=True, alpha=w_factor*w)
            #plot_ellipses(pos, covar, ax, w_factor*w)

def uncertainty_test(model, input_var, heat_thresh, ax):

    model.train()
    model.apply(set_dropout_to_train)

    T = 300

    all_kps = None
    gmm_component_num = 0

    # do sampling
    for i in range(T):
        output = model(input_var)
        hm = output[-1].data.cpu().numpy()

        ps = parseHeatmap(hm[0], heat_thresh)
        kp_num = len(ps[0])
        if kp_num > gmm_component_num:
            gmm_component_num = kp_num

        for k in range(kp_num):
            kp = [ps[1][k] * 4, ps[0][k] * 4]
            if all_kps is None:
                all_kps = kp
            else:
                all_kps = np.vstack((all_kps, kp))

    #print("debug: gmm_component_num {}".format(gmm_component_num))
    #print("debug: all kp {}".format(all_kps[:, 0]))
    #exit()

    #gmm = GaussianMixture(n_components=gmm_component_num, covariance_type='full', random_state=42).fit(all_kps)
    
    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=gmm_component_num,
                                            covariance_type='full',
                                            weight_concentration_prior_type="dirichlet_process",
                                            init_params='kmeans',
                                            mean_precision_prior=1,
                                            weight_concentration_prior=None).fit(all_kps)
    plot_gmm(dpgmm, all_kps, ax)

    # print("debug: cov {}".format(dpgmm.covariances_))
    # print("debug: gmm {}".format(gmm_component_num))
    # print("active components: {}".format(np.sum(dpgmm.weights_ > 0.01)))

    #return gmm.means_, gmm.covariances_, gmm.weights_

def main():

    # use the model trained with dropout enabled
    model_path = '/home/erl/moshan/orcvio_gamma/orcvio_gamma/pytorch_models/starmap/trained_models/with_dropout/model_cpu.pth'
    img_path = './images/car2.png'
    det_name = './det/car2.png'

    # by default img size is 256
    inputRes = 256
    outputRes = 64
    CUDA = torch.cuda.is_available()

    model = torch.load(model_path)

    img = cv2.imread(img_path)
    s = max(img.shape[0], img.shape[1])*1.0
    c = np.array([img.shape[1]/2., img.shape[0]/2.])
    
    # img = cv2.resize(img, (320, 240))
    # print(img.shape)
    
    # crop only change h, w, c to c, h, w for images with size 256 x 256
    img = Crop(img, c, s, 0, inputRes).astype(np.float32).transpose(2, 0, 1) / 256.
    input = torch.from_numpy(img.copy()).float()
    
    # change to b, c, h, w
    input = input.view(1, input.size(0), input.size(1), input.size(2))
    input_var = torch.autograd.Variable(input).float()

    if CUDA:
        model.cuda()
        input_var = input_var.cuda()

    output = model(input_var)
    hm = output[-1].data.cpu().numpy()

    # convert to bgr, uint8 for display
    img = (input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8).copy()
    inp = img.copy()

    # hm[0, 0] is an image, since 1st dim is batch 
    star = (cv2.resize(hm[0, 0], (inputRes, inputRes)) * 255)
    
    # clip the values to 0-255
    star[star > 255] = 255
    star[star < 0] = 0
    
    # tile Construct an array by repeating A the number of times given by reps.
    # convert to 3 channels, for bgr
    star = np.tile(star, (3, 1, 1)).transpose(1, 2, 0)
    trans = 0.8
    star = (trans * star + (1. - trans) * img).astype(np.uint8)
    
    # select peaks and perform nms
    heat_thresh = 0.25
    ps = parseHeatmap(hm[0], heat_thresh)
    canonical, pred, color, score = [], [], [], []

    # mc dropout
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.imshow(img)
    uncertainty_test(model, input_var, heat_thresh, ax1)

    for k in range(len(ps[0])):
        # camviewfeature
        x, y, z = ((hm[0, 1:4, ps[0][k], ps[1][k]] + 0.5) * outputRes).astype(np.int32)
        dep = ((hm[0, 4, ps[0][k], ps[1][k]] + 0.5) * outputRes).astype(np.int32)
        canonical.append([x, y, z])
    
        pred.append([ps[1][k], outputRes - dep, outputRes - ps[0][k]])
        # kp confidence score
        score.append(hm[0, 0, ps[0][k], ps[1][k]])
  
        color.append((1.0 * x / outputRes, 1.0 * y / outputRes, 1.0 * z / outputRes))
        
        # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) → img
        # -1 means that a filled circle is to be drawn
        cv2.circle(img, (ps[1][k] * 4, ps[0][k] * 4), 6, (0, 0, 255), -1)
        cv2.circle(img, (ps[1][k] * 4, ps[0][k] * 4), 2, (int(z*4), int(y*4), int(x*4)), -1)
    
        # plot cov
        # pos = kps_mean[k]
        # covar = kps_cov[k]
        # draw_ellipse(pos, covar, ax1)

    plt.axis('off')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    plt.show()
    f1.savefig('kp_cov.png', bbox_inches='tight', pad_inches = 0)
    # plt.pause(5)

    pred = np.array(pred).astype(np.float32)
    canonical = np.array(canonical).astype(np.float32)

    pointS = canonical * 1.0 / outputRes
    pointT = pred * 1.0 / outputRes

    # calculate viewpoint
    R, t, s = horn87(pointS.transpose(), pointT.transpose(), score)
    
    rotated_pred = s * np.dot(R, canonical.transpose()).transpose() + t * outputRes

    # # init contains some plotting stuff
    # debugger = Debugger()
    # # inp is input img
    # debugger.addImg(inp, 'inp')
    # # star is heatmap
    # debugger.addImg(star, 'star')
    # # nms is kp peask
    # debugger.addImg(img, 'nms')
    
    # debugger.addPoint3D(canonical / outputRes - 0.5, c = color, marker = '^')
    # debugger.addPoint3D(pred / outputRes - 0.5, c = color, marker = 'x')
    # debugger.addPoint3D(rotated_pred / outputRes - 0.5, c = color, marker = '*')

    # debugger.showAllImg(pause = False)
    # debugger.show3D()

    # cv2.imwrite(det_name, img)

if __name__ == '__main__':
    main()
