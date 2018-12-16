### This file includes the functions for preprocessing, visualization,
### and generating dataset in 'Image Denosing Problem'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mp1 import generate_dataset_regression

def normalize_perim(y):
    #print(y)
    U=y[[0,2,4]]
    V=y[[1,3,5]]
    order=np.argsort(V)
    U_sort=U[order]
    V_sort=V[order]
    return [U_sort[0], V_sort[0], U_sort[1], V_sort[1], U_sort[2], V_sort[2]]


def custom_visual_pred(x, y, y_predict, str):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((72, 72))
    ax.imshow(I, extent=[-0.15, 1.15, -0.15, 1.15], cmap='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    xy = y.reshape(3, 2)
    tri = patches.Polygon(xy, closed=True, fill=False, edgecolor='r', linewidth=5, alpha=0.5)
    ax.add_patch(tri)

    xy_predict = y_predict.reshape(3, 2)
    tri_predict = patches.Polygon(xy_predict, closed=True, fill=False, edgecolor='g', linewidth=5, alpha=0.5)
    ax.add_patch(tri_predict)

    plt.legend(['original vertex', 'predicted vertex'], loc='upper left')
    plt.savefig(str)

    plt.show()


def vertex_predict(index,model,X_test,Y_test,x_test):
    y_tp_1 = model.predict(x_test[index][:, np.newaxis])
    custom_visual_pred(X_test[index], Y_test[index], y_tp_1, 'vertex_{}'.format(index))


def generate_test_set_regression():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(300, 20)
    return [X_test, Y_test]


def generate_a_drawing(figsize, U, V, noise=0.0):
    fig = plt.figure(figsize=(figsize,figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    plt.close(fig)
    return imdata

def generate_a_rectangle(noise=0.0, free_location=False):
    figsize = 1.0
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    picture_noise=generate_a_drawing(figsize, U, V, noise)
    picture=generate_a_drawing(figsize, U, V)
    return [picture_noise,picture]


def generate_a_disk(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    picture_noise=generate_a_drawing(figsize, U, V, noise)
    picture=generate_a_drawing(figsize, U, V)
    return [picture_noise,picture]

def generate_a_triangle(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    picture_noise = generate_a_drawing(figsize, U, V, noise)
    picture=generate_a_drawing(figsize, U, V)
    return [picture_noise,picture]



def generate_dataset_denoising(nb_samples, noise_low,noise_high,free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle()[0].shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples,im_size])
    #print('Creating data:')
    for i in range(nb_samples):
        noise=np.random.uniform(noise_low,noise_high)
        category = np.random.randint(3)
        if category == 0:
            [X[i], Y[i]] = generate_a_rectangle(noise, free_location)
        elif category == 1:
            [X[i], Y[i]] = generate_a_disk(noise, free_location)
        else:
            [X[i], Y[i]] = generate_a_triangle(noise, free_location)
    X = (X + noise) / (255 + 2 * noise)
    Y = Y/255.0
    return [X, Y]

def generate_testset_denoising(noise_low,noise_high):
    # Getting im_size:
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_denoising(300,noise_low,noise_high, True)
    return [X_test, Y_test]

def visualize_predhourglass(x,str):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((72,72))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    plt.savefig(str)
    plt.show()