### This file includes the functions for plotting

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_acc_loss(his1,his2,legend1='adm',legend2='sgd',title1='Linear classifier accuracy',title2='Linear classifier loss'):
    plt.clf()
    plt.subplot(121)
    plt.subplots_adjust(top=1.2, bottom=0.08, left=0.10, right=3.0, hspace=0.2, wspace=0.15)
    plt.plot(his1.history['acc'])
    plt.plot(his2.history['acc'])
    plt.title(title1, fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend([legend1, legend2], loc='upper left', fontsize=20)
    # plt.savefig('Acc1.jpg')

    plt.subplot(122)
    plt.plot(his1.history['loss'])
    plt.plot(his2.history['loss'])
    plt.title(title2, fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend([legend1, legend2], loc='upper left', fontsize=20)
    # plt.savefig('Loss1.jpg')
    plt.subplots_adjust(top=1.2, bottom=0.08, left=0.10, right=3.0, hspace=0.15, wspace=0.15)
    
def plot_trainval(his):
    plt.subplot(121)
    plt.plot(his.history['acc'])
    plt.plot(his.history['val_acc'])
    plt.title('model accuracy', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(['train', 'test'], loc='upper left', fontsize=20)

    plt.subplot(122)
    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.title('model loss', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(['train', 'test'], loc='upper left', fontsize=20)
    plt.subplots_adjust(top=1.2, bottom=0.08, left=0.10, right=3.0, hspace=0.15,wspace=0.15)
    
def visual_solution(solution):
    plt.clf()
    fig = plt.figure(figsize=(15,4))
    ax1 = fig.add_subplot(131)
    ax1.imshow(solution[:,0].reshape(72,72),aspect='auto', cmap = cm.Greys)
    ax1.set_title('Rectangle')
    ax2 = fig.add_subplot(132)
    ax2.imshow(solution[:,1].reshape(72,72),aspect='auto', cmap = cm.Greys)
    ax2.set_title('Disk')
    ax3 = fig.add_subplot(133)
    ax3.set_title('Triangle')
    ax3.imshow(solution[:,2].reshape(72,72),aspect='auto', cmap = cm.Greys) 
