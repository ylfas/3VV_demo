import matplotlib.pyplot as plt
import os

# def loss_plot(args, loss, name):
#     num = args.epoch
#     x = [i for i in range(num)]
#     plot_save_path = r'/t9k/mnt/JBHI_3VV_revision1/loss_picture/'
#     if not os.path.exists(plot_save_path):
#         os.makedirs(plot_save_path)
#     save_loss = plot_save_path + name + '_loss.jpg'
#     plt.figure()
#     plt.plot(x, loss, label= name +'_loss')
#     plt.legend()
#     plt.savefig(save_loss)

# def val_loss_plot(args,loss):
#     num = args.epoch
#     x = [i for i in range(num)]
#     plot_save_path = r'result/plot/'
#     if not os.path.exists(plot_save_path):
#         os.makedirs(plot_save_path)
#     save_loss = plot_save_path+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'_val_loss.jpg'
#     plt.figure()
#     plt.plot(x,loss,label='val_loss')
#     plt.legend()
#     plt.savefig(save_loss)

def metrics_plot(arg,name,*args):
    num = arg.epoch
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/' + str(arg.arch) + '/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.arch) + '_' + str(arg.batch_size) + '_' + str(arg.dataset) + '_' + str(arg.epoch) + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)

