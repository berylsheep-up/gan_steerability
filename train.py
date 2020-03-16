import logging
import utils.logging
from utils import util
import sys
import graphs
import time
import importlib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from IPython import embed
import tensorflow as tf 

sys.path.append('resources/stylegan')
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import config
from training import dataset,misc
from metrics import metric_base
from dnnlib import EasyDict
import copy
from training import training_loop

if 1:
    desc          = 'stylegan'                                                                 # 包含在结果子目录名称中的描述字符串。
    train         = EasyDict(run_func_name='train.joint_train')         # 训练过程设置。
    G             = EasyDict(func_name='training.networks_stylegan.G_style')               # 生成网络架构设置。
    D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # 判别网络架构设置。
    G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # 生成网络优化器设置。
    D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # 判别网络优化器设置。
    G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating')           # 生成损失设置。
    D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) # 判别损失设置。
    dataset_args  = EasyDict()                                                             # 数据集设置，在后文确认。
    sched         = EasyDict()                                                             # 训练计划设置，在后文确认。
    grid          = EasyDict(size='4k', layout='random')                                   # setup_snapshot_image_grid()相关设置。
    metrics       = [metric_base.fid50k]                                                   # 指标方法设置。
    submit_config = dnnlib.SubmitConfig()                                                  # dnnlib.submit_run()相关设置。
    tf_config     = {'rnd.np_random_seed': 1000}                                           # tflib.init_tf()相关设置。
    train.total_kimg = 25000

    # 数据集。
    desc += '-character';     dataset_args = EasyDict(tfrecord_dir='character');                 #train.mirror_augment = True
    
    # GPU数量。
    desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
    #desc += '-2gpu'; submit_config.num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
    #desc += '-4gpu'; submit_config.num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    #desc += '-8gpu'; submit_config.num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

    # 默认设置。

    sched.lod_initial_resolution = 8
    sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(is_train=True, dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc

def train(g, graph_inputs, output_dir, save_freq=100):
    # configure logging file
    logging_file = os.path.join(output_dir, 'log.txt')
    utils.logging.configure(logging_file, append=False)

    Loss_sum = 0;
    n_epoch = 1
    optim_iter = 0
    batch_size = constants.BATCH_SIZE
    num_samples = graph_inputs[g.z].shape[0]
    loss_values = []

    # train loop
    for epoch in range(n_epoch):
        for batch_start in range(0, num_samples, batch_size):
            start_time = time.time()

            s = slice(batch_start, min(num_samples, batch_start + batch_size))
            graph_inputs_batch = util.batch_input(graph_inputs, s)
            zs_batch = graph_inputs_batch[g.z]

            # loop through the graph and target alphas returned,
            # we can have g.get_train_alpha return lists for
            # alpha_for_graph and alpha_for_target, which corresponds to
            # training with different alphas for the same zs
            # make sure that the alphas are ordered correctly though,
            # particularly for NN
            alpha_for_graph, alpha_for_target = g.get_train_alpha(zs_batch)
            if not isinstance(alpha_for_graph, list):
                alpha_for_graph = [alpha_for_graph]
                alpha_for_target = [alpha_for_target]
            for ag, at in zip(alpha_for_graph, alpha_for_target):
                feed_dict_out = graph_inputs_batch
                out_zs = g.sess.run(g.outputs_orig, feed_dict_out)

                target_fn, mask_out = g.get_target_np(out_zs, at)
                feed_dict = graph_inputs_batch
                feed_dict[g.alpha] = ag
                feed_dict[g.target] = target_fn
                feed_dict[g.mask] = mask_out
                curr_loss, _ = g.sess.run([g.loss, g.train_step], feed_dict=feed_dict)
                Loss_sum = Loss_sum + curr_loss
                loss_values.append(curr_loss)

            elapsed_time = time.time() - start_time

            logging.info('T, epc, bst, lss, a: {}, {}, {}, {}, {}'.format(
                elapsed_time, epoch, batch_start, curr_loss, at))

            if (optim_iter % save_freq == 0) and (optim_iter > 0):
                g.saver.save(g.sess, './{}/model_{}.ckpt'.format(
                    output_dir, optim_iter*batch_size),
                    write_meta_graph=False, write_state=False)

            optim_iter = optim_iter + 1

    if optim_iter > 0:
        print('average loss with this metric: ', Loss_sum/(optim_iter*batch_size))
    g.saver.save(g.sess, "./{}/model_{}_final.ckpt".format(
        output_dir, num_samples),
        write_meta_graph=False, write_state=False)
    return loss_values


def joint_train(
    submit_config,
    opt,
    metric_arg_list,
    sched_args              = {},       # 训练计划设置。
    grid_args               = {},       # setup_snapshot_image_grid()相关设置。
    dataset_args            = {},       # 数据集设置。
    total_kimg              = 15000,    # 训练的总长度，以成千上万个真实图像为统计。
    drange_net              = [-1,1],   # 将图像数据馈送到网络时使用的动态范围。
    image_snapshot_ticks    = 1,        # 多久导出一次图像快照？
    network_snapshot_ticks  = 10,       # 多久导出一次网络模型存储？
    D_repeats               = 1,        # G每迭代一次训练判别器多少次。
    minibatch_repeats       = 4,        # 调整训练参数前要运行的minibatch的数量。
    mirror_augment          = False,    # 启用镜像增强？
    reset_opt_for_new_lod   = True,     # 引入新层时是否重置优化器内部状态（例如Adam时刻）？
    save_tf_graph           = False,    # 在tfevents文件中包含完整的TensorFlow计算图吗？
    save_weight_histograms  = False,    # 在tfevents文件中包括权重直方图？
    resume_run_id           = None,     # 运行已有ID或载入已有网络pkl以从中恢复训练，None = 从头开始。
    resume_snapshot         = None,     # 要从哪恢复训练的快照的索引，None = 自动检测。
    resume_kimg             = 0.0,      # 在训练开始时给定当前训练进度。影响报告和训练计划。
    resume_time             = 0.0,     # 在训练开始时给定统计时间。影响报告。
    *args,
    **kwargs
    ):

    output_dir = opt.output_dir

    graph_kwargs = util.set_graph_kwargs(opt)

    graph_util = importlib.import_module('graphs.' + opt.model + '.graph_util')
    constants = importlib.import_module('graphs.' + opt.model + '.constants')

    model = graphs.find_model_using_name(opt.model, opt.transform)
    g = model(submit_config=submit_config, dataset_args=dataset_args, **graph_kwargs, **kwargs)
    g.initialize_graph()

    # create training samples
    #num_samples = opt.num_samples
    # if opt.model == 'biggan' and opt.biggan.category is not None:
    #     graph_inputs = graph_util.graph_input(g, num_samples, seed=0, category=opt.biggan.category)
    # else:
    #     graph_inputs = graph_util.graph_input(g, num_samples, seed=0)



    w_snapshot_ticks = opt.model_save_freq

    ctx = dnnlib.RunContext(submit_config, train)
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)
    
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    
    # 设置快照图像网格
    print('Setting up snapshot image grid...')
    grid_size, grid_reals, grid_labels, grid_latents = misc.setup_snapshot_image_grid(g.G, training_set, **grid_args)
    sched = training_loop.training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
    grid_fakes = g.Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus)
    # 建立运行目录
    print('Setting up run dir...')
    misc.save_image_grid(grid_reals, os.path.join(submit_config.run_dir, 'reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)
    misc.save_image_grid(grid_fakes, os.path.join(submit_config.run_dir, 'fakes%06d.png' % resume_kimg), drange=drange_net, grid_size=grid_size)
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        g.G.setup_weight_histograms(); g.D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)
    # 训练
    print('Training...\n')
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    loss_values = []
    while cur_nimg < total_kimg * 1000:
        if ctx.should_stop(): break

        # 选择训练参数并配置训练操作。
        sched = training_loop.training_schedule(cur_nimg=cur_nimg, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
        training_set.configure(sched.minibatch // submit_config.num_gpus, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                g.G_opt.reset_optimizer_state(); # D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # 进行训练。
        for _mb_repeat in range(minibatch_repeats):
            alpha_for_graph, alpha_for_target = g.get_train_alpha(constants.BATCH_SIZE)
            if not isinstance(alpha_for_graph, list):
                alpha_for_graph = [alpha_for_graph]
                alpha_for_target = [alpha_for_target]
            for ag, at in zip(alpha_for_graph, alpha_for_target):
                feed_dict_out = graph_util.graph_input(g, constants.BATCH_SIZE, seed=0)
                out_zs = g.sess.run(g.outputs_orig, feed_dict_out)

                target_fn, mask_out = g.get_target_np(out_zs, at)
                feed_dict = feed_dict_out
                feed_dict[g.alpha] = ag
                feed_dict[g.target] = target_fn
                feed_dict[g.mask] = mask_out
                feed_dict[g.lod_in] = sched.lod
                feed_dict[g.lrate_in] = sched.D_lrate
                feed_dict[g.minibatch_in] = sched.minibatch
                curr_loss, _, Gs_op, G_op = g.sess.run([g.joint_loss, g.train_step, g.Gs_update_op, g.G_train_op], feed_dict=feed_dict)
                loss_values.append(curr_loss)
            
            cur_nimg += sched.minibatch
            #tflib.run([g.Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
            #tflib.run([g.G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})

        # 每个tick执行一次维护任务。
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = ctx.get_time_since_last_update()
            total_time = ctx.get_time_since_start() + resume_time

            # 报告进度。
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %-4.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # 保存快照。
            if cur_tick % image_snapshot_ticks == 0 or done:
                grid_fakes = g.Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus)
                misc.save_image_grid(grid_fakes, os.path.join(submit_config.run_dir, 'fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if cur_tick % network_snapshot_ticks == 0 or done or cur_tick == 1:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((g.G, g.D, g.Gs), pkl)
                metrics.run(pkl, run_dir=submit_config.run_dir, num_gpus=submit_config.num_gpus, tf_config=tf_config)
            if cur_tick % w_snapshot_ticks == 0 or done:
                g.saver.save(g.sess, './{}/model_{}.ckpt'.format(
                    output_dir, (cur_nimg // 1000)),
                    write_meta_graph=False, write_state=False)

            # 更新摘要和RunContext。
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            ctx.update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = ctx.get_last_update_interval() - tick_time

    # 保存最终结果。
    misc.save_pkl((g.G, g.D, g.Gs), os.path.join(submit_config.run_dir, 'network-final.pkl'))
    summary_log.close()

    ctx.close()

    loss_values = np.array(loss_values)
    np.save('./{}/loss_values.npy'.format(output_dir), loss_values)
    f, ax  = plt.subplots(figsize=(10, 4))
    ax.plot(loss_values)
    f.savefig('./{}/loss_values.png'.format(output_dir))


if __name__ == '__main__':
    opt = TrainOptions().parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # train loop
    kwargs.update(opt=opt, **kwargs)
    dnnlib.submit_run(**kwargs)
    # loss_values = joint_train(g, graph_inputs, output_dir, opt.model_save_freq, **kwargs)
    # loss_values = np.array(loss_values)
    # np.save('./{}/loss_values.npy'.format(output_dir), loss_values)
    # f, ax  = plt.subplots(figsize=(10, 4))
    # ax.plot(loss_values)
    # f.savefig('./{}/loss_values.png'.format(output_dir))