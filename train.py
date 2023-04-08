#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 11:06
# @Author  : Chiancc


import argparse
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

from torch.utils.data import TensorDataset, DataLoader
from utils.myutils import print_environment_info, provide_determinism

from utils.logger import *
from utils import torchsummary
import SCMAT
from SCMAT import gmm, tsne, umap

import os


def get_dataloader(df, batch_size):
    y = np.log2(df)

    y = (y - y.mean()) / y.std()  # 对输入数据进行标准化

    train_set = TensorDataset(torch.from_numpy(y.values).to(torch.float32))

    dataloader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            )
    return dataloader


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2) / depth
    return torch.exp(-numerator)


def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()


def run():
    parser = argparse.ArgumentParser(description='SCMAT v1.0')
    parser.add_argument("-i", dest='file_input', default="./input/input.list",
                        help="file input")
    parser.add_argument("-e", dest='epochs', type=int, default=100, help="Number of iterations")
    parser.add_argument("-m", dest='run_mode', default="SCMAT", help="run_mode: SCMAT,GMM,show,map")
    parser.add_argument("-n", dest='cluster_num', type=int, default=-1, help="cluster number")
    parser.add_argument("-w", dest='disc_weight', type=float, default=1e-4, help="weight")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument("-p", dest='other_approach', default="spectral", help="kmeans, spectral, tsne_gmm, tsne")
    parser.add_argument("-t", dest='type', default="pbmc3k", help="dataset type")
    parser.add_argument("-save", dest='save', type=bool, default=False, help="whether save model")
    parser.add_argument("-f", dest='model_cfg', default="./config/model.cfg", help="cancer type: BRCA, GBM")
    parser.add_argument("-k", dest='run_kind', default="result", help="result or test")
    parser.add_argument("-v", dest='verbose', default=False, help="show model")
    parser.add_argument("-model", dest='model', default="linear", help="linear, transformer")
    args = parser.parse_args()
    log.info(str(args))

    checkpoint_path = './model/' + args.type
    os.makedirs(checkpoint_path, exist_ok=True)

    dataset_type = args.type.split('-')[0]

    tmp_dir = './fea/' + dataset_type + '/'
    os.makedirs(tmp_dir, exist_ok=True)  # ./fea/PACA/

    out_file_path = './results/' + dataset_type + '/'  # ./results/PACA/
    os.makedirs(out_file_path, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        log.info("use GPU to train")

    dataset_dict = {"pbmc3k": 8, "pbmc10k": 8, "yao": 8, "cell19": 8,
                    "simulate3": 3, "simulate4": 4, "simulate5": 5}

    if args.run_mode == 'SCMAT':

        if dataset_type not in dataset_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
        elif args.cluster_num == -1:
            args.cluster_num = dataset_dict[dataset_type]

        fea_tmp_file = './fea/' + dataset_type + '.fea'

        ldata = []
        feature_name = []

        omics_data_type = []
        omics_data_size = []
        for line in open(args.file_input, 'rt'):
            base_file = os.path.splitext(os.path.basename(line.rstrip()))[0]
            fea_save_file = tmp_dir + base_file + '.fea'  # ./fea/BRCA/rna.fea
            if os.path.isfile(fea_save_file):
                log.info(f"directly read data: {fea_save_file}")
                df_new = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)
                df_new = df_new.T
                feature_name = df_new.index
            else:
                log.info(f'no {base_file} data type')
                continue
                log.info(f"get {base_file} data from raw file ")
                clinic_params = ['# donor_unique_id', 'project_code', 'donor_vital_status', 'donor_survival_time',
                                 'donor_interval_of_last_followup', 'donor_sex', 'donor_age_at_diagnosis']
                df = pd.read_csv(args.surv_path, header=0, sep='\t',
                                 usecols=clinic_params)  # 2834
                df = df[df['donor_vital_status'].notnull()]  # (2665, 7)
                df['status'] = np.where(df['donor_vital_status'] == 'deceased', 1, 0)
                df['days'] = df.apply(lambda r: r['donor_survival_time'] if r['donor_vital_status'] == 1 else r[
                    'donor_interval_of_last_followup'],
                                      axis=1)

                df = df[df['days'].notnull()]  # (1757, 9)
                df['acronym'] = df['project_code'].apply(lambda x: str(x).split('-')[0])
                df.index = df['# donor_unique_id']

                if dataset_type == 'ALL':
                    pass
                else:
                    df = df.loc[df['acronym'] == dataset_type, ::]

                clinic_save_file = out_file_path + dataset_type + '.clinic'

                df_new = pd.read_csv(line.rstrip(), sep=',', header=0, index_col=0)
                nb_line += 1

                if nb_line == 1:
                    ids = list(df.index)
                    ids_sub = list(df_new)
                    feature_name = list(set(ids) & set(ids_sub))
                    df_clinic = df.loc[
                        feature_name, ['status', 'days', 'donor_sex', 'donor_age_at_diagnosis']]
                    df_clinic.to_csv(clinic_save_file, index=True, header=True, sep=',')
                df_new = df_new.loc[::, feature_name]
                df_new = df_new.fillna(0)
                if 'miRNA' in base_file or 'rna' in base_file:
                    df_new = np.log2(df_new + 1)

                log.info(f"before VarianceThreshold select data type: {base_file}, number: {df_new.shape}", )

                scaler = preprocessing.StandardScaler()  # 对列进行标准化,基因，使样本内的基因之间标准化
                mat = scaler.fit_transform(df_new.values.astype(float))
                df_new.iloc[::, ::] = mat
                df_new = df_new.T  # index 样本,column基因
                selector = VarianceThreshold(threshold=0.8)  # 0.8方差过滤, 按照列方差(基因）过滤
                try:
                    selector.fit(df_new)
                    df_new = df_new.loc[:, selector.get_support()]
                    log.info(f"{base_file} data after VarianceThreshold selector{df_new.shape}")
                except:
                    log.warning(f"no {base_file} data after VarianceThreshold selector")
                    continue
                else:
                    pass
                log.info(f"save {args.type} {base_file} data to {fea_save_file}")
                df_new.to_csv(fea_save_file, index=True, header=True, sep=',')

            log.info(f"cancer type: {args.type}, data type: {base_file}, number: {df_new.shape}")
            omics_data_type.append(base_file)
            omics_data_size.append(df_new.shape[1])
            ldata.append(torch.from_numpy(df_new.values).float())

        start_time = time.time()

        # Loss function
        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()

        #########
        # Initialize model

        model = SCMAT.SCMAT(args.model_cfg, omics_data_type, omics_data_size, model=args.model)

        if use_gpu:
            model = model.cuda()
            mse_loss = mse_loss.cuda()
            bce_loss = bce_loss.cuda()

        latent_dim = model.latent_dim
        batch_size = model.hyperparams["batch_size"]

        # if torch.cuda.is_available():
        #     model.cuda()
        #     for data in ldata:
        #         data.cuda()
        # test_model = SCMAT(args.model_cfg, ['test'], [sum(omics_data_size)])
        # log.info(model)

        if args.verbose:
            torchsummary.summary(model, input_size=[tuple([i]) for i in omics_data_size], batch_size=64, device='cuda',
                                 single=True)

        # Optimizers原来为Adam
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.hyperparams["learning_rate"])

        # for name, parameters in model.named_parameters():
        #     print(name, ':', parameters.size())
        # exit(0)

        # ##########
        #  Training
        # ##########

        real = torch.ones((batch_size, 1)).float()
        fake = torch.zeros((batch_size, 1)).float()
        if use_gpu:
            real = real.cuda()
            fake = fake.cuda()

        loss = []
        if use_gpu:
            ldata = [ldata[i].cuda() for i in range(len(ldata))]

        for epoch in range(args.epochs):
            #  Train Discriminator
            X = []
            # 随机产生样本 64,0 ~ sample number
            # idx = np.random.randint(0, ldata[0].shape[0], batch_size)
            idx = np.arange(0, ldata[0].shape[0])
            np.random.shuffle(idx)

            for i, _ in enumerate(omics_data_size):
                tmp = ldata[i][idx[0:batch_size]]
                # if use_gpu:
                #     tmp = tmp.cuda()
                X.append(tmp)  # 每个组学随机选择64个真实样本用于训练，会重复使用样本

            # ---------------------
            #  Train Discriminator
            # ---------------------
            latent_fake = model.encode(X).detach()

            # Generate a batch of images
            # latent_real = torch.Tensor(np.random.normal(size=(batch_size, latent_dim))).float()
            latent_real = torch.randn(batch_size, latent_dim).float()
            if use_gpu:
                latent_real = latent_real.cuda()

            # Loss measures generator's ability to fool the discriminator
            # test1 = model.disc(latent_real)
            # test2 = model.disc(latent_fake)
            # log.info(f"real {test1.cpu().reshape(-1)}")
            # log.info(f"fake {test2.cpu().reshape(-1)}")
            d_loss_real = bce_loss(model.disc(latent_real), real)  # 和 1相比
            d_loss_fake = bce_loss(model.disc(latent_fake), fake)  # 和 0 相比

            d_loss = 0.5 * torch.add(d_loss_real, d_loss_fake)  # disc loss
            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()

            # -----------------
            # Train Encoder_GAN
            # -----------------

            latent_fake, disc_res, con_x = model(X)
            g_loss = mse_loss(torch.cat(con_x, dim=1), torch.cat(X, dim=1))
            con_loss = g_loss.cpu()
            disc_loss = bce_loss(disc_res, real)
            g_loss += 5 * disc_loss

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            if use_gpu:
                g_loss = g_loss.cpu()
            loss.append(g_loss.detach().numpy())

            if epoch % 10 == 0:
                log.info(f"### epoch:      {epoch},   d_loss_real: {d_loss_real}, d_loss_fake {d_loss_fake}")
                log.info(f"### con_x_loss: {con_loss}   desc_loss: {disc_loss}")
        df = pd.DataFrame(data=[time.time() - start_time])
        time_file = out_file_path + dataset_type + '.SCMAT.time'
        log.info(f"save run time file: file to: {time_file}, run time: {df[0][0]}")
        df.to_csv(time_file, header=True, index=False, sep='\t')

        stamp = int(time.time())
        Timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(stamp))
        if args.save:
            torch.save({'epoch': args.epochs + 1, 'state_dict': model.state_dict(),
                        'best_loss': min(loss), 'optimizer': optimizer.state_dict()},
                       checkpoint_path + '/' + Timestamp + '-' + str("%.4f" % min(loss)) + '.pth.tar')
        # plot loss
        log.info("plot loss figure")
        fig = plt.figure(figsize=(8, 8))

        plt.plot([i for i in range(args.epochs)], loss, 'b-')
        plt.title("loss", fontweight="bold")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        fig.savefig(checkpoint_path + '/' + Timestamp + args.type + '-loss' + '.png')

        with torch.no_grad():
            vec = model.encode(ldata)
        if use_gpu:
            vec = vec.cpu()
        log.info(f"vec:  {vec.shape}")
        vec = vec.detach().numpy()

        vec = pd.DataFrame(vec)
        vec.index = feature_name
        log.info(f"save fusion fea file to: {fea_tmp_file}, feature shape: {vec.shape}")
        vec.to_csv(fea_tmp_file, header=True, index=True, sep='\t')

        if os.path.isfile(fea_tmp_file):
            X = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
            X['SCMAT'] = gmm(args.cluster_num).fit_predict(X.values) + 1
            X = X.loc[:, ['SCMAT']]
            out_file = out_file_path + dataset_type + '.SCMAT'
            log.info(f"save cluster file to : {out_file}", )
            X.to_csv(out_file, header=True, index=True, sep=',')
        else:
            log.warn(f"file does not exist! can't successfully save feature file :{fea_tmp_file}")

    elif args.run_mode == 'show':
        dataset_type = args.type
        out_file_path = './results/' + dataset_type + '/'
        fea_tmp_file = './fea/' + dataset_type + '.fea'
        # tsne_out_file = out_file_path + dataset_type + '.tsne'
        # if os.path.isfile(fea_tmp_file):
        #     df = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
        #     mat = df.values.astype(float)
        #     labels = tsne(mat)
        #     df['x'] = labels[:, 0]
        #     df['y'] = labels[:, 1]
        #     df = df.loc[:, ['x', 'y']]
        #     log.info(f"save tsne file to : {tsne_out_file}")
        #     df.to_csv(tsne_out_file, header=True, index=True, sep=',')
        umap_out_file = out_file_path + dataset_type + '.umap'
        if os.path.isfile(fea_tmp_file):
            df = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
            mat = df.values.astype(float)
            labels = umap(mat)
            df['x'] = labels[:, 0]
            df['y'] = labels[:, 1]
            df = df.loc[:, ['x', 'y']]
            log.info(f"save umap file to : {umap_out_file}")
            df.to_csv(umap_out_file, header=True, index=True, sep=',')
        else:
            log.warn("file does not exist! no feature file %s: ", fea_tmp_file)

    elif args.run_mode == "GMM":
        dataset_type = args.type.split('-')[0]
        out_file_path = './results/' + dataset_type + '/'  # results 目录
        fea_tmp_file = './fea/' + dataset_type + '.fea'
        X = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
        X['SCMAT'] = gmm(args.cluster_num).fit_predict(X.values) + 1
        X = X.loc[:, ['SCMAT']]
        out_file = out_file_path + dataset_type + '.SCMAT'
        log.info(f"save cluster file to : {out_file}", )
        X.to_csv(out_file, header=True, index=True, sep=',')


if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print_environment_info()
    provide_determinism(0)
    run()
