import sys

from sklearn.metrics import r2_score
import numpy as np

import os

import scanpy as sc
import cpa

sc.settings.set_figure_params(dpi=100)

import argparse

parser = argparse.ArgumentParser(description="A sample program demonstrating argparse.")
parser.add_argument("--inputemb", type=str, default=1, help="Number of times to process the file.")

args = parser.parse_args()

data_path = './tahoe_100m_test.h5ad'
# read_path_new = "./embeddings_mol3/embeddings_mol3_gemini_temperature0.0.pkl"
read_path_new = args.inputemb

from importlib import reload

adata = sc.read(data_path)


import pandas as pd

seed = 0
adata.X = adata.layers['counts'].copy()


# In[13]:


cpa.CPA.setup_anndata(adata,
                      perturbation_key='drug_list',
                      dosage_key='dosage',
                      control_group='DMSO_TF',
                      smiles_key = 'drug_list',
                      batch_key=None,
                      is_count_data=True,
                      categorical_covariate_keys=['cell_line'],
                      deg_uns_key='rank_genes_groups',
                      deg_uns_cat_key='cov_drug_dose',
                      max_comb_len=2,
                     )

ae_hparams = {
    "n_latent": 1536,
    "recon_loss": "nb",
    "doser_type": "logsigm",
    "n_hidden_encoder": 512,
    "n_layers_encoder": 3,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 3,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": True,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.1,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": seed,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 3,
    "mixup_alpha": 0.1,
    "adv_steps": 2,
    "n_hidden_adv": 64,
    "n_layers_adv": 2,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 20.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": False,
    "gradient_clip_value": 1.0,
    "step_size_lr": 45,
}

model = cpa.CPA(adata=adata,
                split_key='split',
                train_split='train',
                valid_split='valid',
                test_split='test',
                gene_embeddings = [[0.]],
                use_rdkit_embeddings=True,
                use_gene_emb = False,
                read_path = read_path_new,
                **ae_hparams,
               )



read_path_new = read_path_new.replace('/', '_')
model.train(max_epochs=20,
            use_gpu=True,
            batch_size=128,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=5,
            save_path=f'./cpa_out_smiles_tahoe_{read_path_new}/',
          )

sc.settings.verbosity = 3

model.predict(adata, batch_size=256)



import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm

n_top_degs = [10, 20, 50, None] # None means all genes
results = defaultdict(list)
ctrl_adata = adata[adata.obs['drug_list'] == 'DMSO_TF'].copy()
for cat in tqdm(adata.obs['drug_list'].unique()):
    if 'DMSO_TF' not in cat:
        cat_adata = adata[adata.obs['drug_list'] == cat].copy()

        deg_cat = f'{cat}'
        deg_list = adata.uns['rank_genes_groups']['names'][deg_cat]

        x_true = cat_adata.layers['counts'].toarray()
        x_pred = cat_adata.obsm['CPA_pred']
        x_ctrl = ctrl_adata.layers['counts'].toarray()

        x_true = np.log1p(x_true)
        x_pred = np.log1p(x_pred)
        x_ctrl = np.log1p(x_ctrl)

        for n_top_deg in n_top_degs:
            if n_top_deg is not None:
                degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
            else:
                degs = np.arange(adata.n_vars)
                n_top_deg = 'all'
                
            x_true_deg = x_true[:, degs]
            x_pred_deg = x_pred[:, degs]
            x_ctrl_deg = x_ctrl[:, degs]
            
            r2_mean_deg = r2_score(x_true_deg.mean(0), x_pred_deg.mean(0))
            r2_var_deg = r2_score(x_true_deg.var(0), x_pred_deg.var(0))

            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
            r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))

            # cov, cond, dose = cat.split('_')
            cov, cond, dose = cat,cat,cat
            
            results['cell_type'].append(cov)
            results['condition'].append(cond)
            results['dose'].append(dose)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_var_deg'].append(r2_var_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
            results['r2_var_lfc_deg'].append(r2_var_lfc_deg)

df = pd.DataFrame(results)

read_path_new = read_path_new.replace('/', '_')
df.to_csv(f"./output/cpa_checkseed_SMILES_tahoe100m_{seed}_{read_path_new}.csv")
