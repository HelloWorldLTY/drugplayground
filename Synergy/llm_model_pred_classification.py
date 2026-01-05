# Here we use a regression task as an example.
import os
import torch
import torch.nn.functional as F
import lightning as L
import os
import scipy.stats
import sklearn.metrics
import pandas as pd
import numpy as np
import pickle
import sklearn.model_selection
from torch.utils.data import DataLoader
from torch import nn
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping



df_grountruth_score = pd.read_csv("./labels_synergy_value.csv")
df_train = pd.read_csv("./deepsynergy_data/d_train_00.csv")
df_test = pd.read_csv("./deepsynergy_data/d_test_00.csv")

with open("./ensem_emb_celllinedrugleave.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("./processed_drug/gamma_embed_mol2_temp0.0.pkl", 'rb') as f:
    drug_name_getembedding = pickle.load(f)


# Please modify this variable for testing fold
test_fold = 0

train_index = df_train.index

train_list = {}
for item in train_index:
    d1, d2, cl = df_train.loc[item]['drug1'], df_train.loc[item]['drug2'], df_train.loc[item]['cell']
    value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    value_list = np.hstack([value_list, 2])
    train_list[item] = value_list

X_train = np.array(list(train_list.values()))
y_train = df_train.loc[train_index]['label'].values

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)
X_tr = X_train 
y_tr = y_train



layers = [10240,4096,1] 
epochs = 1000 
act_func = 'GELU'
dropout = 0.5 
input_dropout = 0.1
eta = 1e-4 
norm = 'tanh' 

test_index = df_test.index

test_list = {}
for item in df_test.index.values:
    d1, d2, cl = df_test.loc[item]['drug1'], df_test.loc[item]['drug2'], df_test.loc[item]['cell']
    value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    value_list = np.hstack([value_list, 2])
    test_list[item] = value_list

X_test = np.array(list(test_list.values()))
y_test = df_test.loc[test_index]['label'].values

n_dim = len(drug_name_getembedding['5-FU'])
cell_dim = 1536


class MultiTaskLoss(torch.nn.Module):



    def __init__(self, is_regression, reduction='none'):

        super(MultiTaskLoss, self).__init__()

        self.is_regression = is_regression

        self.n_tasks = len(is_regression)

        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))

        self.reduction = reduction



    def forward(self, losses):

        dtype = losses.dtype

        device = losses.device

        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)

        self.is_regression = self.is_regression.to(device).to(dtype)

        coeffs = 1 / ( (self.is_regression+1)*(stds**2) )

        multi_task_losses = coeffs*losses + torch.log(stds)



        if self.reduction == 'sum':

            multi_task_losses = multi_task_losses.sum()

        if self.reduction == 'mean':

            multi_task_losses = multi_task_losses.mean()



        return multi_task_losses

class Encoder(nn.Module):

    def __init__(self, k=3):

        super().__init__()

        self.l1 = nn.Sequential(nn.Linear(layers[1], layers[0]), 

                                nn.BatchNorm1d(layers[0]),

                                nn.ReLU(), 

                                nn.Dropout(input_dropout),

                                nn.Linear(layers[0], layers[1]),

                                nn.BatchNorm1d(layers[1]),

                                nn.ReLU(), 

                                nn.Dropout(input_dropout),

                               )





        self.loewe = nn.Linear(layers[1], layers[2])

        self.hsa = nn.Linear(layers[1], layers[2])

        self.bliss = nn.Linear(layers[1], layers[2])

        self.sigm = nn.Sequential(nn.Linear(layers[1], layers[2]),

                                  nn.Sigmoid()

                               )





        self.loewe_input = nn.Linear(n_dim + cell_dim + drug_num_dim, layers[1])

        self.hsa_input = nn.Linear(n_dim+ cell_dim + drug_num_dim, layers[1])

        self.bliss_input = nn.Linear(n_dim+ cell_dim + drug_num_dim, layers[1])

        self.sigm_input = nn.Linear(n_dim+ cell_dim + drug_num_dim, layers[1])

        self.drug_num_emb = nn.Embedding(10, drug_num_dim)



    def forward(self, x):



        x_new = torch.cat([(x[:,0:n_dim] + x[:,n_dim:n_dim*2])/2, x[:,n_dim*2:-1], self.drug_num_emb(x[:,-1].long())], axis=1)
        return {

            'loewe':self.loewe(self.l1(self.loewe_input(x_new))),
            'classify':self.sigm(self.l1(self.sigm_input(x_new)))

        }



    def inference_task(self,x,label):

        if label == 'ri_row':

            x_new = x

            return self.forward(x_new)[label]



        if label == 'ri_col':

            x_new = x

            return self.forward(x_new)[label]

        x_new = x

        output = self.forward(x_new)[label]

        return output





class LitAutoEncoder(L.LightningModule):

    def __init__(self, encoder):

        super().__init__()

        self.encoder = encoder

        self.is_regression = torch.Tensor([True, True, False])

        self.loss_mode = MultiTaskLoss(self.is_regression, reduction = 'mean')



    def training_step(self, batch, batch_idx):

        # training_step defines the train loop.

        x, y = batch

        z = self.encoder(x)

        loss = F.binary_cross_entropy(z["classify"],y.view(x.size(0), -1))

        multitaskloss = loss

        return multitaskloss

    def validation_step(self, batch, batch_idx):

        # this is the validation loop

        x, y = batch

        z = self.encoder(x)

        val_loss = F.binary_cross_entropy(z["classify"],y.view(x.size(0), -1))

        self.log("val_loss", val_loss)

        return val_loss





    def test_step(self, batch, batch_idx):

        # this is the test loop

        x, y = batch

        z = self.encoder(x)



        test_loss = F.binary_cross_entropy(z["classify"],y.view(x.size(0), -1))


        self.log("test_loss", test_loss)

        return test_loss



    def forward(self, x):

        return self.encoder(x)



    def configure_optimizers(self):

            optimizer = torch.optim.Adam(

                params=self.parameters(), 

                lr=eta

            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

                optimizer,

                patience=10,

                verbose=True

            )

            return {

               'optimizer': optimizer,

               'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler

               'monitor': 'val_loss'

           }


# In[22]:


X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)

valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

layers = [10240,4096,1] 

epochs = 1000 

act_func = 'GELU'

dropout = 0.5 

input_dropout = 0.2

eta = 1e-4

norm = 'tanh' 

drug_num_dim = 16


# In[23]:


model = LitAutoEncoder(Encoder())
lr_monitor = LearningRateMonitor(logging_interval='step')
train_loader = DataLoader(train_dataset, batch_size=4096, num_workers=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=1)

# train with both splits

trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)], max_epochs=100)

trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_dataset, batch_size=1024, num_workers=5))


# In[25]:


for m in model.encoder.modules():

    for child in m.children():

        if type(child) == nn.BatchNorm1d:

            child.track_running_stats = False

            child.running_mean = None

            child.running_var = None

model.encoder.eval()
with torch.no_grad():
    y_pred = model.encoder(X_test)['classify'].detach()

import scipy.stats
import sklearn.metrics

print("We use the fold with number:", test_fold)
print(sklearn.metrics.roc_auc_score(y_test, y_pred.t()[0].cpu().numpy()), sklearn.metrics.accuracy_score(y_test, (y_pred.t()[0].cpu().numpy()>0.5)*1))


# In[28]:


y_test


# In[ ]:


for seed in range(0,5):
# for seed in [2,3,4]:
    for temp in [0.0, 0.2,0.4,0.6,0.8,1.0]:
        df_train = pd.read_csv(f"./deepsynergy_data/d_train_{seed}{seed}.csv")

        df_test = pd.read_csv(f"./deepsynergy_data/d_test_{seed}{seed}.csv")
        test_fold = seed

        with open("./ensem_emb_celllinedrugleave.pickle", 'rb') as f:
            cellline_name_getembedding = pickle.load(f)
        with open(f"./processed_drug/qwen_embed_mol2_temp{temp}.pkl", 'rb') as f:
            drug_name_getembedding = pickle.load(f)

        train_index = df_train.index

        train_list = {}
        for item in train_index:
            d1, d2, cl = df_train.loc[item]['drug1'], df_train.loc[item]['drug2'], df_train.loc[item]['cell']
            value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
            value_list = np.hstack([value_list, 2])
            train_list[item] = value_list

        X_train = np.array(list(train_list.values()))
        y_train = df_train.loc[train_index]['label'].values

        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)
        X_tr = X_train 
        y_tr = y_train

        len(drug_name_getembedding['5-FU'])

        X_train.shape

        layers = [10240,4096,1] 
        epochs = 1000 
        act_func = 'GELU'
        dropout = 0.5 
        input_dropout = 0.1
        eta = 1e-4 
        norm = 'tanh' 

        test_index = df_test.index

        test_list = {}
        for item in df_test.index.values:
            d1, d2, cl = df_test.loc[item]['drug1'], df_test.loc[item]['drug2'], df_test.loc[item]['cell']
            value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
            value_list = np.hstack([value_list, 2])
            test_list[item] = value_list

        X_test = np.array(list(test_list.values()))
        y_test = df_test.loc[test_index]['label'].values

        n_dim = len(drug_name_getembedding['5-FU'])
        cell_dim = 1536

        X_test.shape

        # 768*2 + 1536 + 1

        class MultiTaskLoss(torch.nn.Module):



            def __init__(self, is_regression, reduction='none'):

                super(MultiTaskLoss, self).__init__()

                self.is_regression = is_regression

                self.n_tasks = len(is_regression)

                self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))

                self.reduction = reduction



            def forward(self, losses):

                dtype = losses.dtype

                device = losses.device

                stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)

                self.is_regression = self.is_regression.to(device).to(dtype)

                coeffs = 1 / ( (self.is_regression+1)*(stds**2) )

                multi_task_losses = coeffs*losses + torch.log(stds)



                if self.reduction == 'sum':

                    multi_task_losses = multi_task_losses.sum()

                if self.reduction == 'mean':

                    multi_task_losses = multi_task_losses.mean()



                return multi_task_losses

        class Encoder(nn.Module):

            def __init__(self, k=3):

                super().__init__()

                self.l1 = nn.Sequential(nn.Linear(layers[1], layers[0]), 

                                        nn.BatchNorm1d(layers[0]),

                                        nn.ReLU(), 

                                        nn.Dropout(input_dropout),

                                        nn.Linear(layers[0], layers[1]),

                                        nn.BatchNorm1d(layers[1]),

                                        nn.ReLU(), 

                                        nn.Dropout(input_dropout),

                                       )





                self.loewe = nn.Linear(layers[1], layers[2])

                self.hsa = nn.Linear(layers[1], layers[2])

                self.bliss = nn.Linear(layers[1], layers[2])

                self.sigm = nn.Sequential(nn.Linear(layers[1], layers[2]),

                                          nn.Sigmoid()

                                       )





                self.loewe_input = nn.Linear(n_dim + cell_dim + drug_num_dim, layers[1])

                self.hsa_input = nn.Linear(n_dim+ cell_dim + drug_num_dim, layers[1])

                self.bliss_input = nn.Linear(n_dim+ cell_dim + drug_num_dim, layers[1])

                self.sigm_input = nn.Linear(n_dim+ cell_dim + drug_num_dim, layers[1])

                self.drug_num_emb = nn.Embedding(10, drug_num_dim)



            def forward(self, x):



                x_new = torch.cat([(x[:,0:n_dim] + x[:,n_dim:n_dim*2])/2, x[:,n_dim*2:-1], self.drug_num_emb(x[:,-1].long())], axis=1)
                return {

                    'loewe':self.loewe(self.l1(self.loewe_input(x_new))),
                    'classify':self.sigm(self.l1(self.sigm_input(x_new)))

                }



            def inference_task(self,x,label):

                if label == 'ri_row':

                    x_new = x

                    return self.forward(x_new)[label]



                if label == 'ri_col':

                    x_new = x

                    return self.forward(x_new)[label]

                x_new = x

                output = self.forward(x_new)[label]

                return output





        class LitAutoEncoder(L.LightningModule):

            def __init__(self, encoder):

                super().__init__()

                self.encoder = encoder

                self.is_regression = torch.Tensor([True, True, False])

                self.loss_mode = MultiTaskLoss(self.is_regression, reduction = 'mean')



            def training_step(self, batch, batch_idx):

                # training_step defines the train loop.

                x, y = batch

                z = self.encoder(x)

                loss = F.binary_cross_entropy(z["classify"],y.view(x.size(0), -1))

                multitaskloss = loss

                return multitaskloss

            def validation_step(self, batch, batch_idx):

                # this is the validation loop

                x, y = batch

                z = self.encoder(x)

                val_loss = F.binary_cross_entropy(z["classify"],y.view(x.size(0), -1))

                self.log("val_loss", val_loss)

                return val_loss





            def test_step(self, batch, batch_idx):

                # this is the test loop

                x, y = batch

                z = self.encoder(x)



                test_loss = F.binary_cross_entropy(z["classify"],y.view(x.size(0), -1))


                self.log("test_loss", test_loss)

                return test_loss



            def forward(self, x):

                return self.encoder(x)



            def configure_optimizers(self):

                    optimizer = torch.optim.Adam(

                        params=self.parameters(), 

                        lr=eta

                    )

                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

                        optimizer,

                        patience=10,

                        verbose=True

                    )

                    return {

                       'optimizer': optimizer,

                       'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler

                       'monitor': 'val_loss'

                   }

        X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

        train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)

        valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        layers = [10240,4096,1] 

        epochs = 1000 

        act_func = 'GELU'

        dropout = 0.5 

        input_dropout = 0.2

        eta = 1e-4

        norm = 'tanh' 

        drug_num_dim = 16

        model = LitAutoEncoder(Encoder())
        lr_monitor = LearningRateMonitor(logging_interval='step')
        train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=1, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=1)

        # train with both splits

        trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)], max_epochs=100)

        trainer.fit(model, train_loader, valid_loader)

        trainer.test(model, dataloaders=DataLoader(test_dataset, batch_size=1024, num_workers=5))

        for m in model.encoder.modules():

            for child in m.children():

                if type(child) == nn.BatchNorm1d:

                    child.track_running_stats = False

                    child.running_mean = None

                    child.running_var = None

        model.encoder.eval()
        with torch.no_grad():
            y_pred = model.encoder(X_test)['classify'].detach()

        import scipy.stats
        import sklearn.metrics

        print("We use the fold with number:", test_fold)
        print(sklearn.metrics.roc_auc_score(y_test, y_pred.t()[0].cpu().numpy()), sklearn.metrics.accuracy_score(y_test, (y_pred.t()[0].cpu().numpy()>0.5)*1))
        df = pd.DataFrame()
        df['pred'] = y_pred.t()[0]
        df['test'] = y_test

        df.to_csv(f"./outdata2/qwen_embed_mol2_temp{temp}_fold{test_fold}.csv")
