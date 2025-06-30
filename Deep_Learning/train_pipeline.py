# -*- coding: utf-8 -*-
import os
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
def set_seed(seed=3047):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility for certain operations
    torch.backends.cudnn.benchmark = False 
set_seed()

class train_pip(nn.Module):
    def __init__(self,model,criterion):
        super(train_pip,self).__init__()
        self.lr = 1e-4
        self.num_epochs = 20
        self.batch_size = 64
        self.independent_batch = 32
        self.model = model
        self.criterion = criterion
        self.delta = 1e-2
        self.clip_threshold = 2e-1
        self.beta = 2/3
        self.moving_average = 0.25
        """
        implement I-NSGD and base line methods here, baseline methods: SGD, AdaGrad, Adam (three are built-in methods in pytorch)
        Additional baseline methods: NSGD, NSGD with momentum, Clipp SGD.
        
        hyper_parameter: 
            lr: learning rate
            num_epochs; batch_size
            independent_batch: used when INSGD is used
            model: given neural network model we want to train
            criterion: loss function, defeault as CE loss for any classification tasks.
            delta: used in Clip_SGD and I-NSGD, the numerical stablizier for clipping structure.
            clip_threshold: threshold turn adaptive methods into SGD, used in Clip_SGD and I-NSGD
            beta: normalization power for I-NSGD
            moving avegrage: Specialized for momentum update, i.e., NSGDm.
        """
    
    def SGD_train(self, data_train, targets_train):
        """
        SGD update rule:
            w_{t+1} = w_t -\gamma g_t, where g_t is stochastic gradient.

        """
        optimizer = optim.SGD(self.model.parameters(),self.lr)
        self.model.train()
        if torch.cuda.is_available():
            data_train, targets_train = data_train.cuda(), targets_train.cuda()
            self.model.cuda()
        train_loss_list = []
        print("****train start****")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _ in range(len(data_train) // self.batch_size):
                indices = np.arange(len(data_train))

                # Sample two independent mini-batches from the entire dataset
                batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
            
                # Split data and target into two mini-batches
                batch_data, batch_target = data_train[batch_indices], targets_train[batch_indices]
            
                self.model.zero_grad()

                # Compute g_{B_1}: gradient for the first, larger mini-batch
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_loss = epoch_loss/self.batch_size
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")
        return self.model, train_loss_list

    def Adam_train(self, data_train, targets_train):
        """
        Please refer to pytorch official implements and documents for Adam.
        Too long to present here.

        """
        optimizer = optim.Adam(self.model.parameters(),self.lr,betas = (0.9,0.99))
        self.model.train()
        if torch.cuda.is_available():
            data_train, targets_train = data_train.cuda(), targets_train.cuda()
            self.model.cuda()
        train_loss_list = []
        print("****train start****")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _ in range(len(data_train) // self.batch_size):
                indices = np.arange(len(data_train))

                # Sample two independent mini-batches from the entire dataset
                batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
            
                # Split data and target into two mini-batches
                batch_data, batch_target = data_train[batch_indices], targets_train[batch_indices]
            
                self.model.zero_grad()

                # Compute g_{B_1}: gradient for the first, larger mini-batch
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_loss = epoch_loss/len(batch_data)
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")
        return self.model, train_loss_list
    
    def Adagrad_train(self, data_train, targets_train):
        """
        v_t = v_{t-1}+g_t^2
        w_t = w_t - \gamma g_t/\sqrt{v_t}
        

        """
        optimizer = optim.Adagrad(self.model.parameters(),self.lr)
        self.model.train()
        if torch.cuda.is_available():
            data_train, targets_train = data_train.cuda(), targets_train.cuda()
            self.model.cuda()
        train_loss_list = []
        print("****train start****")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _ in range(len(data_train) // self.batch_size):
                indices = np.arange(len(data_train))

                # Sample two independent mini-batches from the entire dataset
                batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
            
                # Split data and target into two mini-batches
                batch_data, batch_target = data_train[batch_indices], targets_train[batch_indices]
            
                self.model.zero_grad()

                # Compute g_{B_1}: gradient for the first, larger mini-batch
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            epoch_loss = epoch_loss/len(batch_data)
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")
        return self.model, train_loss_list

    

    def NSGD_train(self, data_train, targets_train):
        """
        Update Rule:
            w_{t+1} = w_t -\gamma g_t/||g_t||, where g_t is stochastic gradient

        """
        self.model.train()
        if torch.cuda.is_available():
            data_train, targets_train = data_train.cuda(), targets_train.cuda()
            self.model.cuda()
        train_loss_list = []
        print("****train start****")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _ in range(len(data_train) // self.batch_size):
                indices = np.arange(len(data_train))

                # Sample two independent mini-batches from the entire dataset
                batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
            
                # Split data and target into two mini-batches
                batch_data, batch_target = data_train[batch_indices], targets_train[batch_indices]
            
                self.model.zero_grad()

                # Compute g_{B_1}: gradient for the first, larger mini-batch
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                epoch_loss += loss.item()
                loss.backward()
                normalization = sum([param.grad.view(-1).dot(param.grad.view(-1)) for param in self.model.parameters() if param.grad is not None ])
                #g_norms = [(param.grad.norm()+1e-8) for param in self.model.parameters() if param.grad is not None]
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.data -= self.lr*param.grad/(torch.sqrt(normalization)+1e-8)
            epoch_loss = epoch_loss/len(batch_data)
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")
        return self.model, train_loss_list

    def NSGDm_train(self, data_train, targets_train):
        """
        m_t = (1-moving_average_parameter) m_{t-1} + moving_average_parameter g_t
        w_{t+1} = w_t - \gamma m_t/||m_t||

        

        """
        self.model.train()
        momentum_coeff = self.moving_average
        if torch.cuda.is_available():
            data_train, targets_train = data_train.cuda(), targets_train.cuda()
            self.model.cuda()
        train_loss_list = []
        momentum_buffers = {param: torch.zeros_like(param.data) for param in self.model.parameters() if param.requires_grad}
        print("****train start****")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _ in range(len(data_train) // self.batch_size):
                indices = np.arange(len(data_train))

                # Sample two independent mini-batches from the entire dataset
                batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
            
                # Split data and target into two mini-batches
                batch_data, batch_target = data_train[batch_indices], targets_train[batch_indices]
            
                self.model.zero_grad()

                #
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                epoch_loss += loss.item()
                loss.backward()
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # update momentum
                            momentum_buffers[param] = (1 - momentum_coeff) * momentum_buffers[param] + momentum_coeff * param.grad
                    normalization = sum([momentum.view(-1).dot(momentum.view(-1)) for momentum in momentum_buffers if momentum is not None ])
                    for param, momentum in zip(self.model.parameters(), momentum_buffers):
                        normalized_momentum = momentum_buffers[param] / (torch.sqrt(normalization)+ 1e-8)
                        param.data -= self.lr*normalized_momentum
            epoch_loss = epoch_loss/len(batch_data)
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")
        return self.model, train_loss_list
    
    
    def ClipSGD_train(self, data_train, targets_train):
        """
        Update Rule:
            w_{t+1} = w_t - g_t/max{||g_t||+\delta, clip_threshold}

        """
        
        self.model.train()
        if torch.cuda.is_available():
            data_train, targets_train = data_train.cuda(), targets_train.cuda()
            self.model.cuda()
        train_loss_list = []
        print("****train start****")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _ in range(len(data_train) // self.batch_size):
                indices = np.arange(len(data_train))

                # Sample two independent mini-batches from the entire dataset
                batch_indices = np.random.choice(indices, size=self.batch_size, replace=False)
            
                # Split data and target into two mini-batches
                batch_data, batch_target = data_train[batch_indices], targets_train[batch_indices]
            
                self.model.zero_grad()

                # Compute g_{B_1}: gradient for the first, larger mini-batch
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                epoch_loss += loss.item()
                loss.backward()
            
                normalization = sum([param.grad.view(-1).dot(param.grad.view(-1)) for param in self.model.parameters() if param.grad is not None ])
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # Compute the gradient norm
                            # Compute the scaling factor
                            scaling_factor = 1 / max(torch.sqrt(normalization) + self.delta, self.clip_threshold)
                            # Apply the scaled gradient update
                            param.data -= self.lr * scaling_factor * param.grad
            
            epoch_loss = epoch_loss/len(batch_data)
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")
        return self.model, train_loss_list

    

    def INSGD_train(self, all_data, all_targets):
        """
        Update rule:
            w_{t+1} = w_t -\gamma g^1_t/\max{||g_2||+\delta,clip_threshold}**(2/3)

        """
        
        self.model.train()  
        if torch.cuda.is_available():
            all_data, all_targets = all_data.cuda(), all_targets.cuda()
            self.model.cuda()
        train_loss_list = []
        print("****train start****")
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _ in range(len(all_data) // self.batch_size):
                indices = np.arange(len(all_data))

                # Sample two independent mini-batches from the entire dataset
                batch1_indices = np.random.choice(indices, size=self.batch_size, replace=False)
                batch2_indices = np.random.choice(indices, size=self.independent_batch, replace=False)
            
                # Split data and target into two mini-batches
                data_B1, target_B1 = all_data[batch1_indices], all_targets[batch1_indices]
                data_B2, target_B2 = all_data[batch2_indices], all_targets[batch2_indices]

                # clean gradients
                self.model.zero_grad()

                # Compute g_{B_1}: gradient for the first, larger mini-batch
                output_B1 = self.model(data_B1)
                loss_B1 = self.criterion(output_B1, target_B1)
                epoch_loss += loss_B1.item()
                loss_B1.backward()
                
                # used for gradient computation.
                g_B1 = [param.grad.clone() for param in self.model.parameters() if param.grad is not None]

                # Compute g_{B_2} norm: used for normalization.
                self.model.zero_grad()  # Reset gradients for second batch calculation
                output_B2 = self.model(data_B2)
                loss_B2 = self.criterion(output_B2, target_B2)
                loss_B2.backward()
                normalization = sum([param.grad.view(-1).dot(param.grad.view(-1)) for param in self.model.parameters() if param.grad is not None ])

                    # Update parameters using I-NSGD
                    # Update parameters, normalizing each layer’s gradient separately
                with torch.no_grad():
                    for param, g_B1_component in zip(self.model.parameters(), g_B1):
                        if param.grad is not None:
                            normalization_factor = 1/max(torch.sqrt(normalization)+self.delta, self.clip_threshold)**self.beta
                            param.data -= self.lr * g_B1_component*normalization_factor   # Layer-wise normalization
                            # Avoid division by zero

            epoch_loss = epoch_loss/len(data_B1)
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {epoch_loss:.4f}")
        return self.model, train_loss_list



