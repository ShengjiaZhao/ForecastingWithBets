import torch
import os
import torchvision
from torchvision import datasets, transforms
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class Dataset():
    def __init__(self, batch_size, max_train_size=0):
        self.train_x = None
        self.train_y = None
        self.train_z = None
        
        self.test_x = None
        self.test_y = None
        self.test_z = None
        
        self.train_ptr = 0
        self.test_ptr = 0
        self.batch_size = batch_size
        self.max_train_size = max_train_size
    
    def train_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.train_ptr + batch_size > self.train_x.shape[0]:
            self.train_ptr = 0
        if self.max_train_size > 0 and self.train_ptr + batch_size > self.max_train_size:
            self.train_ptr = 0
        bx, by, bz = self.train_x[self.train_ptr:self.train_ptr+batch_size], \
                     self.train_y[self.train_ptr:self.train_ptr+batch_size], \
                     self.train_z[self.train_ptr:self.train_ptr+batch_size]
        self.train_ptr += batch_size
        return bx.type(torch.float32), by.type(torch.float32), bz.type(torch.float32)

    def test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size 
        if self.test_ptr + batch_size > self.test_x.shape[0]:
            self.test_ptr = 0
        bx, by, bz = self.test_x[self.test_ptr:self.test_ptr+batch_size], \
                     self.test_y[self.test_ptr:self.test_ptr+batch_size], \
                     self.test_z[self.test_ptr:self.test_ptr+batch_size]
        self.test_ptr += batch_size
        return bx.type(torch.float32), by.type(torch.float32), bz.type(torch.float32)
    
    def test_size(self):
        return self.test_x.shape[0]
    
    def test_all(self):
        return self.test_x.type(torch.float32), self.test_y.type(torch.float32), self.test_z.type(torch.float32)
  
    

class FlightDataset(Dataset):
    def __init__(self, device, batch_size=128, max_train_size=0):
        super(FlightDataset, self).__init__(batch_size, max_train_size)
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'flights.csv'), low_memory=False)
        df = df[df['AIRLINE'] == 'WN']
        variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR', 
                       'MONTH','DAY','DAY_OF_WEEK', 'AIR_SYSTEM_DELAY',
                       'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                       'WEATHER_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
                       'FLIGHT_NUMBER', 'TAIL_NUMBER', 'AIR_TIME', 'AIRLINE', 'DEPARTURE_TIME', 'ARRIVAL_TIME', 'ELAPSED_TIME']
        df.drop(variables_to_remove, axis = 1, inplace = True)
        df.dropna(inplace = True)
        
        # Convert categorical variables to one-hot
        origin_onehot = pd.get_dummies(df["ORIGIN_AIRPORT"],prefix='origin',drop_first=True)
        dest_onehot = pd.get_dummies(df["DESTINATION_AIRPORT"],prefix='destination',drop_first=True)
        df.drop(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], axis=1, inplace=True)
        
        # Obtain real valued variables. Shift cyclic variables (time) a little 
        feature = df[['SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'DISTANCE', 'SCHEDULED_ARRIVAL']]
        feature = feature.values
        feature[:, 3] += 2400 * (feature[:, 3] < 500).astype(np.float)   
        feature = feature - np.mean(feature, axis=0, keepdims=True)
        feature = feature / np.std(feature, axis=0, keepdims=True)
        
        feature = np.concatenate([feature, origin_onehot.values, dest_onehot.values], axis=1)
        target = df[['ARRIVAL_DELAY']].values
        target = (target > 20).astype(np.float32)
        
        test_size = 10000
        if max_train_size == 0:
            max_train_size = feature.shape[0] - test_size
            
        total_size = max_train_size + test_size
        selected = np.zeros(target.shape[0], dtype=np.int)
        selected[:total_size] = 1
        selected = np.random.permutation(selected)
        
        feature = feature[selected == 1]
        target = target[selected == 1]

        z = np.reshape((np.argsort(np.argsort(feature[:, 0])) * 10.0 / feature.shape[0]).astype(np.int), [-1, 1])

        x = torch.from_numpy(feature).to(device).type(torch.float32)
        y = torch.from_numpy(target).to(device).type(torch.float32)
        z = torch.from_numpy(z).to(device).type(torch.float32)
        self.train_x, self.train_y, self.train_z = x[:-test_size], y[:-test_size], z[:-test_size]
        self.test_x, self.test_y, self.test_z = x[-test_size:], y[-test_size:], z[-test_size:]
        
        self.x_dim = feature.shape[1]
            
        
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    dataset = MnistDataset(device=torch.device('cpu'))
    bx, by, bz = dataset.train_batch(128)
    
    plt.subplot(1, 3, 1)
    plt.hist(bx, bins=30)
    plt.subplot(1, 3, 2)
    plt.hist(by)
    plt.subplot(1, 3, 3)
    plt.hist(bz)
    plt.show()
    

       
   
