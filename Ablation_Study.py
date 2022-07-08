import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import time 
import ssl

"""In case there is an error downloading the package"""

ssl._create_default_https_context = ssl._create_unverified_context



"""The Network with Relu as an activation function adapted from The image classification network"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""The network from the Image Classification Tutorial adapted with Leaky Relu"""
class Net_Leaky(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.leaky_relu= nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

""""Train function defined for the K-Fold CV method"""
def train(model,epochs,dataset,criterion,optimizer,current_fold,batch_size):
    """The time stamp is being calculated here

        Loss list, prediction accuracy list and other variables are defined only to keep a track of scores
    
    """
    
    go=time.time()
    loss_list=[]
    pred_accuracy_score = []
    
    final_average_loss=0
    final_average_accuracy=0
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        loss_current = 0.0
        no_of_iter = 0
        right_preds= 0
        for i, data in enumerate(dataset, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_current += loss.item() 
            
            loss.backward()
            optimizer.step()
            no_of_iter += 1

            _, guesses = torch.max(outputs, 1)
            right_preds += (guesses == labels).sum().item()
        
        loss_list.append(loss_current/no_of_iter)
        
        pred_accuracy_score.append(100*(right_preds / (len(dataset)*batch_size)))
        final_average_loss += loss_list[-1]
        final_average_accuracy+= pred_accuracy_score[-1]
        
        """The function not only returns the metrics but also reports them"""

        print("This is epoch {} out {} in fold {} training".format(epoch+1, epochs,current_fold+1))
        print("The CE loss during training is {:.4f}".format(loss_list[-1]))
        print("The accuracy is in training for this epoch is {:.4f}".format(pred_accuracy_score[-1]))

    stop=time.time()
    exect=stop-go
    print("Average loss is {:.4f}".format(final_average_loss/epochs))
    print("Average accuracy is {:.4f}".format(final_average_accuracy/epochs))
    print("Time taken for training {:.4f} seconds".format(exect))
    return final_average_loss/epochs, final_average_accuracy/epochs,exect

""""Validation function is defined here that is called by K folds CV function for the development set"""
def validation(model,data,criterion,current_fold,batch_measure):
    """The time stamps and losses are kept track of by declaring the lists here, much like the function above"""
    
    go=time.time()
    validation_accuracy=[]
    validation_loss=[]
    loss_current = 0.0
    right_preds=0
    no_of_iter = 0

    
    model.eval()
    for i, pick in enumerate(data, 0):
        inputs, labels = pick
        check = model(inputs)
        loss = criterion(check, labels) 
        loss_current  += loss.item()
        _, guessed = torch.max(check.data, 1)
        right_preds += (guessed == labels).sum().item()
        no_of_iter +=1
    

    
    validation_loss.append(loss_current/no_of_iter)
    # Record the Testing accuracy
    validation_accuracy.append((100*(right_preds / (len(data)*batch_measure))))
    
    """This metrics are reported below in a per epoch way as well as they are return as values """
    stop=time.time()
    exect=stop-go
    print("This is the fold {} in training".format(current_fold+1))
    print("The CE loss during training is {:.4f}".format(validation_loss[-1]))
    print("The accuracy in validation for this epoch is {:.4f}".format(validation_accuracy[-1]))
    print("Time taken for validating is {:.4f} seconds".format(exect))   
    return validation_loss[-1],validation_accuracy[-1],exect

"""This is a training function specifically designed for training the development sets, it tracks metrics but also saves the models"""
def train_development(model,train_loader,epochs,criterion,optimizer,model_name,batch_size):
  go=time.time()
  pred_accuracy_score = []
  final_average_accuracy=0
  """
  A list is declared to keep track of the accuracy score and and then a variable is declared 
  
  """

  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    no_of_iter = 0
    right_preds= 0
    for i, data in enumerate(train_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
    # zero the parameter gradients
      optimizer.zero_grad()
    # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      _, guesses = torch.max(outputs, 1)
      right_preds += (guesses == labels).sum().item()
      no_of_iter+=1
      """This has been the basic template adopted for training functions from the image classification tutorial
      
      All variables in this pipeline keep track of the model training and the metrics are stored in lists 
      
      """

    # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0
      
    pred_accuracy_score.append(100*(right_preds / (len(train_loader)*batch_size)))
    print("The accuracy is in training for this epoch is {:.4f}".format(pred_accuracy_score[-1]))
    final_average_accuracy+= pred_accuracy_score[-1]
  stop=time.time()
  exect=stop-go
  print("Training done, time taken for training is {:.4f} seconds".format(exect))
  print(f"Saving The Model Now")
  torch.save(model.state_dict(), model_name+'.pt')
  return exect,final_average_accuracy/epochs

"""This is a function defined to validate the models trained without k-fold on the split holdout set of 10,0000 Images"""
def validation_development(model,data,criterion,batch_measure):
  go=time.time()
  validation_accuracy=[]
  validation_loss=[]
  loss_current = 0.0
  right_preds=0
  no_of_iter = 0
  """The function essentially follows a similar template to others declared
  
  Various lists are defined to keep track of losses and then report and return them
  """

    
  model.eval()
  for i, pick in enumerate(data, 0):
    inputs, labels = pick
    check = model(inputs)
    loss = criterion(check, labels) 
    loss_current  += loss.item()
    _, guessed = torch.max(check.data, 1)
    right_preds += (guessed == labels).sum().item()
    no_of_iter +=1
    
  validation_loss.append(loss_current/no_of_iter)
  # Record the Testing accuracy
  validation_accuracy.append((100*(right_preds / (len(data)*batch_measure))))
  
  stop=time.time()
  exect=stop-go
  print("The CE loss during training is {:.4f}".format(validation_loss[-1]))
  print("The accuracy in validation is {:.4f}".format(validation_accuracy[-1]))
  
  print("Time taken for validating the set is {:.4f} seconds".format(exect))
  return exect,validation_accuracy[-1]

"""This is the K-Folds Cross Validation Model implemented in py-torch

  The model essentially calculates indexes and then passes it on functions outside to train and validate

  It also maintains the records of each training and validation for each fold for each of the models

"""
def kfcv(trainer,loss,method_otp,load_dat,kf_val,batch_measure):

    """Various lists are defined here to store the average loss per fold in trainig
    
      Accuracy, that is, the evaluation metric is also kept a track of through a list 

      Time is also stored in a list and returned through a list so that it can be kept a track record of 

    """

    kf_cv_training_loss=[]
    kf_cv_training_accuracy=[]
    
    kf_cv_validation_loss=[]
    kf_cv_validation_accuracy=[]
    
    time_training_per_fold=[]
    time_validation_list=[]

    """"This is to check the total length of the data and then it is divided by 1/no. of folds to obtain the length of splits"""
    req_len = len(load_dat)
    fraction=1/kf_val
    limit = int(req_len * (fraction))
    
    """The for loop implemented below helps to seperate out chunks of the data supplied"""
    """Chunk is used to calculate the indexes as well, with iterations it helps in random splits"""
    """Ranger approaches train indexes from the left  and other end stores the other end of the first train index"""
    """Close here  and current chunk hold the values for validation indices, from the left and then the right"""
    """copy_curr and total are used as outer bounds for the other train indices"""
    chunk=0
    while chunk < kf_val:
        
        ranger = 0 
        other_end = chunk * limit
        close_here = other_end
        curr_chunk = (chunk * limit) + limit
        copy_curr = curr_chunk
        total= req_len


        """Loss values are declared here again, these will be passed into the list at the end of each fold
        
          Thus they refresh in each iteration

         """ 
        average_loss_training=0
        average_accuracy_training=0
        average_loss_validation=0
        average_accuracy_validation=0
        
        time_training=0
        time_validation=0
        
        print(f"#####################################################")
        print(f"The data has been split and this is fold"+" "+str(chunk+1))
        print(f"The Data Summary is as follows")
        print("We are training from {}-{}*excluding and {}-{}*excluding".format(ranger,other_end,copy_curr,total))
        print("And we are testing in {}-{}*excluding".format(close_here,curr_chunk))
        
        """gathering indexes from left and right for training and then putting them into lists"""
        lr_ind = range(ranger,other_end)
        rr_ind= range(copy_curr,total)
        
        lr_ind=list(lr_ind)
        rr_ind=list(rr_ind)

        """These are combined train lists values which will be passed into train loader below"""
        total_all=lr_ind+rr_ind

        print(f"The total length of training sample is"+" "+str(len(total_all)))

        """Keep holding is basically keeping in the validation indices"""
        keep_holding= range(close_here,curr_chunk)        
        keep_holding= list(keep_holding)
        
        print("The total length of validation sample is"+" "+str(len(keep_holding)))
        
        """"Now the data is sub-settted and passed into loaders"""

        sampled_for_training = torch.utils.data.Subset(load_dat,total_all)
        measure_against = torch.utils.data.Subset(load_dat,keep_holding)
        
      
        
        
        train_loader = torch.utils.data.DataLoader(sampled_for_training, batch_size=batch_measure,
                                          shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(measure_against, batch_size=batch_measure,
                                          shuffle=True, num_workers=2)
        
        """Printing summaries and passing it into functions outside to test and validate for each cycle"""
        print("There are"+" "+str(batch_measure)+" "+"batches of length"+" "+str(len(train_loader))+" "+"for training")
        print("There are"+" "+str(batch_measure)+" "+"batches of length"+" "+str(len(val_loader))+" "+"for validation")
        average_loss_training,average_accuracy_training,time_training=train(trainer,15,train_loader,loss,method_otp,chunk,batch_measure)
        average_loss_validation,average_accuracy_validation,time_validation=validation(trainer,val_loader,loss,chunk,batch_measure)
        kf_cv_training_loss.append(average_loss_training)
        kf_cv_training_accuracy.append(average_accuracy_training)
        
        kf_cv_validation_loss.append(average_loss_validation)
        kf_cv_validation_accuracy.append(average_accuracy_validation)
        
        time_training_per_fold.append(time_training)
        time_validation_list.append(time_validation)
        """The lists are appended at each fold and the values are then returned in forms of lists"""

        if chunk==kf_val:
          break
        chunk+=1
    
    return kf_cv_training_loss, kf_cv_training_accuracy, kf_cv_validation_loss, kf_cv_validation_accuracy,time_training_per_fold,time_validation



if __name__ == '__main__':
    ## cifar-10 dataset
    print("implementing ReLU vs Leaky ReLU with (negative slope alpha =0.1)")

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    

    development_set,holdout_set = torch.utils.data.random_split(trainset, [40000, 10000])
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    
    Relu_Model = Net()
    Leaky_Relu_Model = Net_Leaky()
    
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_Relu = optim.SGD(Relu_Model.parameters(), lr=0.001, momentum=0.9)
    optimizer_Leaky_Relu = optim.SGD(Leaky_Relu_Model.parameters(), lr=0.001, momentum=0.9)
    


    """K-Fold method for the Image Classification network with ReLU"""
    print(f"##########################################################")
    print(f"Beginning the K-fold process for Relu based Image classification network ")
    relu_training_loss, relu_training_accuracy, relu_validation_loss, relu_validation_accuracy,relu_time_training,relu_time_validation=kfcv(Relu_Model,criterion,optimizer_Relu,development_set,3,20)
    """K-Fold method for the Image Classification network with Leaky ReLU"""
    print(f"##########################################################")
    print(f"Beginning the K-fold process for Leaky ReLU based Image classification network ")
    leaky_relu_training_loss, leaky_relu_training_accuracy, leaky_relu_validation_loss, leaky_relu_validation_accuracy,leaky_relu_time_training,leaky_relu_time_validation=kfcv(Leaky_Relu_Model,criterion,optimizer_Leaky_Relu,development_set,3,20)
    print(f"###########################################################")
    print(f"Now Beginning the training for models on the development set")
    
    """Passing the development set into a different loader and the holdout into another for training and validation"""
    trainloader_development = torch.utils.data.DataLoader(development_set, batch_size=20,
                                            shuffle=True, num_workers=2)
    holdout_development = torch.utils.data.DataLoader(holdout_set, batch_size=20,
                                            shuffle=True, num_workers=2)
    
    """New Model instances are declared and new optimizers are declared"""
    
    Relu_2= Net()
    Leaky_2 = Net_Leaky()
    optimizer_Relu_dev = optim.SGD(Relu_2.parameters(), lr=0.001, momentum=0.9)
    optimizer_Leaky_Relu_dev = optim.SGD(Leaky_2.parameters(), lr=0.001, momentum=0.9)
    
    print(f"Training the Relu based Image Classification network on the development set")
    time_stamp_relu,accuracy_relu=train_development(Relu_2,trainloader_development,15,criterion,optimizer_Relu_dev,"Relu_model_development_set",20)
    print("##########################################################################")
    
    print("Relu Model saved")
    time_stamp_leaky_relu,accuracy_leaky=train_development(Leaky_2,trainloader_development,15,criterion,optimizer_Leaky_Relu_dev,"Leaky_Relu_model_development_set",20)
    print("##########################################################################")
   
    print("Leaky Relu Model saved")
    print(f"############################################################")
    print(f"Now beginning the validation on Holdout set for both the models")
    print(f"Validation for the Relu Model")
    Relu_2.load_state_dict(torch.load('Relu_model_development_set.pt'))
    time_relu_dev,val_relu_dev=validation_development(Relu_2,holdout_development,criterion,20)
    print(f"Validation for the Leaky Relu Model")
    Leaky_2.load_state_dict(torch.load('Leaky_Relu_model_development_set.pt'))
    time_leaky_relu_dev,val_leaky_relu_dev=validation_development(Leaky_2,holdout_development,criterion,20)
    print("###########################################################")
    print("Summary for Relu Model using K Folds")
    print("For Relu model in k fold the average accuracy per respecect folds was")
    print(relu_training_accuracy)
    print("For Relu model in k fold the average loss per respecect folds was")
    print(relu_training_loss)
    print("The time taken to train was")
    print(relu_time_training)
    print("Validation accuracy was")
    print(relu_validation_accuracy)
    print("###########################################################")
    print("Summary for Leaky Relu Model using K Folds")
    print("For Leaky Relu model in k fold the average accuracy per respecect folds was")
    print(leaky_relu_training_accuracy)
    print("For Relu model in k fold the average loss per respecect folds was")
    print(leaky_relu_training_loss)
    print("The time taken to train was")
    print(leaky_relu_time_training)
    print("Validation accuracy was")
    print(leaky_relu_validation_accuracy)
    print("###########################################################")
    print("Summary for Relu Model on development and validation on holdout")
    print("The time taken to train on development set for RELU was")
    print(time_stamp_relu)
    print("The validation accuracy was")
    print(val_relu_dev)
    print("###########################################################")
    print("Summary for Leaky Relu Model on development and validation on holdout")
    print("The time taken to train on development set for RELU was")
    print(time_stamp_leaky_relu)
    print("The validation accuracy was")
    print(val_leaky_relu_dev)
    print("###########################################################")
    print("Overall comments on the exercise")
    print("The average validation accuracy was higher for both ReLU and Leaky ReLU in the K-Folds for 15 Epochs per fold training")
    print("The K-folds was in that regard was more effective but as can be seen the accuracy in last fold goes high ")
    print("There's significant difference between the training and validation, sometimes to the tune of 10% or hgiher")
    print("But essentially Leaky Relu was found to be a better optimizer")
    print("The training time was much higher in K fold when compared to the development and holdout validation with only marginal gains of ~4-7% in holdout accuracy")
    print("In effect K-Folds is a better way to train then but overall the results are very similar but the evalution metric drops more b/w training and testing in K-Folds")
    print("In essence these problems are also subject to design, so in effect a regularizer might help with the problem of training-testing loss")

