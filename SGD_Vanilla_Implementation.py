
import torch
import time
import torch.nn as nn

"""
This function defines RMSE Loss function


"""


def RMSE(predictions,ground_truth):
    """The RMSE loss is calculated using a square root of MSE Loss"""
    loss = nn.MSELoss()
    msq = torch.sqrt(loss(predictions, ground_truth))

    return msq
"""
This function defines the polynomial function to generate data


"""
def polynomial_fun(w,x):
    """This is to calculate the required polynomial y, returns the ground truth"""
    y=0
    x=torch.tensor(x)
    for i in range(0,len(w)):
        y += w[i]*x**i
    
    return y
"""
This function returns the training data when called 

"""

def generator():

    """
    Generating x for training data


    """
    training=[]
    for i in range(0,100):
        x = torch.distributions.uniform.Uniform(-20,20).sample([1]) 
        training.append(x)

    training=torch.stack(training)


    """
    Generating x for testing data

    """

    testing=[]
    for i in range(0,50):
        x = torch.distributions.uniform.Uniform(-20,20).sample([1]) 
        testing.append(x)
    
    testing=torch.stack(testing)

    """Declaring a weight tensor"""
    w=torch.tensor([1,2,3,4])

    """
    generating the function as per instruction set by adding noise

    """
    true_curve=polynomial_fun(w, training)
    noise = torch.distributions.normal.Normal(0, 0.2).sample((len(true_curve),1))
    t=torch.add(noise,true_curve)

    """"
    generating the testing curve by adding noise too

    """

    true_curve_testing=polynomial_fun(w,testing)
    noise_two = torch.distributions.normal.Normal(0, 0.2).sample((len(true_curve_testing),1))
    t_testing=torch.add(noise_two,true_curve_testing)


    """ Adding Noise"""

    """Returning values"""

    return t,t_testing,training,testing,true_curve,true_curve_testing,w


"""
Fitting through least squares using the w argmin

"""

def fit_polynomial_ls(x,t,pol_degree):
    """The time is calculated in the begining and returned along with weight hat vector"""
    go=time.time()
    training=torch.cat([x**i for i in range(0,pol_degree+1)],dim=1)

    """"Solution here is calculated using the normal equation"""
    
    weight_hat=torch.matmul(torch.matmul(torch.inverse(torch.matmul(torch.transpose(training,0,1),training)),torch.transpose(training,0,1)),t)
    stop=time.time()
    exect = stop - go
    return weight_hat,exect

"""
Fitting through SGD

"""

def fit_polynomial_sgd(x,t,pol_degree,learning_rate,batch_size):
    """"Time is registered before training is started and the SGD is calculated by calculating a gradient of a hypothesis function"""
    """The hypothesis here is to look at a polynomial of degree 4, but the function is hard coded to work for the specific problem"""
    go=time.time()
    training_size=len(x)
    training=torch.cat([x**i for i in range(0,pol_degree+1)],dim=1)
    import torch.nn.functional as F
    loss_function = F.mse_loss
    
    """Random Values are allocated to the weight"""
    w_grad=torch.randn([5,1])
    
    """Running it for 300,000 epochs in batches of size 32"""
    for epochs in range(300000):
        loss_append=0
        for step in range(0,training_size,batch_size):
            """Training size and batch size basically are parameters that help with the minibatch algorithm"""
            "The threshold is added here to calculate steps in dividing the batches here "
            
            threshold = step + batch_size
            train_batch, target_batch = training[step:threshold, :], t[step:threshold, -1:]
            
            pred=torch.matmul(train_batch,w_grad)
            error=(target_batch.flatten()-pred.flatten())
            msq=loss_function(target_batch,pred).mean()
            """We calculated gradient by differentiation the loss with respect to the gradient vector """
            loss_append+=msq.item()
            n_eeta= (-learning_rate*((-2*torch.matmul(error,train_batch))/len(train_batch))).unsqueeze(1)
            w_grad+=n_eeta
            """Reporting the loss per 10000 epochs"""
        if epochs%10000==0:
          print("Epoch Loss till {} is {:.4f} ".format(epochs,(loss_append/(training_size/len(train_batch))/1000)))
    stop=time.time()
    exect = stop - go
    print("Time {:.4f} seconds".format(exect))
    """Returns a weight hat vector along with the time stamp"""
    return w_grad,exect


""""The Section below is the execution of the script"""
print(f"#############################################")
print(f"Generating data")

"""Returned values are extracted from the generator function """
t,t_testing,training,testing,true_curve,true_curve_testing,w=generator()


"""Below the function specified for Least Squares fit Method is called and returned values are kept"""
print(f"The data has been now generated, now fitting ls method")
weight_hat,time_stamp_ls=fit_polynomial_ls(training,t,4)
"""The predictions for Y is calculated using the returned weight vector for both training and testing"""
y_ls_trained=polynomial_fun(weight_hat,training)
y_ls_testing=polynomial_fun(weight_hat,testing)

"""This section calculates the required differences between true curve and observed curve"""
print(f"############################################")
mean_t_minus_y=(t-true_curve).mean()
std_t_minus_y=(t-true_curve).std()

"""This section calculates the required differences between the true curve and least squares curves from testing"""
mean_ls_minus_y=(y_ls_testing-true_curve_testing).mean()
std_ls_minus_y=(y_ls_testing-true_curve_testing).std()

print(f"Statistics for observed and true curve(given)")
print("The mean is {:.4f}".format(mean_t_minus_y))
print("The standard deviation is {:.4f}".format(std_t_minus_y))
print("#############################################")
print(f"Statistics for ls predicted and true curve")
print("The mean is {:.4f}".format(mean_ls_minus_y))
print("The standard deviation is {:.4f}".format(std_ls_minus_y))
print(f"############################################")
print(f"The data has been now generated, now fitting SGD")

"""Now the SGD function is being called and the returned values are stored in variables"""

weight_hat_2, time_stamp_sgd=fit_polynomial_sgd(training,t,4,0.00000000001, 8)
print(f"SGD was trained for 300000 epochs with a batch size of 8 and learning rate of 0.00000000001")

"""Predictions are made using the returned weight vector for both training and testing and the required metrics are calculated"""
y_sgd_trained=polynomial_fun(weight_hat_2,training)
y_sgd_testing=polynomial_fun(weight_hat_2,testing)
mean_sgd_minus_y=(y_sgd_testing-true_curve_testing).mean()
std_sgd_minus_y=(y_sgd_testing-true_curve_testing).std()
print(f"Statistics for SGD predicted and true curve")
print("The mean is {:.4f}".format(mean_sgd_minus_y))
print("The standard deviation is {:.4f}".format(std_sgd_minus_y))

"""
Calculating RMSE between predictions done by LS and SGD
And the Weight Vectors Retreived through both of them

"""
ls_rmse = RMSE(y_ls_testing,t_testing)
w_ls_rmse = RMSE(weight_hat,torch.cat((w,torch.tensor([0])),dim=0).unsqueeze(1))
sgd_rmse= RMSE(y_sgd_testing,t_testing)
w_sgd_rmse= RMSE(weight_hat_2,torch.cat((w,torch.tensor([0])),dim=0).unsqueeze(1))

print(f"###############################################")
print(f"RMSE statistics for LS Method")
print("The RMSE for ls curve vs ground truth is {:.4f}".format(ls_rmse))
print("The RMSE for ls Weight vs actual weight is {:.4f}".format(w_ls_rmse))
print(f"################################################")
print(f"RMSE statistics for SGD Method")
print("The RMSE for sgd curve vs ground truth is {:.4f}".format(sgd_rmse))
print("The RMSE for sgd Weight vs actual weight is {:.4f}".format(w_sgd_rmse))
print(f"#################################################")
print(f"Summary Statistics in terms of training time")
print("Time taken to train LS method is {:.4f} seconds".format(time_stamp_ls))
print("Time taken to train the SGD method was {:.4f} seconds".format(time_stamp_sgd))
print(f"###################################################")
print(f"Observations and overall comments")
print(f"Overall the LS fit method was found to be more accurate for a small dataset")
print(f"For the given polynomial, which is non convex, the SGD failed to converge properly")
print(f"As per the question, we couldn't use optimizers which provide an improvement over the vanilla SGD")
