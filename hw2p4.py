import numpy as np
import matplotlib.pyplot as plt

rv_gen = np.random.default_rng(seed=20221107)

# generating synthetic data
n = 50
p = 1000
Sigma = np.zeros((p,p))
for i in range(Sigma.shape[0]):
    for j in range(Sigma.shape[1]):
        Sigma[i,j] = 0.5**abs(i-j)


x_array = rv_gen.multivariate_normal(np.zeros((p)), Sigma, size=50)
# print(x_array.shape) #(50, 1000)
beta = np.array([1.5, 0, 0.5, 0, 1] + [0 for _ in range(p-5)])
y_array = x_array @ beta + rv_gen.multivariate_normal(np.zeros((n)), 0.04*np.identity(n))
# print(y_array.shape) #(50,)


def average_loss(x: np.array, y: np.array, fitted_beta: np.array):
    n = x.shape[0]
    return sum((y - x@fitted_beta)**2)**0.5 / n

def lasso_loss(x: np.array, y: np.array, lambda_val: int|float, fitted_beta: np.array):
    return average_loss(x,y,fitted_beta) + lambda_val*sum([abs(b) for b in fitted_beta])

def l0_norm(a: np.array):
    return np.count_nonzero(a)

def shooting(x: np.array, y: np.array, lambda_val: int|float, initial_beta: list, epsilon=1e-5, learning_rate=0.1):
    n = x.shape[0]
    p = x.shape[1]
    xtx = np.transpose(x)@x
    xty = np.transpose(x)@y

    beta_plus = np.array([b if b>0 else 0 for b in initial_beta])
    beta_minus = np.array([-b if b<0 else 0 for b in initial_beta])
    beta_tilde = np.concatenate((beta_plus, beta_minus), axis=0)
    # print(beta_tilde.shape) #(2000,)

    num_iter = 0
    delta_norm_vec = []
    while(True):
        num_iter += 1
        gradLt_term1 = (xtx@beta_plus - xtx@beta_minus - xty)/n
        gradLt_plus = gradLt_term1 + lambda_val*np.ones((p), dtype="float32")
        gradLt_minus = -gradLt_term1 + lambda_val*np.ones((p), dtype="float32")
        gradLt_tilde = np.concatenate((gradLt_plus, gradLt_minus), axis=0)

        # print(beta_tilde.shape, gradLt_tilde.shape)
        delta = np.max(np.c_[-beta_tilde, -gradLt_tilde], axis=1)
        delta_norm_vec.append(sum(delta**2)**0.5)
        new_beta_tilde = beta_tilde + delta*learning_rate
        
        if np.isinf((sum(delta**2))**0.5): #diverge
            break
        
        # if num_iter%10 == 0:
        #     print((sum(delta**2))**0.5, lasso_loss(x, y, lambda_val, new_beta_tilde[0:p]-new_beta_tilde[p:2*p]))
        
        
        if np.mean(delta_norm_vec[-5:]) < epsilon:
            # print("iter:", num_iter)
            return new_beta_tilde[0:p]-new_beta_tilde[p:2*p]
        else:
            beta_tilde = new_beta_tilde    
            beta_plus = new_beta_tilde[0:p]
            beta_minus = new_beta_tilde[p:2*p]
            # print(beta_plus>=0)
            # print(beta_minus>=0)



if __name__=="__main__":
    initial_beta = [0 for _ in range(p)]
    lambda_candid = np.arange(0, 1.01, 0.05)
    # lambda_candid = [0, 0.05, 0.1]

    l0_norm_vec = []
    training_average_quad_loss_vec = []
    training_average_lasso_loss_vec = []
    testing_average_quad_loss_vec = []
    testing_average_lasso_loss_vec = []

    for lambda_val in lambda_candid:
        print("lambda_val:", lambda_val)
        l0_norm_at_lambda = []
        training_quad_loss_at_lambda = []
        training_lasso_loss_at_lambda = []
        testing_quad_loss_at_lambda = []
        testing_lasso_loss_at_lambda = []
        
        for batch_idx in range(10): #5-cross-validation
            test_index = [batch_idx*5+i for i in range(5)]
            train_index = [i for i in range(batch_idx*5)] + [i for i in range(test_index[-1]+1, 50)]

            train_x_array = x_array[train_index,:]
            train_y_array = y_array[train_index]
            test_x_array = x_array[test_index,:]
            test_y_array = y_array[test_index]

            beta_fit = shooting(train_x_array, train_y_array, lambda_val, initial_beta, learning_rate=0.03)
            l0_norm_at_lambda.append(l0_norm(beta_fit))
            training_quad_loss_at_lambda.append(average_loss(train_x_array, train_y_array, beta_fit))
            training_lasso_loss_at_lambda.append(lasso_loss(train_x_array, train_y_array, lambda_val, beta_fit))
            testing_quad_loss_at_lambda.append(average_loss(test_x_array, test_y_array, beta_fit))
            testing_lasso_loss_at_lambda.append(lasso_loss(test_x_array, test_y_array, lambda_val, beta_fit))

            # print("lambda:", lambda_val, " l0:", l0_norm(beta_fit))
            # print("average_quad_loss:", average_loss(x_array, y_array, beta_fit))
            # print("average_lasso_loss:", lasso_loss(x_array, y_array, lambda_val, beta_fit))

        
        l0_norm_vec.append(np.mean(l0_norm_at_lambda))
        training_average_quad_loss_vec.append(np.mean(training_quad_loss_at_lambda))
        training_average_lasso_loss_vec.append(np.mean(training_lasso_loss_at_lambda))
        testing_average_quad_loss_vec.append(np.mean(testing_quad_loss_at_lambda))
        testing_average_lasso_loss_vec.append(np.mean(testing_lasso_loss_at_lambda))

    print(lambda_candid)
    print(l0_norm_vec)
    print(training_average_quad_loss_vec)
    print(training_average_lasso_loss_vec)
    print(testing_average_quad_loss_vec)
    print(testing_average_lasso_loss_vec)

    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(lambda_candid, l0_norm_vec)
    ax[0,0].set_title("l0 norm")
    ax[0,1].plot(lambda_candid, training_average_quad_loss_vec)
    ax[0,1].set_title("training_quad_loss")
    ax[0,2].plot(lambda_candid, training_average_lasso_loss_vec)
    ax[0,2].set_title("training_lasso_loss")

    ax[1,1].plot(lambda_candid, testing_average_quad_loss_vec)
    ax[1,1].set_title("testing_quad_loss")
    ax[1,2].plot(lambda_candid, testing_average_lasso_loss_vec)
    ax[1,2].set_title("testing_lasso_loss")
    plt.show()