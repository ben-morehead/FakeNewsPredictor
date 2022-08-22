import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms


class Pipeline():
    #The pipeline controls the overall flow of the data

    def __init__(self, config_data):
        """
        Initializes the modules for the pipeline and determines whether to train a new set of models
        """
        self.model_name = config_data["model_name"]
        self.train_network_flag = self.model_name is None

        # Seeding the randomizer to ensure training consistency
        random.seed(1000)
        np.random.seed(1000)
        torch.manual_seed(1000)

        # Pipeline modules
        self.data_loader = DataLoader(config_data, self.train_network_flag)
        self.neural_net = NeuralNetwork(config_data)

    def run(self):
        """
        Runs the program pipeline training and executes the test and sample evaluations
        """
        
        _, _, test, sample = self.data_loader.get_data_sets(1)
        if self.train_network_flag:
            model_path = self.neural_net.train_network(self.data_loader)
        else:
            model_path = self.model_name
        self.neural_net.test_and_execute_network(test, sample, model_path)


class DataLoader():
    def __init__(self, config_data, train_network_flag):
        """
        Prepares the base datasets that get mixed up and batched each iteration of the training algorithm
        """

        # If using just a sample piece of data (i.e. a model is already trained and defined) nothing is converted for the sake of redundant speed loss)
        self.train_network_flag = train_network_flag
        self.global_vectors = GloVe()
        
        # Generates the GloVe numpy array datasets to be used with the Neural Network
        self.raw_input_data = self.csv_to_df(config_data["input_file_path"])
        print(f"Sample Data: {self.raw_input_data}")
        self.base_numpy_input = self.df_to_glove(self.raw_input_data)
        if self.train_network_flag:
            self.raw_training_data = self.csv_to_df(config_data["training_file_path"])
            self.train_valid_ratio = config_data["tv_ratio"]
            self.base_numpy_train = self.df_to_glove(self.raw_training_data, add_label=1)

        self.raw_test_data = self.csv_to_df(config_data["test_data_file_path"])
        self.raw_test_labels = self.csv_to_df(config_data["test_data_label_path"])
        self.base_numpy_test = self.df_to_glove(self.raw_test_data, add_label=1)
            
    def csv_to_df(self, csv_path):
        """
        Helper for abstracting dataframe from a csv
        """
        df = pd.read_csv(csv_path)
        return df
    
    def df_to_glove(self, data_frame, add_label=0):
        """
        Converts a dataframe containing the author, title and text into a GloVe embedded numpy array
        """
        glove_array = []
        for index, row in data_frame.iterrows():
            tokenizer = get_tokenizer("basic_english")
            
            # Try and except to clean the data and create a fresh block of tokenized vectors
            try:
                tokened_title = tokenizer(row['title'])
                tokened_author = tokenizer(row['author'])
                tokened_text = tokenizer(row['text'])

                # Getting the glove embeddings of the different text fields
                title_embeddings = self.global_vectors.get_vecs_by_tokens(tokened_title)
                author_embeddings = self.global_vectors.get_vecs_by_tokens(tokened_author)
                text_embeddings = self.global_vectors.get_vecs_by_tokens(tokened_text)
            except:
                continue
            
            # Getting the summation of the glove values and creating one input vector
            title_resulting_tensor = torch.sum(title_embeddings, dim=0)
            author_resulting_tensor = torch.sum(author_embeddings, dim=0)
            text_resulting_tensor = torch.sum(text_embeddings, dim=0)
            output_data_tensor = np.array(torch.stack((title_resulting_tensor, author_resulting_tensor, text_resulting_tensor)))
            
            # For testing and training data we extract the label either from the given dataset or the lookup table provided and connect 
            # it to the input label
            if add_label:
                if "label" not in data_frame.columns:
                    test_label = self.raw_test_labels.loc[self.raw_test_labels['id'] == row['id']]["label"]
                    test_label_vector = np.zeros(shape=300)
                    test_label_vector[0] = test_label
                    test_label_vector = np.expand_dims(test_label_vector, axis=0)    
                    output_data_tensor = np.concatenate((output_data_tensor, test_label_vector), axis=0)
                else:
                    train_label = row["label"]
                    train_label_vector = np.zeros(shape=300)
                    train_label_vector[0] = train_label
                    train_label_vector = np.expand_dims(train_label_vector, axis=0) 
                    output_data_tensor = np.concatenate((output_data_tensor, train_label_vector), axis=0)
                    
            glove_array.append(output_data_tensor)

        return glove_array


    def batchify(self, batch_size):
        """
        Converts the initial training dataset into a batched equivalent
        """
        if self.train_network_flag:
            random.shuffle(self.base_numpy_train)
            self.batched_train_set = []
            for i in range(0, int(len(self.base_numpy_train) / batch_size)):
                self.batched_train_set.append(np.array(self.base_numpy_train[i*batch_size:(i+1)*batch_size]))


    def get_data_sets(self, batch_size):
        """
        Prepares the training, validation, test and sample set for model consumption
        """
        self.batchify(batch_size)
        train_formatted = []
        valid_formatted = []
        test_formatted = []


        if self.train_network_flag:
            # Separates the training base set into a training block and validation block based on the tv ratio from the config file
            split_index = int(self.train_valid_ratio * len(self.batched_train_set)/(self.train_valid_ratio + 1))
            random.shuffle(self.batched_train_set)

            train_combined = self.batched_train_set[0:split_index]
            valid_combined = self.batched_train_set[split_index:len(self.batched_train_set)-1]
            
            # The following for-loop blocks create a formatted dataset that is easy to separate for the purpose of training
            for i in range(0, len(train_combined)):
                data_label_split = np.split(train_combined[i], indices_or_sections=[3], axis=1)
                data = data_label_split[0]
                label = np.sum(data_label_split[1], axis=2)
                train_formatted.append((data, label))

            for i in range(0, len(valid_combined)):
                data_label_split = np.split(valid_combined[i], indices_or_sections=[3], axis=1)
                data = data_label_split[0]
                label = np.sum(data_label_split[1], axis=2)
                valid_formatted.append((data, label))

        for i in range(0, len(self.base_numpy_test)):
            data_label_split = np.split(self.base_numpy_test[i], indices_or_sections=[3], axis=0)
            data = data_label_split[0]
            label = np.sum(data_label_split[1], axis=1)
            test_formatted.append((data, label))

        return train_formatted, valid_formatted, test_formatted, self.base_numpy_input

        

class NeuralNetwork():
    def __init__(self, config_data):
        """
        This is a wrapper for the Pytorch Model that allows for automated training based on config file hyperparameters
        """

        self.hyperparams = config_data["hyperparams"]
        self.save_rate = config_data["save_rate"]

    def get_model_name(self, hyper_set, epoch):
        """ 
        Generates a name for the model consisting of the hyperparameter values
        """
        path = "model_{0}_bs{1}_lr{2}_hn{3}_epoch{4}".format(hyper_set["name"], 
                                                            hyper_set["data"]["batch_size"],
                                                            hyper_set["data"]["learning_rate"],
                                                            hyper_set["data"]["hidden_nodes"],
                                                            epoch)
        return path

    def train_network(self, data_loader):
        """
        Generates the set of hyperparameter sweeps to perform and invokes the training of the model
        """
        run_set = []
        first_run = True
        for key in list(self.hyperparams.keys()):
            # Sweeps all of the keys of the hyperparameter list
            base_run_name = f"{key}_"
            key_data = self.hyperparams[key]
            for value in key_data:
                run = {}
                run_name = f"{base_run_name}{value}"

                # This provides a reference name for comparing results
                if first_run:
                    run_name = f"{run_name}_ref"
                    first_run = False
                run["name"] = run_name
                run["data"] = {}
                run["data"][key] = value

                # Adding the base values of the other hyperparameters not being swept
                for off_key in list(self.hyperparams.keys()):
                    if off_key != key:
                        run["data"][off_key] = list(self.hyperparams[off_key])[0]
                
                # Ensures no duplicate hyperparameter combinations
                run_exists = False
                for prev_run in run_set:
                    if run["data"] == prev_run["data"]:
                        run_exists = True
                if not run_exists: run_set.append(run)
            
        # Runs the model training for each hyper parameter set
        for run in run_set:
            model_path = self.train_hyperparameter_set(data_loader, run)
        
        return model_path

    def train_hyperparameter_set(self, data_loader, hyper_set):
        """
        Trains the neural network with the provided set of hyper parameters
        """
        
        # Neural Network setup
        print("Training Neural Network")
        self.network = BasicNetwork(hidden_nodes=hyper_set["data"]["hidden_nodes"])
        num_epochs = hyper_set["data"]["num_epochs"]
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.network.parameters(), lr=hyper_set["data"]["learning_rate"], momentum=0.9)

        # Set up some numpy arrays to store the training/test loss/erruracy and confusion table results
        train_err = np.zeros(num_epochs)
        train_loss = np.zeros(num_epochs)
        val_err = np.zeros(num_epochs)
        val_loss = np.zeros(num_epochs)
        val_fp = np.zeros(num_epochs)
        val_tp = np.zeros(num_epochs)
        val_fn = np.zeros(num_epochs)
        val_tn = np.zeros(num_epochs)

        # Train the network
        # Loop over the data iterator and sample a new batch of training data
        # Get the output from the network, and optimize the loss function.
        start_time = time.time()
        for epoch in range(num_epochs):

            train, valid, _, _ = data_loader.get_data_sets(hyper_set["data"]["batch_size"])

            total_train_loss = 0.0
            total_train_err = 0.0
            total_epoch = 0
            for i, data in enumerate(train, 0): 
                inputs, labels = data
                input_tensor = torch.from_numpy(inputs).type(torch.FloatTensor)
                labels = torch.from_numpy(labels).squeeze().float()
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass, backward pass, and optimize
                # outputs is now a neural-network object, performs the forward pass
                outputs = self.network(input_tensor)
                # Run BCEWithLogitsLoss using out neural-network object
                loss = criterion(outputs, labels)
                # Calculates all of the gradients for the different parameters
                loss.backward()
                # Runs the optimization using the previously calculated gradients
                optimizer.step()

                # Compares the batch_size outputs with the batch_size labels and
                # determines whether it DOESN'T MATCH the labels
                output_formatted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
                incorr = output_formatted != labels

                # Takes the sum of all the errors for this set
                total_train_err += int(incorr.sum())

                # Grabs the loss from this batch and adds it to the total loss for
                # the epoch
                total_train_loss += loss.item()

                #Adds batch size to the total_epoch (thus showing the TOTAL NUMBER
                #OF SAMPLES for this epoch)
                total_epoch += len(labels)
            
            # Updates the training error at this epoch to be the total number of
            # incorrect guesses / total samples
            train_err[epoch] = float(total_train_err) / total_epoch

            # Updates the training loss value at this epoch to be the total additive
            # loss over all batches, divided by the number of training batches in 
            # the epoch
            train_loss[epoch] = float(total_train_loss) / (i+1)
            
            #Repeat the neural network evaluation just with the validation dataset

            total_loss = 0.0
            total_err = 0.0
            total_fp = 0.0
            total_tp = 0.0
            total_fn = 0.0
            total_tn = 0.0
            total_epoch = 0

            for i, data in enumerate(valid, 0):
                inputs, labels = data
                input_tensor = torch.from_numpy(inputs).type(torch.FloatTensor)
                labels = torch.from_numpy(labels).squeeze().float()
                outputs = self.network(input_tensor)
                loss = criterion(outputs, labels)
                output_formatted = (torch.sigmoid(outputs) > 0.5).squeeze().long()

                # Produce the confusion matrix metrics for the validation dataset
                false_pos = (output_formatted == torch.ones(output_formatted.size())).logical_and(output_formatted != labels).sum()
                true_pos = (output_formatted == torch.ones(output_formatted.size())).logical_and(output_formatted == labels).sum()
                false_neg = (output_formatted == torch.zeros(output_formatted.size())).logical_and(output_formatted != labels).sum()
                true_neg = (output_formatted == torch.zeros(output_formatted.size())).logical_and(output_formatted == labels).sum()
                
                total_fp += false_pos
                total_fn += false_neg
                total_tp += true_pos
                total_tn += true_neg
                total_incorr = output_formatted != labels
                total_err += int(total_incorr.sum())
                total_loss += loss.item()
                total_epoch += len(labels)
            
            # Averaging out the validation results over the current epoch
            err = float(total_err) / total_epoch
            loss = float(total_loss) / (i + 1)
            val_err[epoch], val_loss[epoch]=  err, loss
            val_fp[epoch], val_fn[epoch], val_tp[epoch], val_tn[epoch] = (float(total_fp) / total_epoch), (float(total_fn) / total_epoch), (float(total_tp) / total_epoch), (float(total_tn) / total_epoch)
            
            # Print statements to update the user on the progress of the training]

            print(("Epoch {}: Train err: {}, Train loss: {} |"+
                "Validation err: {}, Validation loss: {}").format(
                    epoch + 1,
                    train_err[epoch],
                    train_loss[epoch],
                    val_err[epoch],
                    val_loss[epoch],
                ))

            print("Epoch {}: True Positives: {} | False Positives: {} | True Negatives: {} | False Negatives: {}".format(epoch + 1, val_tp[epoch],
                    val_fp[epoch],
                    val_tn[epoch],
                    val_fn[epoch]))
            
            # Save the current model (checkpoint) to a file
            if epoch % self.save_rate == 0 or epoch == num_epochs - 1:
                model_path = ('output\\{}'.format(self.get_model_name(hyper_set, epoch)))
                torch.save(self.network.state_dict(), f"{model_path}.pt")
                
                #Training/Validation Error and Loss
                np.savetxt("{}_train_err.csv".format(model_path), train_err)
                np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
                np.savetxt("{}_val_err.csv".format(model_path), val_err)
                np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

                #Confusion Matrix
                np.savetxt("{}_val_false_pos.csv".format(model_path), val_fp)
                np.savetxt("{}_val_true_pos.csv".format(model_path), val_tp)
                np.savetxt("{}_val_false_neg.csv".format(model_path), val_fn)
                np.savetxt("{}_val_true_neg.csv".format(model_path), val_tn)

                # Plotting the results on the final epoch of the run
                if epoch == num_epochs - 1:
                    self.plot_training_curve(model_path)
                    self.plot_confusion_matrix(model_path)

        print(f'Finished Training Run: {hyper_set["name"]}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
        del(self.network)
        return model_path
    
    def plot_training_curve(self, path):
        """ 
        Plots the training curve for a model run, given the csv files
        containing the train/validation error/loss.
        """
        
        train_err = np.loadtxt("{}_train_err.csv".format(path))
        val_err = np.loadtxt("{}_val_err.csv".format(path))
        train_loss = np.loadtxt("{}_train_loss.csv".format(path))
        val_loss = np.loadtxt("{}_val_loss.csv".format(path))

        plt.figure()
        plt.title("Train vs Validation Error")
        n = len(train_err) # number of epochs
        plt.plot(range(1,n+1), train_err, label="Train")
        plt.plot(range(1,n+1), val_err, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend(loc='best')
        plt.savefig(f'{path}_error_chart.png')

        plt.figure()
        plt.title("Train vs Validation Loss")
        plt.plot(range(1,n+1), train_loss, label="Train")
        plt.plot(range(1,n+1), val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.savefig(f'{path}_loss_chart.png')


    def plot_confusion_matrix(self, path):
        """
        Plots the validation confusion matrix curve for a model run, given the csv files
        containing the appropriate metrics.
        """
        val_tp = np.loadtxt("{}_val_true_pos.csv".format(path))
        val_fp = np.loadtxt("{}_val_false_pos.csv".format(path))
        val_tn = np.loadtxt("{}_val_true_neg.csv".format(path))
        val_fn = np.loadtxt("{}_val_false_neg.csv".format(path))

        plt.figure()
        plt.title("Validation Confusion Matrix Values")
        n = len(val_tp)
        plt.plot(range(1,n+1), val_tp, label="True Positives")
        plt.plot(range(1,n+1), val_fp, label="False Positives")
        plt.plot(range(1,n+1), val_tn, label="True Negatives")
        plt.plot(range(1,n+1), val_fn, label="False Negatives")
        plt.xlabel("Epoch")
        plt.ylabel("Percentage")
        plt.legend(loc='best')
        plt.savefig(f'{path}_confusion_matrix.png')
    
    def test_and_execute_network(self, test, sample, model_path):
        """
        Runs the provided model through the test data and whatever sample data is provided
        """

        # Determining hidden nodes for the model instance
        hidden_nodes = int(model_path.split("hn")[1].split("_")[0])
        model = BasicNetwork(hidden_nodes=hidden_nodes)
        total_acc = 0.0
        model.load_state_dict(torch.load(f"{model_path}.pt"))

        # Calculate the test accuracy using the same method as used in training
        for i, data in enumerate(test, 0):
            inputs, labels = data
            input_tensor = torch.from_numpy(inputs).type(torch.FloatTensor)
            labels = torch.from_numpy(labels).squeeze().float()
            outputs = model(input_tensor)
            output_formatted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            total_corr = output_formatted == labels
            total_acc += int(total_corr.sum())
        accuracy = float(total_acc) / (i+1)
        print(f"Testing Accuracy for Model {model_path}: {accuracy}")

        # Calculate predictions on the sample values
        for i, data in enumerate(sample, 0):
            inputs = data
            input_tensor = torch.from_numpy(inputs).type(torch.FloatTensor)
            outputs = model(input_tensor)
            output_formatted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            print(f"Model-Predicted Output for Given Sample: {output_formatted}")
        
        # Clear space from the now used model
        del model


class BasicNetwork(nn.Module):
    def __init__(self, hidden_nodes=32):
        """
        Simple artificial neural network for converting GloVe embeddings into a binary value
        """
        super(BasicNetwork, self).__init__()
        self.name = "basic_artificial"
        self.fc1 = nn.Linear(3 * 300, hidden_nodes) # Flattening conv. layer to 32 nodes
        self.fc2 = nn.Linear(hidden_nodes, 1) # Getting a single output node

    def forward(self, x):
        """
        Passes the input x through the defined neural network, formatting along the way and using a ReLu activation function
        """
        x = x.view(-1, 3 * 300) #Flattens the image to a 1D single vector
        x = F.relu(self.fc1(x)) # Puts it through fc1 and relu's the result
        x = self.fc2(x) # Runs final output layer
        x = x.squeeze() # Flatten to [batch_size]
        
        return x

if __name__ == "__main__":
    print("Data Pipeline Python Script")