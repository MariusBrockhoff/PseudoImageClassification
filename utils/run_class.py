import time
from utils.model_initializer import *
from utils.wandb_initializer import *
from utils.finetune_models import *
from utils.load_data import *


from config_files.config_finetune import *
from config_files.config_pretraining import *


class Run:
    """
    Class for running the full Spike sorting pipeline on raw MEA recordings (MCS file format) (if benchmark=False)
    or benchmarking the method on simulated spike files (if benchmark=True)

    This class integrates different components of the spike sorting pipeline, including
    data preparation, model initialization, pretraining, finetuning, prediction, clustering,
    and evaluation.

    Attributes:
        model_config: Configuration for the model.
        data_path: Path to the data.
        pretrain_method: Method for pretraining the model.
        fine_tune_method: Method for finetuning the model.
        pretraining_config: Configuration for pretraining.
        finetune_config: Configuration for finetuning.

    Methods:
        prepare_data(): Prepare the data for training and testing.
        initialize_model(): Initialize the machine learning model.
        initialize_wandb(): Initialize Weights & Biases for tracking experiments.
        pretrain(): Pretrain the model.
        finetune(): Finetune the model.
        predict(): Predict latent representations using the model.
        cluster_data(): Perform clustering on the encoded data.
        evaluate_spike_sorting(): Evaluate the results of spike sorting.
        execute_pretrain(): Execute the pretraining phase.
        execute_finetune(): Execute the finetuning phase.
    """

    def __init__(self, model_config, data, benchmark, pretrain_method, fine_tune_method):
        self.model_config = model_config
        self.data_name = data
        self.benchmark = benchmark
        self.pretrain_method = pretrain_method
        self.fine_tune_method = fine_tune_method
        self.pretraining_config = Config_Pretraining(self.data_name, self.model_config.MODEL_TYPE)
        self.finetune_config = Config_Finetuning(self.data_name, self.model_config.MODEL_TYPE)


    def data_loader(self):
        dataset, dataset_test = load_data(data_name=self.data_name, pretraining_config=self.pretraining_config,
                                          finetune_config=self.finetune_config)
        return dataset, dataset_test
    def initialize_model(self):
        model = model_initializer(self.model_config)
        return model

    def initialize_wandb(self, method):
        wandb_initializer(self.model_config, self.pretraining_config, self.finetune_config, method)

    def pretrain(self, model, dataset, dataset_test):
        print('---' * 30)
        print('PRETRAINING MODEL...')

        pretrain_model(model=model,
                       pretraining_config=self.pretraining_config,
                       dataset=dataset,
                       dataset_test=dataset_test,
                       save_weights=self.pretraining_config.SAVE_WEIGHTS,
                       save_dir=self.pretraining_config.SAVE_DIR)

    def finetune(self, model, dataset, dataset_test):
        print('---' * 30)
        print('FINETUNING MODEL...')
        y_finetuned, y_true = finetune_model(model=model, finetune_config=self.finetune_config,
                                     dataset=dataset, dataset_test=dataset_test)

        return y_finetuned, y_true




    def evaluate_spike_sorting(self, y_pred, y_true, y_pred_test=None, y_true_test=None):
        print('---' * 30)
        print('EVALUATE RESULTS...')
        train_acc, test_acc = evaluate_clustering(y_pred, y_true, y_pred_test, y_true_test)
        return train_acc, test_acc

    def execute_pretrain(self):
        start_time = time.time()
        self.initialize_wandb(self.pretrain_method)
        dataset, dataset_test = self.data_loader()
        model = self.initialize_model()
        self.pretrain(model=model, dataset=dataset, dataset_test=dataset_test)
        end_time = time.time()
        print("Time Run Execution: ", end_time - start_time)


    def execute_finetune(self):
        start_time = time.time()
        self.initialize_wandb(self.fine_tune_method)
        dataset, dataset_test = self.data_loader()
        model = self.initialize_model()
        y_pred_finetuned, y_true = self.finetune(model=model, dataset=dataset, dataset_test=dataset_test)
        end_time = time.time()
        print("Time Finetuning Execution: ", end_time - start_time)


