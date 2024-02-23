# from const import *
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
import wandb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from running_code.processing.YH.model import AlexNet,ResNet,BasicBlock, AlexNet, Alex_Autoencoder, SegNetAlexnet, Classifier, Conv1dClassifier
from running_code.processing.YH.utils_eeg import *
from running_code.processing.YH.utils import *
from running_code.preprocessing.wavelet_transform.load_conti import load_conti_LOO

with open('/home/infres/annguyen/running_code/processing/YH/convo.yaml', 'r') as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)


datasets = [
    {"name": "4", "window_time": 4, "window_shift_rate": 1},
    {"name": "4", "window_time": 4, "window_shift_rate": 0.5},
    {"name": "4", "window_time": 4, "window_shift_rate": 0.25},
    
    {"name": "8", "window_time": 8, "window_shift_rate": 1},
    {"name": "8", "window_time": 8, "window_shift_rate": 0.5},
    {"name": "8", "window_time": 8, "window_shift_rate": 0.25},

    {"name": "12", "window_time": 12, "window_shift_rate": 1},
    {"name": "12", "window_time": 12, "window_shift_rate": 0.5},
    {"name": "12", "window_time": 12, "window_shift_rate": 0.25},
    
    {"name": "24", "window_time": 24, "window_shift_rate": 1},
    {"name": "24", "window_time": 24, "window_shift_rate": 0.5},
    {"name": "24", "window_time": 24, "window_shift_rate": 0.25}
]



for dataset in datasets:
    dataset_name = dataset['name']
    window_time = dataset['window_time']
    window_shift_rate = dataset['window_shift_rate']
    window_shift = int( window_shift_rate * window_time * SAMPLING_FQ )

    wandb.init(project="conti", entity="dung0907", name=f"LOO_{dataset_name}_{window_shift}")

    sweep_id = wandb.sweep(sweep=sweep_config, project="conti")

    def train_and_test():
        config = wandb.config
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        epochs = config['epochs']
        dropout = config['dropout']
        subject_input = config['subject_in']
        optimizer_name = config['optimizer_name']

        x_train, y_train, x_test, y_test = load_conti_LOO(subject_except=subject_input, window_time=window_time, window_shift=window_shift)
        set_seed(43)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        subject = init_conti_LOO(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, batch_size=batch_size, dimension=1, subject_input=subject_input)
        subject.shuffle_trial()
        subject.build_dataloaders()

        input_size = subject.train_size

        # model = NeuralNetworkNasim(input_size=input_size).to(device)
        # model = Classifier(keep_prob=0.2, input_size=input_size).to(device)
        # model = AlexNet(num_classes=3,dropout=0.2).to(device)
        # model = Conv1dClassifier(7,3).to(device)
        # model = RandomForestClassifier(n_estimators=100, n_jobs=30)
        # model = SVC(kernel='rbf', C=1.0, random_state=43)
        model = EEGTransformer(n_channels=7, n_bins=7, output_size=3).to(device)
        # model = Alexnet_feature().to(device)
        # model = Conv1dClassifier(n_input=input_size, n_output=3, drop_rate=dropout).to(device)

        criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=13000,
                                                    gamma=0.1, 
                                                    last_epoch=- 1, 
                                                    verbose=True)
        model_met = init_metrics(["acc", "loss"])

        for epoch in range(epochs):
            print(f"Epoc, {epoch}")
            model_met = train(model, optimizer, criterion,
                                subject, 
                                device=device,
                                metrics_list=["bacc","acc", "loss","acc_class_0","acc_class_1","acc_class_2"],
                                num_classes_task=3)
            
            train_acc = print_ver(model_met["acc"])
            train_loss = print_ver(model_met["loss"])
            train_balenced_acc = print_ver(model_met["bacc"])
            focused_acc = print_ver(model_met["acc_class_0"])
            unfocused_acc = print_ver(model_met["acc_class_1"])
            drowsed_acc = print_ver(model_met["acc_class_2"])
            
            with torch.no_grad():
                scheduler.step()
                model_met = test(model=model,
                                subject=subject,
                                criterion=criterion,
                                metrics_list=["bacc","acc", "loss","acc_class_0","acc_class_1","acc_class_2"]
                                )
            
            test_acc = print_ver(model_met["acc"])
            test_loss = print_ver(model_met["loss"])
            test_balenced_acc = print_ver(model_met["bacc"])
            test_focused_acc = print_ver(model_met["acc_class_0"])
            test_unfocused_acc = print_ver(model_met["acc_class_1"])
            test_drowsed_acc = print_ver(model_met["acc_class_2"])
                
            wandb.log({f"LOO_{dataset_name}_{window_shift}_train_acc": train_acc, 
                        f"LOO_{dataset_name}_{window_shift}_train_loss": train_loss,
                        f"LOO_{dataset_name}_{window_shift}_train_balenced_acc": train_balenced_acc,
                        f"LOO_{dataset_name}_{window_shift}_focused_acc": focused_acc,
                        f"LOO_{dataset_name}_{window_shift}_unfocused_acc": unfocused_acc,
                        f"LOO_{dataset_name}_{window_shift}_drowsed_acc":drowsed_acc,
                        f"LOO_{dataset_name}_{window_shift}_test_acc": test_acc, 
                        f"LOO_{dataset_name}_{window_shift}_test_loss": test_loss,
                        f"LOO_{dataset_name}_{window_shift}_test_focused_acc": test_balenced_acc,
                        f"LOO_{dataset_name}_{window_shift}_test_unfocused_acc": test_focused_acc,
                        f"LOO_{dataset_name}_{window_shift}_test_drowsed_acc": test_unfocused_acc,
                        f"LOO_{dataset_name}_{window_shift}_test_balenced_acc": test_drowsed_acc
                        })
    
    wandb.agent(sweep_id, function=train_and_test)
