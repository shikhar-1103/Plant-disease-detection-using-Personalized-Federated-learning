import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
import copy
import time
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def main():
    start_time = time.time()
    LOG_FILE = "pfl_dataset1_main_detailed_log.txt"
    CSV_FILE = "pfl_dataset1_main_detailed_results.csv"
    
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"--- Experiment Started: {datetime.datetime.now()} ---\n")

    # ---- Hyperparameters ----
    EPOCHS_PER_PHASE = 5
    NUM_PHASES = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    L2_WEIGHT_DECAY = 1e-4 
    CLIENT_NAMES = ["Client 1", "Client 2"]

    # ---- Data Load (WITH Data Augmentation) ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    
    # !!! IMPORTANT: Apna COMBINED dataset ka folder path yahan daalein !!!
    data_dir = r"C:\Users\shikh\Desktop\Thesis\dataset1"
    
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except FileNotFoundError:
        print(f"Error: Dataset not found at path '{data_dir}'. Please update the data_dir variable.")
        return

    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Classes found: {num_classes}")
    print(f"Full dataset size: {len(dataset)}")

    # ---- Data Splitting Logic ----
    client1_prevalent_classes = ["Apple___Apple_scab", "Tomato___Bacterial_spot"]
    client2_prevalent_classes = ["Apple___Black_rot", "Tomato___Early_blight", "Tomato___Late_blight"]
    common_classes = [ "Apple___healthy", "Maize___Northern_Leaf_Blight", "Maize___healthy", "Tomato___healthy" ]
    class_to_idx = dataset.class_to_idx
    client1_prevalent_indices = {class_to_idx[name] for name in client1_prevalent_classes if name in class_to_idx}
    client2_prevalent_indices = {class_to_idx[name] for name in client2_prevalent_classes if name in class_to_idx}
    common_indices = {class_to_idx[name] for name in common_classes if name in class_to_idx}
    samples_by_class = defaultdict(list)
    for i in range(len(dataset)): _, label = dataset.samples[i]; samples_by_class[label].append(i)
    client1_indices, client2_indices = [], []
    random.seed(42)
    for cls_idx, item_indices in samples_by_class.items():
        item_indices = item_indices.copy(); random.shuffle(item_indices)
        if cls_idx in common_indices: split_point = int(0.5 * len(item_indices))
        elif cls_idx in client1_prevalent_indices: split_point = int(0.9 * len(item_indices))
        elif cls_idx in client2_prevalent_indices: split_point = int(0.1 * len(item_indices))
        else: split_point = int(0.5 * len(item_indices))
        client1_indices.extend(item_indices[:split_point]); client2_indices.extend(item_indices[split_point:])
    
    parent_frac, validation_frac = 0.35, 0.10
    total_len = len(dataset)
    all_indices = list(range(total_len)); random.shuffle(all_indices)
    parent_size, validation_size = int(parent_frac * total_len), int(validation_frac * total_len)
    parent_indices = all_indices[:parent_size]; val_indices = all_indices[parent_size : parent_size + validation_size]
    client_pool_indices = set(all_indices[parent_size + validation_size:])
    c1_client_indices = [idx for idx in client1_indices if idx in client_pool_indices]
    c2_client_indices = [idx for idx in client2_indices if idx in client_pool_indices]
    c1_dataset = Subset(dataset, c1_client_indices); c2_dataset = Subset(dataset, c2_client_indices)
    parent_dataset = Subset(dataset, parent_indices); val_dataset = Subset(dataset, val_indices)

    def phase_split(client_dataset, n_phases=10):
        length = len(client_dataset)
        if length == 0: return [Subset(client_dataset, []) for _ in range(n_phases)]
        phase_sizes = [length // n_phases] * (n_phases - 1); phase_sizes.append(length - sum(phase_sizes))
        indices = list(range(length)); random.shuffle(indices)
        subsets, start_idx = [], 0
        for size in phase_sizes:
            subsets.append(Subset(client_dataset, indices[start_idx : start_idx + size])); start_idx += size
        return subsets

    c1_phases = phase_split(c1_dataset, n_phases=NUM_PHASES); c2_phases = phase_split(c2_dataset, n_phases=NUM_PHASES)
    print("Split complete:"); print(f"Parent dataset size: {len(parent_dataset)}"); print(f"Validation set size: {len(val_dataset)}")
    print(f"{CLIENT_NAMES[0]} dataset size: {len(c1_dataset)}"); print(f"{CLIENT_NAMES[1]} dataset size: {len(c2_dataset)}")

    class HierarchicalCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1a, self.conv1b = nn.Conv2d(3, 32, 5, padding=2), nn.Conv2d(32, 64, 3, padding=1)
            self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
            self.conv2a, self.conv2b = nn.Conv2d(3, 32, 7, padding=3), nn.Conv2d(32, 64, 3, padding=1)
            self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
            self.fc1, self.fc2 = nn.Linear(64, 64), nn.Linear(64, num_classes)
            self.activation = nn.ReLU(); self.dropout = nn.Dropout(0.5)
        def forward(self, x):
            x1 = self.activation(self.conv1b(self.activation(self.conv1a(x)))); x1 = self.pool1(x1).view(x1.size(0), -1)
            x2 = self.activation(self.conv2b(self.activation(self.conv2a(x)))); x2 = self.pool2(x2).view(x2.size(0), -1)
            x = x1 + x2; x = self.activation(self.fc1(x)); x = self.dropout(x)
            return self.fc2(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    def make_loader(ds, shuffle=True): return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2, pin_memory=True)
    parent_loader = make_loader(parent_dataset); val_loader = make_loader(val_dataset, shuffle=False)
    def compute_metrics(model, loader):
        model.eval()
        if len(loader.dataset) == 0: return 0.0, 0.0, 0.0, 0.0
        correct, total = 0, 0; all_labels, all_preds = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device); preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item(); total += y.size(0)
                all_labels.extend(y.cpu().numpy()); all_preds.extend(preds.cpu().numpy())
        accuracy = correct / total if total > 0 else 0.0
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        return accuracy, p, r, f1

    parent_model = HierarchicalCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(parent_model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    print("\n--- Initial Parent Training ---")
    parent_train_start = time.time()
    for epoch in range(EPOCHS_PER_PHASE):
        parent_model.train()
        for xb, yb in parent_loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = criterion(parent_model(xb), yb); loss.backward(); optimizer.step()
    parent_train_end = time.time()
    initial_parent_training_time = parent_train_end - parent_train_start
    print(f"Initial parent training finished in {initial_parent_training_time:.2f} seconds.")
    
    parent_weights = copy.deepcopy(parent_model.state_dict())
    client_models = [HierarchicalCNN(num_classes).to(device) for _ in range(2)]; phase_results = []
    
    for phase in range(NUM_PHASES):
        print(f"\n--- Phase {phase+1}/{NUM_PHASES} ---")
        for model in client_models: model.load_state_dict(parent_weights)

        # NEW: Client data 80/20 split for Train/Test for each phase
        client_phase_datasets = [c1_phases[phase], c2_phases[phase]]
        client_train_loaders, client_test_loaders = [], []
        for client_ds in client_phase_datasets:
            if len(client_ds) == 0:
                client_train_loaders.append(make_loader(client_ds))
                client_test_loaders.append(make_loader(client_ds))
                continue
            
            indices = list(range(len(client_ds)))
            random.shuffle(indices)
            split_idx = int(0.8 * len(indices))
            train_indices, test_indices = indices[:split_idx], indices[split_idx:]
            
            train_subset = Subset(client_ds, train_indices)
            test_subset = Subset(client_ds, test_indices)
            
            client_train_loaders.append(make_loader(train_subset))
            client_test_loaders.append(make_loader(test_subset, shuffle=False))

        local_training_start = time.time()
        for i, model in enumerate(client_models):
            if len(client_train_loaders[i].dataset) == 0:
                print(f"Skipping training for client: {CLIENT_NAMES[i]} (no data)."); continue
            print(f"Training client: {CLIENT_NAMES[i]}...")
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT_DECAY)
            for epoch in range(EPOCHS_PER_PHASE):
                model.train()
                for xb, yb in client_train_loaders[i]:
                    xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
                    loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
        local_training_end = time.time()
        local_client_training_time = local_training_end - local_training_start
        print(f"Local client training finished in {local_client_training_time:.2f} seconds.")
        
        # This code uses the original paper's aggregation, NOT Parameter De-coupling
        agg_state, old_parent_state = {}, parent_model.state_dict()
        state_c1, state_c2 = client_models[0].state_dict(), client_models[1].state_dict()
        for k in old_parent_state:
            summed_weights = old_parent_state[k] + state_c1[k] + state_c2[k]
            agg_state[k] = summed_weights / (len(client_models) + 1)
        parent_weights = parent_model.state_dict(); parent_weights.update(agg_state); parent_model.load_state_dict(parent_weights)
        for model in client_models:
            local_state = model.state_dict()
            for k in old_parent_state: local_state[k] = (local_state[k] + parent_weights[k]) / 2
            model.load_state_dict(local_state)

        # --- Evaluation ---
        pa_train, pp_train, pr_train, pf1_train = compute_metrics(parent_model, parent_loader)
        pa_test, pp_test, pr_test, pf1_test = compute_metrics(parent_model, val_loader)
        
        c1a_train, c1p_train, c1r_train, c1f1_train = compute_metrics(client_models[0], client_train_loaders[0])
        c1a_test, c1p_test, c1r_test, c1f1_test = compute_metrics(client_models[0], client_test_loaders[0])
        
        c2a_train, c2p_train, c2r_train, c2f1_train = compute_metrics(client_models[1], client_train_loaders[1])
        c2a_test, c2p_test, c2r_test, c2f1_test = compute_metrics(client_models[1], client_test_loaders[1])
        
        print(f"Parent | Train Acc: {pa_train:.3f} | Test Acc: {pa_test:.3f}")
        print(f"{CLIENT_NAMES[0]} | Train Acc: {c1a_train:.3f} | Test Acc: {c1a_test:.3f}")
        print(f"{CLIENT_NAMES[1]} | Train Acc: {c2a_train:.3f} | Test Acc: {c2a_test:.3f}")
        
        results_this_phase = { 
            "phase": phase+1, 
            "parent_acc_train": pa_train, "parent_precision_train": pp_train, "parent_recall_train": pr_train, "parent_f1_train": pf1_train,
            "parent_acc_test": pa_test, "parent_precision_test": pp_test, "parent_recall_test": pr_test, "parent_f1_test": pf1_test,
            "client1_acc_train": c1a_train, "client1_precision_train": c1p_train, "client1_recall_train": c1r_train, "client1_f1_train": c1f1_train, 
            "client1_acc_test": c1a_test, "client1_precision_test": c1p_test, "client1_recall_test": c1r_test, "client1_f1_test": c1f1_test,
            "client2_acc_train": c2a_train, "client2_precision_train": c2p_train, "client2_recall_train": c2r_train, "client2_f1_train": c2f1_train,
            "client2_acc_test": c2a_test, "client2_precision_test": c2p_test, "client2_recall_test": c2r_test, "client2_f1_test": c2f1_test,
            "local_client_training_time": local_client_training_time 
        }
        if phase == 0:
            results_this_phase['initial_parent_training_time'] = initial_parent_training_time
        phase_results.append(results_this_phase)

        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"\n--- Phase {phase+1}/{NUM_PHASES} Results ---\n")
            if phase == 0: log_file.write(f"Initial Parent Training Time: {initial_parent_training_time:.2f} seconds\n")
            log_file.write(f"Local Client Training Time (This Phase): {local_client_training_time:.2f} seconds\n\n")
            log_file.write("Parent Model (on Training Set):\n")
            log_file.write(f"  Accuracy: {pa_train:.4f}, Precision: {pp_train:.4f}, Recall: {pr_train:.4f}, F1-Score: {pf1_train:.4f}\n")
            log_file.write("Parent Model (on Test Set):\n")
            log_file.write(f"  Accuracy: {pa_test:.4f}, Precision: {pp_test:.4f}, Recall: {pr_test:.4f}, F1-Score: {pf1_test:.4f}\n\n")
            log_file.write(f"{CLIENT_NAMES[0]} (on Training Set):\n")
            log_file.write(f"  Accuracy: {c1a_train:.4f}, Precision: {c1p_train:.4f}, Recall: {c1r_train:.4f}, F1-Score: {c1f1_train:.4f}\n")
            log_file.write(f"{CLIENT_NAMES[0]} (on Test Set):\n")
            log_file.write(f"  Accuracy: {c1a_test:.4f}, Precision: {c1p_test:.4f}, Recall: {c1r_test:.4f}, F1-Score: {c1f1_test:.4f}\n\n")
            log_file.write(f"{CLIENT_NAMES[1]} (on Training Set):\n")
            log_file.write(f"  Accuracy: {c2a_train:.4f}, Precision: {c2p_train:.4f}, Recall: {c2r_train:.4f}, F1-Score: {c2f1_train:.4f}\n")
            log_file.write(f"{CLIENT_NAMES[1]} (on Test Set):\n")
            log_file.write(f"  Accuracy: {c2a_test:.4f}, Precision: {c2p_test:.4f}, Recall: {c2r_test:.4f}, F1-Score: {c2f1_test:.4f}\n")
            log_file.write("---------------------------------\n")

    end_time = time.time()
    results_df = pd.DataFrame(phase_results)
    results_df['total_execution_time'] = end_time - start_time
    results_df.to_csv(CSV_FILE, index=False, na_rep='N/A')
    print(f"\nResults successfully saved to {CSV_FILE}")
    
    phases = [pr["phase"] for pr in phase_results]
    plt.figure(figsize=(12, 7)); plt.plot(phases, [pr['parent_acc_test'] for pr in phase_results], 'o-', label="Parent (Test)")
    plt.plot(phases, [pr.get('client1_acc_test', 0) for pr in phase_results], 's-', label=f"{CLIENT_NAMES[0]} (Test)")
    plt.plot(phases, [pr.get('client2_acc_test', 0) for pr in phase_results], '^-', label=f"{CLIENT_NAMES[1]} (Test)")
    plt.xlabel("Phase"); plt.ylabel("Test Accuracy"); plt.title("Test Accuracy over Phases"); plt.legend(); plt.grid(True); plt.xticks(phases); plt.ylim(0, 1.05)
    plt.savefig("final_test_accuracy.svg")
    
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"\n--- Experiment Concluded: {datetime.datetime.now()} ---\n")
        log_file.write(f"Total Run Time: {end_time - start_time:.2f} seconds\n\n")
    print(f"\nExperiment concluded. Runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()