import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
import os

# ==========================================
# Google Colab Helmet Classifier Training
# ==========================================
# Instructions:
# 1. Upload this script to Google Colab.
# 2. Upload your dataset in the format:
#    dataset/
#      train/
#        0_no_helmet/
#        1_helmet/
#      val/
#        0_no_helmet/
#        1_helmet/
# 3. Run: !python train_helmet.py
# 4. Download 'helmet_model.pt' and place it in the violation-vision-mvp/backend directory.
# ==========================================

def main():
    # Configuration
    data_dir = 'dataset'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found. Please create it and organize your images.")
        # Create dummy directories to show the structure if they don't exist
        os.makedirs(os.path.join(data_dir, 'train', '0_no_helmet'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'train', '1_helmet'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', '0_no_helmet'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', '1_helmet'), exist_ok=True)
        print("Created dummy directory structure. Please populate with images and re-run.")
        return

    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                     shuffle=True, num_workers=2)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print(f"Classes found: {class_names}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load pre-trained MobileNetV3 Small
    model = mobilenet_v3_small(weights='DEFAULT')
    
    # Modify the final classification layer for 2 classes
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'helmet_model.pt')
                
        print()

    print(f'Training complete. Best val Acc: {best_acc:.4f}')
    print("Model saved as 'helmet_model.pt'. Please download this file and place it in the backend folder.")

if __name__ == '__main__':
    main()
