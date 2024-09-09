import os
import glob
import torch
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from config import max_checkpoint_num, proposalN, eval_trainset, set
from utils.eval_model import eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_metrics(metrics, save_path, epoch, phase):
    """Plots the training/testing metrics."""
    for metric_name, values in metrics.items():
        plt.figure()
        plt.plot(range(1, epoch + 1), values)
        plt.title(f'{phase} {metric_name} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.savefig(os.path.join(save_path, f'{phase}_{metric_name}_epoch_{epoch}.png'))
        plt.close()

def save_accuracies(epoch, train_accuracy, test_accuracy, save_path):
    """Saves train and test accuracies to a CSV file."""
    accuracy_file = os.path.join(save_path, 'accuracies.csv')
    file_exists = os.path.isfile(accuracy_file)

    with open(accuracy_file, mode='a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Accuracy', 'Test Accuracy'])  # Header row
        writer.writerow([epoch, train_accuracy, test_accuracy])

def train(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):

    train_metrics = {
        'learning_rate': [],
        'raw_accuracy': [],
        'local_accuracy': [],
        'raw_loss_avg': [],
        'local_loss_avg': [],
        'windowscls_loss_avg': [],
        'total_loss_avg': []
    }

    test_metrics = {
        'raw_accuracy': [],
        'local_accuracy': [],
        'raw_loss_avg': [],
        'local_loss_avg': [],
        'windowscls_loss_avg': [],
        'total_loss_avg': []
    }
    raw_correct = 0
    total_samples = 0

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        train_metrics['learning_rate'].append(lr)

        for i, data in enumerate(trainloader):
            if set == 'CUB':
                images, labels, _, _ = data
            else:
                images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = raw_loss + local_loss + windowscls_loss

            total_loss.backward()

            optimizer.step()
        # Calculate number of correct predictions
        #_, predicted = torch.max(raw_logits, 1)
        #correct_predictions += (predicted == labels).sum().item()
        pred = raw_logits.max(1, keepdim=True)[1]
        raw_correct += pred.eq(labels.view_as(pred)).sum().item()
        total_samples += labels.size(0)

        # Calculate and store training accuracy
        raw_accuracy =  raw_correct / total_samples
        train_metrics['raw_accuracy'].append(raw_accuracy)
        
        scheduler.step()

        # Evaluation every epoch
        if eval_trainset:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(f'Train set: raw accuracy: {100. * raw_accuracy:.2f}%, local accuracy: {100. * local_accuracy:.2f}%')

            # Store metrics
            train_metrics['raw_accuracy'].append(raw_accuracy)
            train_metrics['local_accuracy'].append(local_accuracy)
            train_metrics['raw_loss_avg'].append(raw_loss_avg)
            train_metrics['local_loss_avg'].append(local_loss_avg)
            train_metrics['windowscls_loss_avg'].append(windowscls_loss_avg)
            train_metrics['total_loss_avg'].append(total_loss_avg)

            # Plot training metrics
            plot_metrics(train_metrics, save_path, epoch, 'Train')

        if epoch % 10 == 0:
            # Eval test set
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
            local_loss_avg = eval(model, testloader, criterion, 'test', save_path, epoch)
    
            print(f'Test set: raw accuracy: {100. * raw_accuracy:.2f}%, local accuracy: {100. * local_accuracy:.2f}%')
    
            # Store metrics
            test_metrics['raw_accuracy'].append(raw_accuracy)
            test_metrics['local_accuracy'].append(local_accuracy)
            test_metrics['raw_loss_avg'].append(raw_loss_avg)
            test_metrics['local_loss_avg'].append(local_loss_avg)
            test_metrics['windowscls_loss_avg'].append(windowscls_loss_avg)
            test_metrics['total_loss_avg'].append(total_loss_avg)

        # Plot test metrics
        plot_metrics(test_metrics, save_path, epoch, 'Test')

        # Save train and test accuracies
        save_accuracies(epoch, raw_accuracy, test_metrics['raw_accuracy'][-1], save_path)

        # Save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
                'train_accuracy': train_metrics['raw_accuracy'][-1],
                'test_accuracy': test_metrics['raw_accuracy'][-1],
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

        # Limit the number of checkpoints to max_checkpoint_num
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))
