from torchvision.models import alexnet
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model import Net
from data_loader import get_loader
from utils import accuracy, Tracker, Fscore
from coral import coral


def train(model, optimizer, source_loader, target_loader, tracker, args, epoch=0):

    model.train()
    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

    # Trackers to monitor classification and CORAL loss
    classification_loss_tracker = tracker.track('classification_loss', tracker_class(**tracker_params))
    coral_loss_tracker = tracker.track('CORAL_loss', tracker_class(**tracker_params))

    min_n_batches = min(len(source_loader), len(target_loader))

    tq = tqdm(range(min_n_batches), desc='{} E{:03d}'.format('Training + Adaptation', epoch), ncols=0)

    for _ in tq:

        source_data, source_label = next(iter(source_loader))
        target_data, _ = next(iter(target_loader))  # Unsupervised Domain Adaptation
        source_data = source_data.view(-1,1, 9)############
        target_data = target_data.view(-1,1, 9)############
        #source_label = source_label.view(args.batch_size,-1)############

        source_data, source_label = Variable(source_data.to(device=args.device)), Variable(source_label.to(device=args.device))
        target_data = Variable(target_data.to(device=args.device))

        optimizer.zero_grad()

        out_source = model(source_data)
        out_target = model(target_data)
        #print(source_data.shape, target_data.shape, source_label.shape, out_source.shape, out_target.shape)

        classification_loss = F.cross_entropy(out_source, source_label)

        # This is where the magic happens
        coral_loss = coral(out_source, out_target)
        composite_loss = classification_loss + args.lambda_coral * coral_loss

        composite_loss.backward()
        optimizer.step()

        classification_loss_tracker.append(classification_loss.item())
        coral_loss_tracker.append(coral_loss.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(classification_loss=fmt(classification_loss_tracker.mean.value),
                       coral_loss=fmt(coral_loss_tracker.mean.value))


def evaluate(model, data_loader, dataset_name, tracker, args, epoch=0):
    model.eval()

    tracker_class, tracker_params = tracker.MeanMonitor, {}
    acc_tracker = tracker.track('{}_accuracy'.format(dataset_name), tracker_class(**tracker_params))
    fscore_tracker = tracker.track('{}_f1score'.format(dataset_name), tracker_class(**tracker_params))
    loader = tqdm(data_loader, desc='{} E{:03d}'.format('Evaluating on %s' % dataset_name, epoch), ncols=0)

    accuracies = []
    fscores = []
    with torch.no_grad():
        for target_data, target_label in loader:
            target_data = target_data.view(-1,1, 9)############
        
            target_data = Variable(target_data.to(device=args.device))
            target_label = Variable(target_label.to(device=args.device))

            output = model(target_data)

            accuracies.append(accuracy(output, target_label))
            fscores.append(Fscore(output, target_label))

            acc_tracker.append(sum(accuracies)/len(accuracies))
            fscore_tracker.append(sum(fscores)/len(fscores))
            fmt = '{:.4f}'.format
            loader.set_postfix(accuracy=fmt(acc_tracker.mean.value), f1score=fmt(fscore_tracker.mean.value))


def main():

    # Paper: In the training phase, we set the batch size to 128,
    # base learning rate to 10−3, weight decay to 5×10−4, and momentum to 0.9

    parser = argparse.ArgumentParser(description='Train - Evaluate DeepCORAL model')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--decay', default=5e-4,
                        help='Decay of the learning rate')
    parser.add_argument('--momentum', default=0.9,
                        help="Optimizer's momentum")
    parser.add_argument('--lambda_coral', type=float, default=0.5,
                        help="Weight that trades off the adaptation with "
                             "classification accuracy on the source domain")
    parser.add_argument('--source', default='source',
                        help="Source Domain (dataset)")
    parser.add_argument('--target', default='target',
                        help="Target Domain (dataset)")

    parser.add_argument('--source_eval', default='source_eval',
                        help="Source Domain (dataset)")
    parser.add_argument('--target_eval', default='target_eval',
                        help="Target Domain (dataset)")

    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    source_train_loader = get_loader(name_dataset=args.source, batch_size=args.batch_size, train=True)
    target_train_loader = get_loader(name_dataset=args.target, batch_size=args.batch_size, train=True)

    source_evaluate_loader = get_loader(name_dataset=args.source_eval, batch_size=args.batch_size, train=False)
    target_evaluate_loader = get_loader(name_dataset=args.target_eval, batch_size=args.batch_size, train=False)

    n_classes = 2############len(source_train_loader.dataset.classes)
    

    # ~ Paper : "We initialized the other layers with the parameters pre-trained on ImageNet"
    # check https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    
    model = Net()
    #print(model)
    # ~ Paper : The dimension of last fully connected layer (fc8) was set to the number of categories (31)
    model.classifier[6] = nn.Linear(128, n_classes)
    # ~ Paper : and initialized with N(0, 0.005)
    #torch.nn.init.normal_(model.classifier[6].weight, mean=0, std=5e-3)

    # Initialize bias to small constant number (http://cs231n.github.io/neural-networks-2/#init)
    #model.classifier.bias.data.fill_(0.01)

    model = model.to(device=args.device)

    # ~ Paper : "The learning rate of fc8 is set to 10 times the other layers as it was training from scratch."
    optimizer = torch.optim.SGD([
        {'params':  model.features.parameters(), 'lr': 10 * args.lr},
        {'params': model.classifier.parameters(), 'lr': 10 * args.lr},
        # fc8 -> 7th element (index 6) in the Sequential block
       
    ], lr=args.lr, momentum=args.momentum)  # if not specified, the default lr is used

    tracker = Tracker()

    for i in range(args.epochs):
        train(model, optimizer, source_train_loader, target_train_loader, tracker, args, i)
        evaluate(model, source_evaluate_loader, 'source', tracker, args, i)
        evaluate(model, target_evaluate_loader, 'target', tracker, args, i)

    # Save logged classification loss, coral loss, source accuracy, target accuracy
    torch.save(tracker.to_dict(), "log.pth")


if __name__ == '__main__':
    main()