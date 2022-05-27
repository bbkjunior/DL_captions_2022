import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--device', type=str)
args = parser.parse_args()

from datasets import CaptionDataset
from models import *
from utils import *
from train_utils import *
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
import json

#os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data parameters
main_folder = "./a-PyTorch-Tutorial-to-Image-Captioning"

# data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
data_folder = f'{main_folder}/flickr_prep/'  # folder with data files saved by create_input_files.py
# data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
word_map_file = f'{main_folder}/flickr_prep/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json'
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 10  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)

decoder = decoder.to(device)

decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                     lr=decoder_lr)
encoder = Encoder()
encoder.fine_tune(fine_tune_encoder)
encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                     lr=encoder_lr) if fine_tune_encoder else None
encoder = encoder.to(device)

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index = 0).to(device)

# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

save_name = 'enlarge_resnet.json'
training_track = {'loss':[],'bleu':[]}

# Epochs
for epoch in range(start_epoch, epochs):

    # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
    if epochs_since_improvement == 20:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        adjust_learning_rate(decoder_optimizer, 0.8)
        if fine_tune_encoder:
            adjust_learning_rate(encoder_optimizer, 0.8)

    # One epoch's training
    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch,alpha_c=alpha_c,
          print_freq=print_freq, grad_clip=grad_clip)

    # One epoch's validation
    recent_bleu4, loss = validate(val_loader=val_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            alpha_c=alpha_c,
                            print_freq=print_freq,
                            word_map=word_map)

    training_track['bleu'].append(recent_bleu4)
    training_track['loss'].append(loss)
    with open(save_name, 'w') as f:
        json.dump(training_track, f)

    # Check if there was an improvement
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        epochs_since_improvement = 0

    # Save checkpoint
    save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, recent_bleu4, is_best)
                    