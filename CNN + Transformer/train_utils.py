import torch
from utils import *
import time
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def train(train_loader, encoder, decoder, criterion,
          encoder_optimizer, decoder_optimizer, epoch,
          alpha_c, print_freq, grad_clip, n_heads , decoder_layers):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        #         print("caps",caps.shape, caps, )

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        #         print("scores", scores.shape)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        #         print(scores.shape, targets.shape)

        #         print("decode_lengths", decode_lengths)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        #         scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        #         targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        #         scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        #         targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        #         print("scores padded", scores.data.shape)

        #scores = torch.permute(scores, (0, 2, 1))
        # Calculate loss
        #         print("scores", scores.shape, "targets", targets.shape)
        #         print("targets", targets )
        loss = criterion(scores, targets)
        #         print("loss",loss )

        # Add doubly stochastic attention regularization
        dec_alphas = alphas["dec_enc_attns"]
        alpha_trans_c = alpha_c / (n_heads * decoder_layers)
        for layer in range(decoder_layers):
            cur_layer_alphas = dec_alphas[layer]
            for h in range(n_heads):
                cur_head_alpha = cur_layer_alphas[:, h, :, :]
                loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        #scores = torch.permute(scores, (0, 2, 1))
        #         print("scores", scores.shape, "targets", targets.shape)

        # Keep track of metrics
        top5 = accuracy(scores, targets,5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        #         raise Exception ("STOP")

        # Print status
        if i % print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(epoch, epochs, i, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val))


def validate(val_loader, encoder, decoder, criterion,
             alpha_c, print_freq, word_map, n_heads , decoder_layers):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            #             print("caps_sorted", caps_sorted)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            #scores_copy = scores.clone()
            #             scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            #             targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            #scores = torch.permute(scores, (0, 2, 1))
            loss = criterion(scores, targets)
            
            # Add doubly stochastic attention regularization
            dec_alphas = alphas["dec_enc_attns"]
            alpha_trans_c = alpha_c / (n_heads * decoder_layers)
            for layer in range(decoder_layers):
                cur_layer_alphas = dec_alphas[layer]
                for h in range(n_heads):
                    cur_head_alpha = cur_layer_alphas[:, h, :, :]
                    loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()


            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            #             top5 = accuracy(scores, targets, 5)

            #scores = torch.permute(scores, (0, 2, 1))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(epoch, epochs, i, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4, losses.avg