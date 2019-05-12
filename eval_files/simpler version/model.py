import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    # def __init__(self, embed_size, model_name):
    #     """Load the pretrained ResNet-152 and replace top fc layer."""
    #     super(EncoderCNN, self).__init__()
    #     resnet = models.resnet152(pretrained=True)
    #     modules = list(resnet.children())[:-1]  # delete the last fc layer.
    #     self.resnet = nn.Sequential(*modules)
    #     self.linear = nn.Linear(resnet.fc.in_features, embed_size)
    #     self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    # def forward(self, images):
    #     """Extract feature vectors from input images."""
    #     with torch.no_grad():
    #         features = self.resnet(images)
    #     features = features.reshape(features.size(0), -1)
    #     features = self.bn(self.linear(features))
    #     return features
    
    def __init__(self, embed_size, model_name):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        if model_name == 'resnet':
            self.model = models.resnet101(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, embed_size)
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
            # self.fine_tune()
        elif model_name == 'vgg':
            #VGG
            self.model = models.vgg16(pretrained=True)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,embed_size)
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
            # self.fine_tune()
        elif model_name == "squeezenet":
            #squeezenet
            self.model = models.squeezenet1_0(pretrained=True)
            self.model.classifier[1] = nn.Conv2d(512, embed_size, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = embed_size
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
            # self.fine_tune()
        elif model_name == "densenet":
            #densenet
            self.model = models.densenet121(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, embed_size)
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model = models.inception_v3(pretrained=True)
            self.model.aux_logit=False
    
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, embed_size)
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
            # self.fine_tune()
    
        else:
            print('Invalid model name, exiting...')
            exit()
    
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.model(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(features)
        return features
    
    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.model.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids