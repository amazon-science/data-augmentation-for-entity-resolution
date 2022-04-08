# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch, re, argparse, json
import numpy as np
import tensorflow as tf
import fasttext.util
from stop_words import get_stop_words
from transformers import BertTokenizer, BertModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class BaseAugmentation: 
    def __init__(self, file):        
        with open(file, 'r') as openfile:
            self.ori_data = json.load(openfile)
        
        self.embedding_data = []        
        self.text_data = []

    def preprocess(self, txt):
        pass

            
class FastTextGAN(BaseAugmentation):
    def __init__(self, file, model_loc, ft_lang='french'):
        super().__init__(file)
        self.ft = fasttext.load_model(model_loc)
        self.stop_lst = get_stop_words(ft_lang)
        
    def preprocess(self, txt):
        # Eliminate stopwords / number / punctuation and then, make it as single word
        # Remove stopwords
        tmp = txt
        result = ''
        loc = 0
        flag = 0
        for i in range(len(tmp)):
            if tmp[i]==' ': 
                if tmp[loc:i] not in self.stop_lst: # Check stopwords before whitespace
                    result += tmp[loc:i]
                    result += ' '
                    flag = 1
                loc = i+1
            if flag == 1 and i == len(tmp)-1: # If there was a space
                if tmp[loc:i] not in self.stop_lst: # Check whether the last word has stopwords
                    result += tmp[loc:]
            if flag == 0 and i == len(tmp)-1: # There is no stopwords
                result = tmp

        # Remove number
        result = re.sub(r'\d+', '', result) 
        # Remove punctuation
        result = re.sub(r'[^\w\s]','', result) 
        # Remove whitespace or not
        result = result.split()
        
        # Separate data according to whitespace and find embedding from FastText
        for i in range(len(result)):
            self.text_data.append(result[i])
            self.embedding_data.append(self.ft.get_word_vector(result[i]))

            
class BERTGAN(BaseAugmentation):
    def __init__(self, file, model_loc):
        super().__init__(file)
        self.tokenizer = BertTokenizer.from_pretrained(model_loc)
        self.model = BertModel.from_pretrained(model_loc, output_hidden_states = True)
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()        

    def preprocess(self, txt):    
                       
        marked_text = "[CLS] " + txt + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers. 
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # Concatenate the tensors for all layers.
        token_embeddings = torch.stack(hidden_states, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        # Average the second to last hidden layer of each token producing a single 768 length vector
        # `token_vecs` is a tensor with shape [-1 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        self.text_data.append(txt)
        self.embedding_data.append(np.array(sentence_embedding))
           

class BARTGAN(BaseAugmentation):
    def __init__(self, file, model_loc, dimension=6, hidden=768):
        super().__init__(file)
        self.tokenizer = AutoTokenizer.from_pretrained(model_loc)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_loc)
        self.encoder_embed = self.model.state_dict()['model.encoder.embed_tokens.weight'] # encoder embedding
        self.model.eval()
        self.dimension = dimension
        self.hidden = hidden

    def preprocess(self, txt):
        id_info = []

        # Exclude first and last, meaning the start and end of sentence
        input_ids = torch.tensor([self.tokenizer.encode(txt, add_special_tokens=True)])[0][1:-1]
        if len(input_ids) <= self.dimension: # Include data with tokens less than the considered tokens
            
            train_embed = np.zeros((self.hidden, self.dimension)) # 
        
            for ii in range(len(input_ids)):   

                train_embed[:,ii] = self.encoder_embed[input_ids[ii]]
       
            self.text_data.append(txt)
            self.embedding_data.append(train_embed)
           
                   
    
def main(args):
    data = None
        
    if args.gan == 'fasttext':
        data = FastTextGAN(args.input, args.model_loc, args.ft_lang)
    elif args.gan == 'bert':
        data = BERTGAN(args.input, args.model_loc)
    elif args.gan == 'bart':
        data = BARTGAN(args.input, args.model_loc, args.dimension, args.hidden)


    if data is None:
        print('Wrong GAN method. Please try one of fasttext / bert / bart')
    
        
    else:
        # Generate synthetic data
        for text in data.ori_data:            
            data.preprocess(text)
            
                
        np.savez(args.output, train_data = data.embedding_data, ori_data = data.text_data)

        
            

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing for GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True, help='Original text data. Only receive json file.')       
    parser.add_argument('--gan', type=str, required=True, help='Choose GAN methods: fasttext / bert / bart.')                
    parser.add_argument('--output', type=str, required=True, help='Name of output file. Only generate npz file.')
    parser.add_argument('--model-loc', type=str, required=True, help='Pre-trained model location.')    
    parser.add_argument('--ft-lang', type=str, default= 'french', help='Language used in FastText')
    parser.add_argument('--dimension', type=int, default = 6, help='Number of tokens considered in BART-GAN')
    parser.add_argument('--hidden', type=int, default = 768, help='Hidden unit for BART-GAN.')

    

    args = parser.parse_args()
    main(args)
    