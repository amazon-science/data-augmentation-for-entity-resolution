# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import tensorflow as tf
import glob, imageio, os, time, random, torch, re, argparse, json
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import fasttext.util
from stop_words import get_stop_words
from transformers import BertTokenizer, BertModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class BaseAugmentation: 
    def __init__(self, file): 
        with np.load(file) as openfile:
            # ori_data: original text, train_data: embedding of original text
            self.train_data = np.float64(openfile['train_data']) # Match the type
            self.ori_data = openfile['ori_data']
        self.embedding_size = np.shape(self.train_data)[1]                        
        self.ori_syn_data = {}
            
    def postprocess(self, txt, embed):
        pass
    
    def GAN_train(self):
        pass


# GAN model structure    
class GANModel(BaseAugmentation):    
    def __init__(self, file, gan, epoch=30, batch_size=10, learning_rate=1e-4, dimension=6, loss_fig='GAN_Loss.png'):
        super().__init__(file)
        self.epochs = epoch
        self.batch_size = batch_size 
        self.learning_rate = learning_rate
        
        if gan == 'bart': # dimension match according to GAN
            self.dim = dimension
        else:
            self.dim = 1    
            
        self.generator = self.make_generator_model(gan) 
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1 = 0, beta_2 = 0.9) 
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1 = 0, beta_2 = 0.9)
        self.loss_fig = loss_fig
        
    def GAN_train(self):
        
        train_data = self.train_data[:len(self.train_data)//self.batch_size*self.batch_size] # Eliminate the few data for matching epoch
        train_data = np.reshape(train_data, [len(train_data), self.embedding_size, 1, self.dim])
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(self.batch_size)

        # Dummy labels. Please ignore it. 
        dummy_labels = np.ones((len(train_data),1)) # Train label
        dummy_labels = tf.data.Dataset.from_tensor_slices(dummy_labels).batch(self.batch_size)


        print('Learning Started!')
        start_time = time.time()

        gen_losses, disc_losses = self.train(train_dataset, dummy_labels) # Train start!

        print('Learning Finished!')
        print('Building time for GAN: {:.2f} seconds'.format(time.time()-start_time))

        fig = plt.figure()
        ax = plt.subplot(111)   

        ax.plot(gen_losses,'r', label='Generator')
        ax.plot(disc_losses,'g', label='Discriminator')
        plt.legend()
        plt.savefig(self.loss_fig) 
        
        
    def make_generator_model(self, gan): # Put embedded input for text

        if gan == 'fasttext': # FastText
            model = tf.keras.Sequential()

            model.add(layers.Flatten(input_shape=(self.embedding_size, self.dim)))
            model.add(layers.Dense(25*1*16, use_bias=False)) 
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            model.add(layers.Reshape((25, 1, 16)))
            assert model.output_shape == (None, 25, 1, 16)
            # Convolutional-Transpose Layer #1
            model.add(layers.Conv2DTranspose(128, (5, 1), strides=(1, 1), padding='same', use_bias=False)) # Conv-T
            assert model.output_shape == (None, 25, 1, 128)
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            # Convolutional-Transpose Layer #2
            model.add(layers.Conv2DTranspose(64, (5, 1), strides=(3, 1), padding='same', use_bias=False)) # Conv-T
            assert model.output_shape == (None, 75, 1, 64)
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            # Convolutional-Transpose Layer #3
            model.add(layers.Conv2DTranspose(32, (5, 1), strides=(2, 1), padding='same', use_bias=False)) # Conv-T
            assert model.output_shape == (None, 150, 1, 32)
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            # Convolutional-Transpose Layer #4
            model.add(layers.Conv2DTranspose(self.dim, (5, 1), strides=(2, 1), padding='same', use_bias=False)) # Conv-T    
            # Output
            assert model.output_shape == (None, self.embedding_size, 1, self.dim)

            return model

        else: # BERT / BART
            model = tf.keras.Sequential()

            model.add(layers.Flatten(input_shape=(self.embedding_size, self.dim)))
            model.add(layers.Dense(32*1*16, use_bias=False)) 
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            model.add(layers.Reshape((32, 1, 16)))
            assert model.output_shape == (None, 32, 1, 16)
            # Convolutional-Transpose Layer #1
            model.add(layers.Conv2DTranspose(128, (5, 1), strides=(2, 1), padding='same', use_bias=False)) # Conv-T
            assert model.output_shape == (None, 64, 1, 128)
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            # Convolutional-Transpose Layer #2
            model.add(layers.Conv2DTranspose(64, (5, 1), strides=(3, 1), padding='same', use_bias=False)) # Conv-T
            assert model.output_shape == (None, 192, 1, 64)
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            # Convolutional-Transpose Layer #3
            model.add(layers.Conv2DTranspose(32, (5, 1), strides=(2, 1), padding='same', use_bias=False)) # Conv-T
            assert model.output_shape == (None, 384, 1, 32)
            model.add(layers.BatchNormalization()) # BN
            model.add(layers.LeakyReLU()) # LeakyReLu
            # Convolutional-Transpose Layer #4
            model.add(layers.Conv2DTranspose(self.dim, (5, 1), strides=(2, 1), padding='same', use_bias=False)) # Conv-T    
            # Output
            assert model.output_shape == (None, self.embedding_size, 1, self.dim)

            return model


    def make_discriminator_model(self):
        model = tf.keras.Sequential()

        # Convolutional Layer #1    
        model.add(layers.Conv2D(32, (5, 1), strides=(2, 1), padding='same', input_shape=[self.embedding_size, 1, self.dim])) # Conv
        model.add(layers.LeakyReLU()) # LeakyReLu
        model.add(layers.Dropout(0.5)) # Dropout
        # Convolutional Layer #2
        model.add(layers.Conv2D(64, (5, 1), strides=(2, 1), padding='same')) # Conv
        model.add(layers.LeakyReLU()) # LeakyReLu
        model.add(layers.Dropout(0.5)) # Dropout
        # Convolutional Layer #3
        model.add(layers.Conv2D(128, (5, 1), strides=(3, 1), padding='same')) # Conv
        model.add(layers.LeakyReLU()) # LeakyReLu
        model.add(layers.Dropout(0.5)) # Dropout
        # Convolutional Layer #4
        model.add(layers.Conv2D(256, (5, 1), strides=(2, 1), padding='same')) # Conv
        model.add(layers.LeakyReLU()) # LeakyReLu
        model.add(layers.Dropout(0.5)) # Dropout
        # FC Layer + Output
        model.add(layers.Flatten())
        model.add(layers.Dense(1))    

        return model


    def generator_loss(self, fake_output):

        original_loss = tf.cast(-tf.reduce_mean(fake_output), dtype=tf.float64) # Loss for WGAN

        return original_loss


    def discriminator_loss(self, real_data, fake_data): 
        # Calculate the gradient penalty term 
        LAMBDA = 10
        alpha = tf.cast(tf.random.uniform([self.batch_size, self.embedding_size, 1, 1], 0.,1.), dtype=tf.float64)
        real_data = tf.cast(real_data, dtype=tf.float64)
        fake_data = tf.cast(fake_data, dtype=tf.float64)    
        interpolates = alpha * real_data + (1-alpha) * fake_data

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolates)
            pred = self.discriminator(interpolates)

        gradients = gp_tape.gradient(pred, [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))

        gradient_penalty = tf.reduce_mean((slopes-1)**2) # Gradient penalty term

        real_output = self.discriminator(real_data, training=True)
        fake_output = self.discriminator(fake_data, training=True)

        wasserstein_dist = tf.cast(tf.reduce_mean(fake_output) - tf.reduce_mean(real_output), dtype=tf.float64) # Loss for WGAN 

        return wasserstein_dist + LAMBDA*gradient_penalty # Loss with gradient penalty term



    def train_step(self, images, epoch):  

        for _ in range(5): # Train discriminator 5 times more than generator

            noise = tf.random.normal([self.batch_size, self.embedding_size, self.dim]) # Input for generator

            images_noise = tf.reshape(images, [self.batch_size, self.embedding_size, self.dim])        
            noise = tf.cast(noise, tf.float64)
            noise = tf.add(noise, images_noise)

            with tf.GradientTape() as disc_tape:

                generated_images = self.generator(noise, training=True) # Synthetic data      

                disc_loss = self.discriminator_loss(images, generated_images) # Loss from discriminator
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables)) # Optimizer for discriminator     


        noise = tf.random.normal([self.batch_size, self.embedding_size, self.dim]) # Input for generator

        noise = tf.cast(noise, tf.float64)
        noise = tf.add(noise, images_noise)

        with tf.GradientTape() as gen_tape:        
            generated_images = self.generator(noise, training=True) # Synthetic data            

            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output) # Loss from generator
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables)) # Optimizer for generator


        return gen_loss, disc_loss
   

    def train(self, dataset, labels): # Train during epoch
        losses_gen = []
        losses_disc = []
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
    
        start = time.time()
        for epoch in range(self.epochs): 
            for image_batch, true_label in zip(dataset, labels):           
                gen_loss, disc_loss = self.train_step(image_batch, epoch)

            losses_gen.append(gen_loss)
            losses_disc.append(disc_loss)                                
            if epoch%10 == 0:            
                print ('Time for epoch {} is {:.2f} sec'.format(epoch + 1, time.time()-start))
                # Save the model every 10 epochs
                checkpoint.save(file_prefix = checkpoint_prefix)
                start = time.time()

        return losses_gen, losses_disc
         
    
# Generate synthetic data by FastText
class FastTextGAN(BaseAugmentation):
    def __init__(self, file, model_loc, generator, ft_score=1.3):   
        super().__init__(file)
        self.ft = fasttext.load_model(model_loc)
        self.dic_data = np.zeros((len(self.ft.words), self.embedding_size)) # Make all words in FastText to embedding

        for i in range(len(self.ft.words)):
            self.dic_data[i] = self.ft.get_word_vector(self.ft.words[i])   

        self.dic_data_norm = np.zeros(np.shape(self.dic_data)) # Calculate the norm
        for i in range(len(self.dic_data)):
            self.dic_data_norm[i] = self.dic_data[i] / np.linalg.norm(self.dic_data[i])       
        self.score = ft_score
        self.generator = generator

        
    def postprocess(self, txt, embed):
        
        original_norm = embed / np.linalg.norm(embed) # Calculate the norm
        original_norm = np.reshape(original_norm, [len(original_norm)])

        # Skip the duplicate if it already has more than 3 synthetic data
        if (txt not in self.ori_syn_data.keys()) or len(self.ori_syn_data[txt])<3: 
            if txt not in self.ori_syn_data.keys():
                self.ori_syn_data[txt] = []       

            input_data = np.reshape(embed, [1, self.embedding_size])
            flag = 0

            while len(self.ori_syn_data[txt])< 3: # # of synthetic data for single word
                if flag > 3: # Try 3 times for finding synthetic data until max #
                    break
                flag +=1


                noise = tf.random.normal([1, self.embedding_size]) # Random noise
                input_syn = input_data + noise # Input for generator
                tmp_data = self.generator(input_syn, training=False) # Target's synthetic data
                tmp_data = np.reshape(tmp_data,[self.embedding_size])            

                syn_norm = tmp_data / np.linalg.norm(tmp_data)  
                angle = np.arccos(np.matmul(self.dic_data_norm, syn_norm))
                min_idx = angle.argsort()[:5] #The smaller the angle, higher the cosine similarity.    
                for j in range(len(min_idx)):                                
                    syn_min_norm = self.ft.get_word_vector(self.ft.words[min_idx[j]])
                    syn_min_norm = syn_min_norm / np.linalg.norm(syn_min_norm)

                    # Check whether it is below threshold. It is important to find the proper synthetic data                            
                    if (np.arccos(np.dot(original_norm, syn_min_norm)) < self.score): 
                        if len(self.ori_syn_data[txt])<3:
                            self.ori_syn_data[txt].append(self.ft.words[min_idx[j]])
                        else:
                            break
                
# Generate synthetic data by BERT               
class BERTGAN(BaseAugmentation):
    def __init__(self, file, dict_bert, model_loc, generator):   
        super().__init__(file)
        with np.load(dict_bert) as openfile:
            self.dict_embed = openfile['train_data']
            self.dict_data = openfile['ori_data']
        self.tokenizer = BertTokenizer.from_pretrained(model_loc)
        self.model = BertModel.from_pretrained(model_loc, output_hidden_states = True)
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()          
        self.generator = generator

    def postprocess(self, txt, embed): 
        
        if (txt not in self.ori_syn_data.keys()):        

            input_data = np.reshape(embed, [1, self.embedding_size])
            noise = tf.random.normal([1, self.embedding_size]) # Random noise                    
            input_syn = input_data + noise # Input for generator
            tmp_data = self.generator(input_syn, training=False) # Target's synthetic data
            tmp_data = np.reshape(tmp_data, [self.embedding_size])  

            val_1 = -100
            val_2 = -100
            val_3 = -100
            loc_1 = -1
            loc_2 = -1
            loc_3 = -1

            original_norm = tmp_data / np.linalg.norm(tmp_data) # norm of synthetic data
            
            for j in range(len(self.dict_embed)):
                candidate = np.reshape(self.dict_embed[j], [self.embedding_size])
                candidate_norm = candidate / np.linalg.norm(candidate)
                similar = np.dot(original_norm, candidate_norm)

                # Find the top 3 similar 
                if val_1 < similar:
                    val_3 = val_2
                    val_2 = val_1
                    val_1 = similar
                    loc_3 = loc_2
                    loc_2 = loc_1
                    loc_1 = j 
                elif val_2 < similar:
                    val_3 = val_2
                    val_2 = similar
                    loc_3 = loc_2
                    loc_2 = j
                elif val_3 < similar:
                    val_3 = similar
                    loc_3 = j
            self.ori_syn_data[txt] = [self.dict_data[loc_1],
                                               self.dict_data[loc_2],self.dict_data[loc_3]]


# Generate synthetic data by BART    
class BARTGAN(BaseAugmentation):
    def __init__(self, file, model_loc, generator, dimension=6, bart_score_low=0.3, bart_score_high=0.7):   
        super().__init__(file)
        self.tokenizer = AutoTokenizer.from_pretrained(model_loc)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_loc)
        self.encoder_embed = self.model.state_dict()['model.encoder.embed_tokens.weight'] # encoder embedding
        self.model.eval()         
        self.dim = dimension
        self.low = bart_score_low  
        self.high = bart_score_high
        self.generator = generator
        
    def postprocess(self, txt, embed): 
                                        
        if (txt not in self.ori_syn_data.keys()):   

            if len(np.where(embed[0]==0)[0])!= 0: # Find the dimensions until considered tokens
                target_len = np.where(embed[0]==0)[0][0]
            else:    
                target_len = self.dim

            self.ori_syn_data[txt] = []
            output_loc = []

            input_data = np.reshape(embed, [1, self.embedding_size, self.dim]) 
            noise = tf.random.normal([1, self.embedding_size,self.dim]) # Random noise
            input_syn = input_data + noise # Input for generator

            tmp_data = self.generator(input_syn, training=False) # Target's synthetic embedding
            # Average between synthetic and original data for proper augmentation
            tmp_data = (np.reshape(tmp_data,[self.embedding_size, self.dim]) + embed) /2             

            for k in range(target_len):
                original_norm = tmp_data[:,k] / np.linalg.norm(tmp_data[:,k])                    

                for jj in range(len(self.encoder_embed)):                                
                    candidate = np.reshape(self.encoder_embed[jj], [self.embedding_size])
                    candidate_norm = candidate / np.linalg.norm(candidate)
                    similar = np.dot(original_norm, candidate_norm)

                    # Define threshold for comparison
                    if similar > self.low and similar < self.high: 
                        output_loc.append([k, jj])

            output_txt = [[] for _ in range(self.dim)] # Generate the possible combination
            for kk in range(len(output_loc)):
                output_txt[output_loc[kk][0]].append(self.tokenizer.decode(output_loc[kk][1], 
                                                                      skip_special_tokens=True))                            

            flag = 0
            for kk in range(target_len):
                if len(output_txt[kk]) == 0: # thresholds are tough and there is no similar word
                    tmp_wrd = ''
                    flag = 1

            if flag == 0:                

                for _ in range(3): # generate 3 synthetic data for each sentence
                    tmp_wrd = ''
                    for k in range(target_len):
                        tmp_wrd += random.sample(output_txt[k], 1)[0]
                        if k != target_len-1:
                            tmp_wrd += ' '

                    self.ori_syn_data[txt].append(tmp_wrd)
            else: 
                self.ori_syn_data[txt].append(tmp_wrd) # There is no synthetic data for this one.

                    
def main(args):
 
    model = GANModel(args.input, args.gan, args.epoch, args.batch_size, args.learning_rate, 
                     args.dimension, args.loss_fig)
    model.GAN_train()
    generator = model.generator
    
    data = None

    if args.gan == 'fasttext':
        data = FastTextGAN(args.input, args.model_loc, generator, args.ft_score)
    elif args.gan == 'bert':
        data = BERTGAN(args.input, args.dict_bert, args.model_loc, generator)
    elif args.gan == 'bart':
        data = BARTGAN(args.input, args.model_loc, generator, args.dimension, args.bart_score_low,
                       args.bart_score_high)
        
    if data is None:
        print('Wrong GAN method. Please try one of fasttext / bert / bart')
    
    else:
        # Generate synthetic data        
        for text, embedding in zip(data.ori_data, data.train_data):
            data.postprocess(text, embedding)

        # Release
        del data.train_data
        del data.ori_data        
        
        if args.gan == 'bert':
            del data.dict_embed
            del data.dict_data
            
        with open(args.output, 'w') as outfile:
            json.dump(data.ori_syn_data, outfile)
        


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='GAN training and generating synthetic data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True, help='Original embedding data. Only receive npz file.')       
    parser.add_argument('--gan', type=str, required=True, help='Choose GAN methods: fasttext / bert / bart.')                
    parser.add_argument('--output', type=str, required=True, help='Name of output file. Only generate json file.')
    parser.add_argument('--model-loc', type=str, required=True, help='Pre-trained model location')    
    parser.add_argument('--dict-bert', type=str, help='Dictionary for BERT from translated traffic')
    parser.add_argument('--batch-size', type=int, default = 10, help='Batch size used in GAN')
    parser.add_argument('--epoch', type=int, default = 30, help='Number of epochs used in GAN')
    parser.add_argument('--learning-rate', type=float, default = 1e-4, help='Learning rate used in GAN')
    parser.add_argument('--ft-score', type=float, default = 1.3, help='Threshold for measuring similarity in FastText-GAN')
    parser.add_argument('--bart-score-low', type=float, default = 0.3, help='Low threshold for measuring similarity in BART-GAN')
    parser.add_argument('--bart-score-high', type=float, default = 0.7, help='High threshold for measuring similarity in BART-GAN')    
    parser.add_argument('--dimension', type=int, default = 6, help='Number of tokens considered in BART-GAN')
    parser.add_argument('--loss-fig', type=str, default = 'GAN_Loss.png', help='Name of loss figure')

    args = parser.parse_args()
    main(args)    
