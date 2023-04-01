import numpy as np
import random 

import matplotlib.image
from PIL import Image as img
from scipy import signal as sg
import scipy.misc
from perlin_noise import PerlinNoise
import random 
import cv2


SIZE = 80
class Chromosome:
    # track next chrom ID
    id_count = 0
   
    def __init__(self , genes):
        self.genes = genes
        self.gene_count = len(self.genes)
        #Variables used to calculate fitness
        self.size = 80
        self.content_image = r"C:\\Users\\RileyHall\\Documents\\Applied AI\\final\\resizedImages\\0@9L6C8Y7J9U.jpg"
        self.style_image = r"C:\\Users\\RileyHall\\Documents\\Applied AI\\final\\resizedImages\\0A3O3S3H9D1Q.jpg"
        # assigns unique ID to each chrom
        self.chrom_index = Chromosome.next_id()

    # generate a new Chromosome with random genes
    @classmethod
    def randomize(cls, gene_count):
        genes = [random.randint(0, 1) for _ in range(gene_count)]
        return cls(genes, gene_count)

    # increment id for each new chromosome
    @classmethod
    def next_id(cls):
        cls.id_count += 1
        return cls.id_count

    # calculate the fitness score (number of 1s in the genes)
    def get_fitness(self):
        style_loss = self.style_loss()
        
        content_loss = self.content_loss()
        
        return content_loss + style_loss

    # format the chromosome genes and fitness score as a string
    def to_string(self):
        return "Chromosome: " + str(self.chrom_index) \
            + " Genes: " + ''.join(str(i) for i in self.genes) \
            + " Fitness: " + str(self.get_fitness())
        


#Generate intial generations
    @classmethod
    def generate(self): 
        seed = random.randint(1,100)
        
        noise = PerlinNoise(octaves=10, seed=seed)
        # image = img.open(self.content_image)
        # image_rgb = list(image.getdata())   
        
        # self.individual.genes = [[noise([i/len(image_rgb),j/len(image_rgb)]for j in range(len(image_rgb)))]for i in range(len(image_rgb))]
        return [[noise([i/SIZE, j/SIZE]) for j in range(SIZE)] for i in range(SIZE)] 


#RILEY HALL CODE,,, ALL LOSS FUNCTIONS 

    def content_loss(self): 

        new_genes_arr = []
       
        for i in range(len(self.genes)): 

            if(type(self.genes[i])!=float):
                new_genes_arr.append(self.genes[i])

        #grey scale the images 
        content_image = img.open(self.content_image)
        content_image_rgb = list(content_image.getdata())


        grey_content = np.dot(np.array(content_image_rgb)[...,:3],[0.2989, 0.5870, 0.1140])
        # grey_content = np.dot(image_rgb,[0.299, 0.587, 0.114])[...,None]
        generated = np.dot(np.array(new_genes_arr)[...,:3], [0.2989, 0.5870, 0.1140])
        #Root Mean Squared Error
        # print(len(grey_content) , len(generated))
        # print(grey_content , generated)

        if(len(generated)>len(grey_content)): 
            end1 = len(generated)-len(grey_content)
            for i in range(len(end1)):
                grey_content = np.append(grey_content,0)
        elif(len(generated)<len(grey_content)):
            end = len(grey_content)-len(generated)
            
            for j in range(end):
                
                generated=np.append(generated,0)
            
        
        loss = np.mean((grey_content-generated)**2)
        

        return loss
    




    #STYLE LOSS FUNCTIONS




    #LAYER CREATION FUNCTIONS

    def get_gaussian_filter(self,sigma): 

        ksize = int(2*np.ceil(2*sigma)+1)
        kernel = cv2.getGaussianKernel(ksize,sigma)

        return np.outer(kernel,kernel)
    
    def get_gaussian_pyramid(self, image , levels): 

        pyramid = [image.astype(np.float32)]

        for i in range(1,levels): 
            sigma = 2**i
            blurred = cv2.filter2D(pyramid[-1],-1,self.get_gaussian_filter(sigma))
            downsampled = blurred[::2,::2]
            pyramid.append(downsampled)

        return pyramid
    
    def define_layers(self, input_image , num_layers): 
        # print(len(input_image) , "IMAGE SIZE ARRAY")
        # print(type(input_image))
     
        # for i in range(len(input_image)): 
        #     try:
        #         print(len(input_image[i]))
        #         new_arr.append(input_image[i])

        #     except: 
        #         print(input_image[i])

        _numpy = np.asarray(input_image)

        pyramid = self.get_gaussian_pyramid(_numpy,num_layers)

        layers = []
        
        for i in range(num_layers):
            layer_name = f'gaussian_{i}'
            layer_output = pyramid[i]
            layers.append((layer_name , layer_output))

        return layers



    #STYLE LOSS FUNCTIONS
    #Stle
    #SADSADASDADS
    #Using Gram matrix to calculate style loss allows us to find the distance between the vectors 
    def style_loss(self): 
        input_grams  = [] 
        style_grams = []

        img2 = img.open(self.style_image)

        # input_image = np.array(self.genes,dtype=np.float32)/255.0

        # style_image = list(img2.getdata())
        style_image = np.array(img2,dtype=np.float32)/255.0
        
        new_genes_arr = []
      
        for i in range(len(self.genes)): 
            if type(self.genes[i]) != float:
                # print(len(self.genes[i]))
                new_genes_arr.append(self.genes[i])
      
                

        layers = self.define_layers(new_genes_arr , 3)
        
        
        for layer in layers: 
            # input_feature_map = cv2.imread(r"C:\\Users\\RileyHall\\Documents\\Applied AI\\final\\output23jpg.jpg", cv2.IMREAD_GRAYSCALE) 0.917694778002361 0.917694778002361

            input_feature_map = cv2.normalize(np.array(new_genes_arr),None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            style_feature_map = cv2.imread(self.style_image, cv2.IMREAD_GRAYSCALE)
            # input_feature_map = np.dot(np.array(input_image)[...,:3],[0.2989, 0.5870, 0.1140])
            # style_feature_map = np.dot(np.array(style_image)[...,:3],[0.2989, 0.5870, 0.1140])

            input_feature_map = np.array(input_feature_map,dtype=np.float32)/255.0
            style_feature_map = np.array(style_feature_map,dtype=np.float32)/255.0
            # print(input_feature_map)
            # print(input_feature_map.ndim)
            if(input_feature_map.ndim==1):
                length = input_feature_map.shape[0]

                h,w =1,length
            else:
                h, w = input_feature_map.shape[:2]
            style_feature_map = cv2.resize(style_feature_map,(h,w))

            input_grams.append(self.gram_matrix(np.expand_dims(input_feature_map,1)))
            style_grams.append(self.gram_matrix(np.expand_dims(style_feature_map,1)))
        
        loss = 0

        for i in range(len(input_grams)):
            _,c=input_grams[i].shape
            size = c*c
            if(input_grams[i].shape>style_grams[i].shape):
                displace = input_grams[i].size-style_grams[i].size
                # print("SIZE" , len(input_grams[i])-len(style_grams[i]))
                # print(input_grams[i])
                for j in range(displace):
                    
                    style_grams[i] = np.append(style_grams[i] , 0)
                
                style_grams[i]=style_grams[i].reshape(input_grams[i].shape)

            loss+=np.sum((input_grams[i]-style_grams[i])**2)/size

        return loss
    

    #The gram matrix is created by multiplying its self with the Transposed (replacing the rows with columns and columns with the rows) version of its self
    #This will allow us to transfer the style across the style image 
    def gram_matrix(self,feature_map):
        h,w,c = feature_map.shape
        flattend = np.reshape(feature_map,(h*w,c))
        gram = np.dot(flattend.T,flattend)

        return gram 
    
