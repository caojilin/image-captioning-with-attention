Prepare Data:  
Download COCO, Flickr8K, or Flickr30k datasets  
Download [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)  
Then run   
`python create_input_files.py`

To train:  
You can train vgg, resnet, squeezenet as encoder 
The default is resnet101   
`python train.py`

To caption a image:  
`python caption.py --img="img/1.jpg" --model=Best_squeezeNet.pth.tar`

demo video : [demo video](https://www.youtube.com/watch?v=EldYl3xzvqk&feature=youtu.be) 

To evaluate:  
`python eval.py --image_folder="data/val2014"`  
This will generate all captions for all images in a given folder  
The output is **references.pickle** and **hypotheses.pickle**   
You'll use these two files and `nlg-eval.py` to calculate metrics such as bleu-1 to bleu-4, meteor, rogue, etc  
`python nlg-eval.py`

Given the time permitted, this is the best   
Bleu_1: 0.732558  
Bleu_2: 0.566442  
Bleu_3: 0.433188  
Bleu_4: 0.332858  
METEOR: 0.260003  
ROUGE_L: 0.545065  
CIDEr: 1.041811  
Beam search =5  

sampling speed, it means how many images per second  
beam size = 1,2,3,4,5  
19.73it/s, 16.02it/s, 14.81it/s, 13.85it, 12.39it/s

Download [pretrained models](https://drive.google.com/drive/folders/12ndT_fOsoXrtPQnkXADNNSiG_ziYVQm0?usp=sharing)