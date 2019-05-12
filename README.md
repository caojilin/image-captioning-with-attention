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

To evaluate:  
`python eval.py --image_folder="data/val2014"`  
This will generate all captions for all images in a given folder  
The output is **references.pickle** and **hypotheses.pickle**   
You'll use these two files and `nlg-eval.py` to calculate metrics such as bleu-1 to bleu-4, meteor, rogue, etc  
`python nlg-eval.py`

Given the time permitted, this is the best   
Bleu_1: 0.647171  
Bleu_2: 0.470034  
Bleu_3: 0.330776  
Bleu_4: 0.227324  
METEOR: 0.187259  
ROUGE_L: 0.473639  
CIDEr: 0.571941  
Beam Search = 5  